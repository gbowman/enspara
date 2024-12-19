import os
import logging
import math

import multiprocessing as mp
from contextlib import closing
from functools import partial, reduce
import ctypes
from operator import mul

import numpy as np
import mdtraj as md

from .. import exception

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sound_trajectory(trj, stride=1, frame=None):
    """Determine the length of a trajectory on disk.

    For H5 file formats, this is a trivial lookup of the shape parameter
    tracked by the HDF5 file system. For other file formats, a binary
    search-like system to figure out how long a trajectory on disk is
    in (maybe) log(n) time and constant space by loading individual
    frames from disk at exponentially increasing indices.

    Additional keyword args are passed on to md.load.

    Parameters
    ----------
    trj: file path
        Path to the trajectory to sound.

    Returns
    ----------
    length: int
        The length (in frames) of a trajectory on disk, as though loaded
        with kwargs

    See Also
    ----------
    md.load
    """
    with md.open(trj) as f:
        n_frames = len(f)

    return math.ceil(n_frames / stride)


def load_as_concatenated(filenames, lengths=None, processes=None,
                         args=None, **kwargs):
    '''Load many trajectories from disk into a single numpy array.

    Additional arguments to md.load are supplied as ``*args`` XOR
    ``**kwargs``. If ``*args`` are supplied, args and filenames must be
    of the same length and the ith arg is applied as the kwargs to the
    md.load (e.g. top, selection) for the ith file. If ``**kwargs`` are
    specified, all are passed as keyword args to all calls to md.load.

    Parameters
    ----------
    filenames : list
        A list of relative paths to the trajectory files to be loaded.
        The md.load function is used, and all file types md.load
        supports are supported by this function.
    lengths : list, optional, default=None
        List of lengths of the underlying trajectories. If None, the
        lengths will be inferred. However, this can be slow, especially
        as the number of trajectories grows large. This option provides
        a speed benefit only.
    processes : int, optional
        The number of processes to spawn for loading in parallel.
    args : list, optional
        A list of dictionaries, each of which corresponds to additional
        kwargs to be passed to each of filenames.

    Returns
    -------
    (lengths, xyz) : tuple
       A 2-tuple of trajectory lengths (list of ints, frames) and
       coordinates (ndarray, shape=(n_atoms, n_frames, 3)).

    See Also
    --------
    md.load
    '''

    # we need access to this as a list, so if we get some kind of
    # wierd iterator we need to build a list out of it
    filenames = list(filenames)

    # configure arguments to md.load
    if kwargs and args:
        raise exception.ImproperlyConfigured(
            "Additional unnamed args can only be supplied iff no "
            "additonal keyword args are supplied")
    elif kwargs:
        args = [kwargs] * len(filenames)
    elif args:
        if len(args) != len(filenames):
            raise exception.ImproperlyConfigured(
                "When add'l unnamed args are provided, len(args) == "
                "len(filenames), but %s != %s." % (len(args), len(filenames)))
    else:  # not args and not kwargs
        args = [{}] * len(filenames)

    logger.debug(
        "Configuring load calls with args[0] == [%s ... %s]",
        args[0], args[-1])

    if lengths is None:
        logger.debug("Sounding %s trajectories with %s processes.",
                     len(filenames), processes)
        with mp.Pool(processes=processes) as pool:
            proc = pool.starmap_async(
                sound_trajectory,
                [(f, kw.get('stride', 1)) for f, kw
                 in zip(filenames, args) if 'frame' not in kw])

        shapes = proc.get()

        # trjs with frame are always length 1, add that to lengths now
        for i, kw in enumerate(args):
            if 'frame' in kw:
                lengths.insert(i, 1)
    else:
        logger.debug("Using given lengths")
        if len(lengths) != len(filenames):
            raise exception.ImproperlyConfigured(
                "Lengths list (len %s) didn't match length of filenames"
                " list (len %s)", len(lengths), len(filenames))

    tmp_args = dict(args[0])
    if 'frame' in tmp_args: del tmp_args['frame']
    full_shape, shared_array = shared_array_like_trj(
        lengths, example_trj=md.load(filenames[0], frame=0, **tmp_args))

    logger.debug("Allocated array of shape %s", full_shape)

    with closing(mp.Pool(processes=processes, initializer=_init,
                         initargs=(shared_array,))) as p:
        proc = p.map_async(
            partial(_load_to_position, arr_shape=full_shape),
            zip([sum(lengths[0:i]) for i in range(len(lengths))],
                filenames, args))

    # gather exceptions.
    shapes = proc.get()

    if sum(s[0] for s in shapes) != full_shape[0]:
        raise exception.DataInvalid(
            "The provided lengths (n=%s, total frames %s) weren't correct. "
            "The correct total number of frames was %s.", len(lengths),
            sum(s[0] for s in shapes), full_shape[0])

    # wait for termination
    p.join()

    xyz = _tonumpyarray(shared_array).reshape(full_shape)

    return lengths, xyz


def concatenate_trjs(trj_list, atoms=None, n_procs=None):
    """Convert a list of trajectories into a single trajectory building
    a concatenated array in parallel.

    Parameters
    ----------
    trj_list : array-like, shape=(n_trjs,)
        The list of md.Trajectory objects
    atoms : str, default=None
        A selection in the MDTraj DSL for atoms to slice out from each
        trajectory in `trj_list`. If none, no slice is performed.
    n_procs : int, default=None
        Number of parallel processes to use when performing this
        operation.

    Returns
    -------
    trj : md.Trajectory
        A concatenated trajectory
    """

    example_center = trj_list[0]
    if atoms is not None:
        example_center = example_center.atom_slice(
            example_center.top.select(atoms))

    lengths = [len(t) for t in trj_list]
    intervals = np.cumsum(np.array([0] + lengths))
    full_shape, shared_xyz = shared_array_like_trj(lengths, example_center)

    with closing(mp.Pool(processes=n_procs, initializer=_init,
                         initargs=(shared_xyz,))) as p:
        func = partial(_slice_and_insert_xyz, atoms=atoms,
                       arr_shape=full_shape)
        p.map(func, [(intervals[i], intervals[i+1], t)
                     for i, t in enumerate(trj_list)])
    p.join()

    xyz = _tonumpyarray(shared_xyz).reshape(full_shape)
    return md.Trajectory(xyz, topology=example_center.top)


def shared_array_like_trj(lengths, example_trj):

    # when we allocate the shared array below, we expect a float32
    # c_double seems to work with trajectories that use float32s. Why?
    # I have no idea.
    assert example_trj.xyz.dtype == np.float32
    shape = example_trj.xyz.shape

    # TODO: check all inputs against root

    full_shape = (sum(lengths), shape[1], shape[2])

    # mp.Arrays are one-dimensional, so multiply the shape together for size
    try:
        dtype = ctypes.c_float
        shared_array = mp.Array(dtype, reduce(mul, full_shape, 1),
                                lock=False)
    except OSError as e:
        if e.args[0] != 28:
            raise
        arr_bytes = reduce(mul, full_shape, 1) * ctypes.sizeof(dtype)
        raise exception.InsufficientResourceError(
            ("Couldn't allocate array of size %.2f GB to %s as part of "
             "loading trajectories in parallel. Check this partition "
             "to ensure it has sufficient space or set $TMPDIR to a "
             "path that does have sufficient space. (See the tempfile "
             "module documentation on this behavior.)") %
            (os.path.basename(mp.util.get_temp_dir()),
             arr_bytes / 1024**3))

    return full_shape, shared_array


def _slice_and_insert_xyz(spec, arr_shape, atoms):
    """Slice out atoms and insert xyz of trajectory into larger array.

    Parameters
    ----------
    spec : tuple, shape=(2,)
        The index,
    arr_shape : tuple, shape=(3,)
        The shape of the master array.
    atoms : int, default=None
        Number of parallel processes to use when performing this
        operation.

    Returns
    -------
    shape : tuple, shape=(3,)
        The sizes of the inserted array
    """
    start, end, c = spec

    if atoms is not None:
        c = c.atom_slice(c.top.select(atoms))

    if c.xyz.shape[1:] != arr_shape[1:]:
        raise exception.DataInvalid(
            'Trajectory at %s had improper shape %s, expected %s.' %
            ((start, end), c.xyz.shape, arr_shape))

    # reshape shared array and set it.
    arr = _tonumpyarray(shared_array).reshape(arr_shape)
    arr[start:end] = c.xyz

    return c.xyz.shape


def _init(shared_array_):
    # for some reason, the shared array must be inhereted, not passed
    # as an argument
    global shared_array
    shared_array = shared_array_


def _tonumpyarray(mp_arr, dtype='float32'):
    # mp_arr.get_obj if Array is locking, otherwise mp_arr.
    return np.frombuffer(mp_arr, dtype=dtype)


def _load_to_position(spec, arr_shape):
    '''
    Load a specified file into a specified position by spec. The
    arr_shape parameter lets us know how big the final array should be.
    '''
    (position, filename, load_kwargs) = spec

    xyz = md.load(filename, **load_kwargs).xyz

    # mp.Array must be converted to numpy array and reshaped
    arr = _tonumpyarray(shared_array).reshape(arr_shape)

    # dump coordinates in.
    arr[position:position+len(xyz)] = xyz

    return xyz.shape
