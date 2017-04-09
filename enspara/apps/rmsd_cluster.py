import sys
import argparse
import os
import logging
import itertools

from functools import partial
from multiprocessing import cpu_count

import numpy as np
import mdtraj as md
from mdtraj import io

from enspara.cluster import KHybrid
from enspara.cluster.util import load_frames
from enspara.util import load_as_concatenated
from enspara import exception

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_command_line(argv):
    '''Parse the command line and do a first-pass on processing them into a
    format appropriate for the rest of the script.'''

    parser = argparse.ArgumentParser(formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--trajectories', required=True, nargs="+", action='append',
        help="The aligned xtc files to cluster.")
    parser.add_argument(
        '--topology', required=True, action='append',
        help="The topology file for the trajectories.")
    parser.add_argument(
        '--algorithm', required=True, choices=["khybrid"],
        help="The clustering algorithm to use.")
    parser.add_argument(
        '--atoms', default='name C or name O or name CA or name N or name CB',
        help="The atoms from the trajectories (using MDTraj atom-selection"
             "syntax) to cluster based upon.")
    parser.add_argument(
        '--rmsd-cutoff', required=True, type=float,
        help="The RMSD cutoff (in nm) to determine cluster size.")
    parser.add_argument(
        '--processes', default=cpu_count(), type=int,
        help="Number processes to use for loading and clustering.")
    parser.add_argument(
        '--partitions', default=None, type=int,
        help="Use this number of pivots when delegating to md.rmsd. "
             "This can avoid md.rmsd's large-dataset segfault.")
    parser.add_argument(
        '--subsample', default=None, type=int,
        help="Subsample the input trajectories by the given factor. "
             "1 implies no subsampling.")
    parser.add_argument(
        '--output-path', default='',
        help="The output path for results (distances, assignments, centers).")
    parser.add_argument(
        '--output-tag', default='',
        help="An optional extra string prepended to output filenames (useful"
             "for giving this choice of parameters a name to separate it from"
             "other clusterings or proteins.")

    args = parser.parse_args(argv[1:])

    if len(args.topology) != len(args.trajectories):
        raise exception.ImproperlyConfigured(
            "The number of --topology and --trajectory flags must agree.")

    return args


def rmsd_hack(trj, ref, partitions=None, **kwargs):

    if partitions is None:
        return md.rmsd(trj, ref)

    pivots = np.linspace(0, len(trj), num=partitions+1, dtype='int')

    rmsds = np.zeros(len(trj))

    for i in range(len(pivots)-1):
        s = slice(pivots[i], pivots[i+1])
        rmsds[s] = md.rmsd(trj[s], ref, **kwargs)

    return rmsds


def filenames(args):

    tag_params = [
        args.output_tag,
        args.algorithm,
        str(args.rmsd_cutoff)]

    if args.subsample:
        tag_params.append(str(args.subsample)+'subsample')

    path_stub = os.path.join(args.output_path, '-'.join(tag_params))

    return {
        'distances': path_stub+'-distances.h5',
        'centers': path_stub+'-centers.h5',
        'assignments': path_stub+'-assignments.h5',
    }


def load(args, selection, stride):

    sentinel_trj = md.load(args.topology[0])
    try:
        # noop, but causes fast-fail w/bad args.atoms
        sentinel_trj.top.select(selection)
    except:
        raise exception.DataInvalid((
            "The provided selection '{s}' didn't match the topology"
            "file, {t}").format(s=selection, t=args.topology))

    flat_trjs = []
    configs = []
    for topfile, trjset in zip(args.topology, args.trajectories):
        top = md.load(topfile).top
        for trj in trjset:
            flat_trjs.append(trj)
            configs.append({
                'top': top,
                'stride': stride,
                'atom_indices': top.select(selection)
                })

    logger.info(
        "Loading %s trajectories with %s atoms using %s processes"
        "(subsampling %s)",
        len(flat_trjs), len(top.select(selection)),
        args.processes, args.subsample)

    lengths, xyz = load_as_concatenated(
        flat_trjs, args=configs, processes=args.processes)

    return lengths, xyz, top.subset(top.select(args.atoms))


def main(argv=None):
    '''Run the driver script for this module. This code only runs if we're
    being run as a script. Otherwise, it's silent and just exposes methods.'''
    args = process_command_line(argv)

    logger.info(
        "Loading finished. Clustering using atoms matching '%s'.", args.atoms)

    lengths, xyz, select_top = load(args, selection=args.atoms,
                                    stride=args.subsample)

    clustering = KHybrid(
        metric=partial(rmsd_hack, partitions=args.partitions),
        cluster_radius=args.rmsd_cutoff)

    # md.rmsd requires an md.Trajectory object, so wrap `xyz` in
    # the topology.
    clustering.fit(md.Trajectory(xyz=xyz, topology=select_top))

    logger.info(
        "Clustered %s frames into %s clusters in %s seconds.",
        sum(lengths), len(clustering.centers_), clustering.runtime_)

    result = clustering.result_.partition(lengths)

    outdir = os.path.dirname(filenames(args)['centers'])
    logger.info("Saving results at %s", outdir)

    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    md.join(load_frames(
        list(itertools.chain(*args.trajectories)),
        result.center_indices,
        top=md.load(args.topology[0]).top,
        stride=args.subsample)).save_hdf5(filenames(args)['centers'])

    if args.subsample:
        logger.info("Reloading entire dataset for reassignment")
        lengths, xyz, _ = load(args, selection=args.atoms, stride=1)

        logger.info("Reassigning dataset.")
        result = clustering.predict(md.Trajectory(
            xyz, topology=select_top)).partition(lengths)

        # overwrite temporary output with actual results
        io.saveh(filenames(args)['distances'], result.distances)
        io.saveh(filenames(args)['assignments'], result.assignments)
    else:
        io.saveh(filenames(args)['distances'], result.distances)
        io.saveh(filenames(args)['assignments'], result.assignments)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
