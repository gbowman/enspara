# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import logging
import csv
import numbers

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.csgraph import connected_components

from .. import exception

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrimMapping:
    """The TrimMapping maps state ids before and after ergodic trimming.

    It stores the injective mapping of trimmed state ids to original
    state ids, as well as the inverse, in its two properties.

    Attributes
    ----------
    to_original : dict
        Dictionary mapping post-trim state ids to original state ids.
    to_mapped : dict
        Dictionary mapping original state ids to post-trim state ids.
    """

    __slots__ = ['to_original']

    def __init__(self, transformations=None):
        '''Construct a new TrimMapping.

        Parameters
        ----------
        transformations : list, optional
            A list of 2-tuples, each of the form
            (original_state_id, trimmed_state_id).
        '''

        if transformations:
            self.to_original = {t: o for o, t in transformations}

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            return cls.read(f)

    @classmethod
    def read(cls, file):
        reader = csv.reader(file)

        headers = next(reader)
        assert headers == ['original', 'mapped']

        column = {h: [] for h in headers}
        for row in reader:
            for h, v in zip(headers, row):
                column[h].append(int(v))

        return TrimMapping(zip(column['original'], column['mapped']))

    @property
    def to_mapped(self):
        return {v: k for k, v in self.to_original.items()}

    @to_mapped.setter
    def to_mapped(self, value):
        self.to_original = {v: k for k, v in value.items()}

    def save(self, filename):
        with open(filename, 'w') as f:
            self.write(f)

    def write(self, file):
        writer = csv.writer(file)

        writer.writerow(['original', 'mapped'])
        writer.writerows(sorted(self.to_mapped.items(),
                                key=lambda x: x[0]))

    def __eq__(self, other):

        if self is other:
            return True
        elif hasattr(other, 'to_original') and hasattr(other, 'to_mapped'):
            return (self.to_original == other.to_original) and \
                   (self.to_mapped == other.to_mapped)
        else:
            try:
                return TrimMapping(other) == self
            except:
                return False

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "to_original:"+str(self.to_original)


def assigns_to_counts(
        assigns, lag_time, max_n_states=None, sliding_window=True):
    """Count transitions between states in a single trajectory.

    Parameters
    ----------
    assigns : array, shape=(traj_len, )
        A 2-D array where each row is a trajectory consisting of a
        sequence of state indices.
    lag_time : int
        The lag time (i.e. observation interval) for counting
        transitions.
    max_n_states : int, default=None
        The number of states. This is useful for controlling the
        dimensions of the transition count matrix in cases where the
        input trajectory does not necessarily visit every state.
    sliding_window : bool, default=True
        Whether to use a sliding window for counting transitions or to
        take every lag_time'th state.

    Returns
    -------
    C :  array, shape=(n_states, n_states)
        A transition count matrix.
    """

    if not isinstance(lag_time, numbers.Integral):
        raise exception.DataInvalid(
            "The lag time must be an integer. Got %s type %s." %
            lag_time, type(lag_time))
    if lag_time < 1:
        raise exception.DataInvalid(
            "Lag times must be be strictly greater than 0. Got '%s'." %
            lag_time)

    # if it's 1d, later stuff will fail
    if len(assigns.shape) == 1:
        raise exception.DataInvalid(
            'The given assignments array has 1-dimensional shape %s. '
            'Two dimensional shapes = (n_trj, n_frames) are expected. '
            'If this is really what you want, try using '
            'assignments.reshape(1, -1) to create a single-row 2d array.')

    assigns = np.array([a[np.where(a != -1)] for a in assigns], dtype='O')

    if max_n_states is None:
        max_n_states = np.concatenate(assigns).max() + 1

    transitions = [
        _transitions_helper(
            assign, lag_time=lag_time, sliding_window=sliding_window)
        for assign in assigns]
    # generate sparse matrix
    mat_coords = np.concatenate(transitions, axis=1)
    mat_data = np.ones(mat_coords.shape[1], dtype=int)
    C = scipy.sparse.coo_matrix(
        (mat_data, mat_coords), shape=(max_n_states, max_n_states))
    return C


def eigenspectrum(T, n_eigs=None, left=True, maxiter=100000, tol=1E-30):
    """Compute the eigenvectors and eigenvalues of a transition
    probability matrix.

    Parameters
    ----------
    T : array, shape=(n_states, n_states)
        A transition probability matrix.
    n_eigs : int, optional
        The number of eigenvalues and eigenvectors to compute. If not
        speficied, all are computed.
    left: bool, default=False
        Compute the left eigenvalues rather than the right eigenvalues.
    maxiter : int, default=100000
        Limit the maximum number of iterations used by the sparse
        eigenvalue solver. (Used only for sparse matrices.)
    tol : float, default=1e-30
        Relative accuracy for eigenvalues (stopping criterion). (Used
        only for sparse matrices.)

    Returns
    -------
    vals, vecs : 2-tuple, (ndarray, ndarray)
        Eigenvalues and eigenvectors for this system, respectively.
    """

    if n_eigs is None:
        n_eigs = T.shape[0]
    elif n_eigs < 2:
        raise ValueError('n_eig must be greater than or equal to 2')

    if n_eigs > T.shape[0]:
        logger.warning(
            ("Trying to compute {n} eigenvalues from an {s} x {s} matrix " +
             "yields only {s} eigenvalues.").format(n=n_eigs, s=T.shape[0]))

    # left eigenvectors input processing (?)
    T = T.T if left else T

    # performance improvement for small arrays; also prevents erroring
    # out when ndim - 2 <= n_eigs and T is sparse.
    if T.shape[0] < 1000 and scipy.sparse.issparse(T):
        T = T.toarray()

    if scipy.sparse.issparse(T):
        vals, vecs = scipy.sparse.linalg.eigs(
            T.tocsr(), n_eigs, which="LR", maxiter=maxiter, tol=tol)
    else:
        vals, vecs = scipy.linalg.eig(T)

    order = np.argsort(-np.real(vals))
    vals = vals[order]
    vecs = vecs[:, order]

    # normalize the first eigenvector to obtain the eq populations
    vecs[:, 0] /= vecs[:, 0].sum()

    vals = np.real(vals[:n_eigs])
    vecs = np.real(vecs[:, :n_eigs])

    return vals, vecs


def trim_disconnected(counts, threshold=1, renumber_states=True):
    """Trim disconnected states from a counts matrix.

    Parameters
    ----------
    counts : array, shape=(n_states, n_states)
        A 2-D array in which the position [i, j] is the number of times
        the transition i->j was observed.
    threshold : int, default=1
        The number of transitions in and out of a state that are
        required to count the state as connected.
    renumber_states : bool, default=False
        Should states be renumbered, reassigning new, contiguous state
        indices after removing disconnected states.

    Returns
    -------
    mapping:  TrimMapping
        The mapping between original and renumbered states (if states
        were renumbered).
    """

    out_type = type(counts)
    if scipy.sparse.issparse(counts):
        counts = counts.toarray()

    thresholded_counts = np.array(counts, copy=True)
    thresholded_counts[counts < threshold] = 0

    n_subgraphs, labels = connected_components(thresholded_counts,
                                               connection="strong",
                                               directed=True)

    pops = counts.sum(axis=1)

    subgraph_pops = [np.sum(pops[labels == i])
                     for i in range(n_subgraphs)]
    maxpop_subgraph = np.argmax(subgraph_pops)

    keep_states = np.where(labels == maxpop_subgraph)[0]

    if renumber_states:
        new_states = np.arange(len(keep_states))

        trimmed_counts = np.zeros((len(keep_states), len(keep_states)),
                                  dtype=counts.dtype)

        trimmed_counts[np.ix_(new_states, new_states)] = \
            counts[np.ix_(keep_states, keep_states)]

        mapping = TrimMapping(zip(keep_states,
                              range(len(trimmed_counts))))

    else:
        trim_states = np.where(labels != maxpop_subgraph)
        trimmed_counts = np.array(counts, copy=True)

        trimmed_counts[trim_states, :] = 0
        trimmed_counts[:, trim_states] = 0

        mapping = TrimMapping(zip(keep_states, keep_states))

    if type(trimmed_counts) is not out_type:
        trimmed_counts = out_type(trimmed_counts)

    return mapping, trimmed_counts


def eq_probs(T, maxiter=100000, tol=1E-30):
    val, vec = eigenspectrum(T, n_eigs=3, left=True, maxiter=maxiter, tol=tol)

    return vec[:, 0]


def _transitions_helper(
        assigns_1d, lag_time=1, sliding_window=True):
    # TODO: check trajectory is 1d array

    if sliding_window:
        start_states = assigns_1d[:-lag_time:1]
        end_states = assigns_1d[lag_time::1]
    else:
        start_states = assigns_1d[:-lag_time:lag_time]
        end_states = assigns_1d[lag_time::lag_time]
    transitions = np.row_stack((start_states, end_states))
    return transitions
