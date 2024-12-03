from itertools import groupby

import numpy as np
import mdtraj as md

GMX_EQUIV_GROUPS = {
    ('*', 'H?'): (('*', 'H'), ('*', 'H2'), ('*', 'H3')),
    ('*', 'OC?'): (('*', 'OC1'), ('*', 'OC2')),
    ('ARG', 'HH?1'): (('ARG', 'HH11'), ('ARG', 'HH21')),
    ('ARG', 'HH?2'): (('ARG', 'HH12'), ('ARG', 'HH22')),
    ('ARG', 'NH?'): (('ARG', 'NH1'), ('ARG', 'NH2')),
    ('ASN', 'HD2?'): (('ASN', 'HD21'), ('ASN', 'HD22')),
    ('ASP', 'OD?'): (('ASP', 'OD1'), ('ASP', 'OD2')),
    ('GLU', 'OE?'): (('GLU', 'OE1'), ('GLU', 'OE2')),
    ('GLN', 'HE2?'): (('GLN', 'HE21'), ('GLN', 'HE22')),
    ('LYS', 'HZ?'): (('LYS', 'HZ1'), ('LYS', 'HZ2'), ('LYS', 'HZ3')),
}


def to_one_letter(k):
    if k[0] != '*':
        return (md.core.residue_names._AMINO_ACID_CODES[k[0]], k[1])
    else:
        return k


def to_tuple(top, aid, resSeq):
    return (
        top.atom(aid).residue.resSeq if resSeq
        else top.atom(aid).residue.index,
        top.atom(aid).residue.code,
        top.atom(aid).name,
        top.atom(aid).element.name
    )


def match(resn, atomn):

    eq_dict = {}
    for k, v in GMX_EQUIV_GROUPS.items():
        for v_i in v:
            eq_dict[to_one_letter(v_i)] = to_one_letter(k)

    for (haystack_resn, haystack_atomn), replace_tuple in eq_dict.items():
        if haystack_atomn == atomn:
            if haystack_resn == '*' or resn == haystack_resn:
                return (resn, replace_tuple[1])
    return (resn, atomn)


def combine_hbonds(all_hbonds, top, resSeq=True):
    """Compute a mapping of hbond definitions to hbond list indices.
    """

    expanded_hbonds = [[to_tuple(aid=x_i, top=top, resSeq=resSeq) for x_i in x]
                       for x in all_hbonds]
    assert np.all([expanded_hbond[1][3] == 'hydrogen'
                   for expanded_hbond in expanded_hbonds])

    deident_hbonds = [[(t[0], *match(*t[1:3]), t[3]) for t in quadruplet]
                      for quadruplet in expanded_hbonds]
    assert np.all([deident_hbond[1][3] == 'hydrogen'
                   for deident_hbond in deident_hbonds])

    combo_hbond_d = {tuple(k): [x[0] for x in g] for k, g
                     in groupby(enumerate(deident_hbonds),
                                key=lambda x: x[1])}
    return combo_hbond_d


def trjmap_combo_hbond_d(combo_hbond_d, hbond_trj):

    combo_hbond_trj_d = {}
    for name, pair_ids in combo_hbond_d.items():
        combo_hbond_trj = np.zeros_like(hbond_trj[:, 0])
        for pair_id in pair_ids:
            combo_hbond_trj = np.logical_or(
                hbond_trj[:, pair_id],
                combo_hbond_trj)

        combo_hbond_trj_d[name] = combo_hbond_trj

    hbond_names = sorted(combo_hbond_trj_d.keys())

    combo_hbond_trjs = np.vstack([combo_hbond_trj_d[k] for k in hbond_names]).T

    return hbond_names, combo_hbond_trjs


def hbond_featurize(trj):
    """Featurize a trajectory into hydrogen bond presence/absence vectors.

    Given an MDTraj trajectory, calculate the presence or absence of all
    hydrogen bonds observed anywhere in the trajectory for each timepoint.

    Parameters
    ----------
    trj : md.Trajectory
        Trajectory to compute hydrogen bond featurization of.

    Returns
    -------
    hbond_defns :

    hbond_features : np.array, shape=(n_frames, n_hbonds)
        A binary array a 1 at position (i, j) indicates the presence of
        hydrogen bond j (defined in `hbond_defns`) in frame i.
    """

    hbond_list = md.baker_hubbard(trj, freq=0)
    hbond_defns = combine_hbonds(hbond_list, trj.top, resSeq=False)

    individual_hbonds = [md.baker_hubbard(t) for t in trj]
    hbond_trj = np.zeros((len(individual_hbonds), len(hbond_list)),
                         dtype='int8')

    for frameid, hbonds_present in enumerate(individual_hbonds):
        hbond_trj[frameid, :] = [np.any(hbond_list[i] == hbonds_present)
                                 for i in range(len(hbond_list))]

    hbond_defns, hbond_features = trjmap_combo_hbond_d(hbond_defns, hbond_trj)
    hbond_defns = combine_hbonds(hbond_list, trj.top)

    return hbond_defns, hbond_features
