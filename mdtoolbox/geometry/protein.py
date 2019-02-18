from biotite.structure import AtomArray, stack
from biotite.structure.util import norm_vector, vector_dot
import numpy as np


def angle_between_helices(atoms, sele1, sele2, carbonyl_oxygen="O",
                          carbonyl_carbon="C"):
    r"""
    Compute the angle between two helices.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        Models to calculate the angle
    sele1 : ndarray, dtype=bool
        Boolean mask for the first helix
    sele2 : ndarray, dtype=bool
        Boolean mask for the second helix
    carbonyl_oxygen : str, optional
        Atom name of the carbonyl oxygen (default: "O")
    carbonyl_carbon : str, optional
        Atom name of the carbonyl carbon (default: "C")

    Returns
    -------
    angles: ndarray, dtype=float
        The angle in degree for every model of atoms
    """
    if isinstance(atoms, AtomArray):
        atoms = stack([atoms])

    h1_c = atoms[:, sele1 & (atoms.atom_name == carbonyl_carbon)]
    h1_o = atoms[:, sele1 & (atoms.atom_name == carbonyl_oxygen)]

    h2_c = atoms[:, sele2 & (atoms.atom_name == carbonyl_carbon)]
    h2_o = atoms[:, sele2 & (atoms.atom_name == carbonyl_oxygen)]

    v1 = (h1_c.coord.T - h1_o.coord.T).T
    v2 = (h2_c.coord.T - h2_o.coord.T).T

    v1_sum = v1.sum(axis=1)
    v2_sum = v2.sum(axis=1)
    norm_vector(v1_sum)
    norm_vector(v2_sum)
    angle = np.arccos(vector_dot(v1_sum, v2_sum))
    angle = np.rad2deg(angle)

    return angle
