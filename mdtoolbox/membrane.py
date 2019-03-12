from biotite.structure import stack, AtomArray, AtomArrayStack
import biotite.structure as struct
import numpy as np


def membrane_leaflet_identification(atoms, lipid_head="P"):
    """
    Simple algorithm to identify membrane leaflets in a topology

    Parameters
    ---------
    atoms : AtomArray or AtomArrayStack
    lipid_head : str, optional
        Name of the lipid headgroup

    Returns
    -------
    tuple of ndarray
        Two masks for AtomArray identifying headgroups of the upper and lower
        leaflet

    """
    if isinstance(atoms, AtomArrayStack):
        atoms = atoms[0]
    lipid_heads = atoms.atom_name == lipid_head

    z_center = struct.centroid(atoms[lipid_heads])[2]
    upper = (atoms.coord[:, 2] < z_center) & lipid_heads
    lower = np.invert(upper) & lipid_heads

    return upper, lower


def membrane_thickness(upper_atoms, lower_atoms):
    """ Calculate membrane thickness. For each atom, the algorithm calculates
    the distance to the closest atom in the other leaflet.

    Parameters
    ----------
    upper_atoms, lower_atoms : AtomArray or AtomArrayStack
        atoms of the lower and upper membrane leaflet to use for distance
        calculation

    Returns
    -------
    ndarray
        For each frame, a set of x and y coordinates and the corresponding
        membrane thickness is returned

    """
    if not isinstance(lower_atoms, AtomArrayStack):
        lower_atoms = stack([lower_atoms])
    if not isinstance(upper_atoms, AtomArrayStack):
        upper_atoms = stack([upper_atoms])

    thickness_upper = np.full([len(upper_atoms), upper_atoms.array_length(), 3],
                              0.0)
    thickness_lower = np.full([len(lower_atoms), lower_atoms.array_length(), 3],
                              0.0)
    for idx in range(upper_atoms.array_length()):
        atom = upper_atoms[:, idx]
        dist = struct.distance(atom, lower_atoms)

        atom_min_dist = dist.min(axis=1)

        thickness_upper[:, idx, 0:2] = atom.coord[:, 0, 0:2]
        thickness_upper[:, idx, -1] = atom_min_dist


    for idx in range(lower_atoms.array_length()):
        atom = lower_atoms[:, idx]
        dist = struct.distance(atom, upper_atoms)

        atom_min_dist = dist.min(axis=1)

        thickness_lower[:, idx, 0:2] = atom.coord[:, 0, 0:2]
        thickness_lower[:, idx, -1] = atom_min_dist

    thickness = np.concatenate([thickness_upper, thickness_lower], axis=1)

    if thickness.shape[0] == 1:
        return thickness[0, ...]
    else:
        return thickness

