from biotite.structure import stack, AtomArray, CellList
import numpy as np
from sklearn import decomposition


def _smallest_eigenvector(coords):
    """ Returns the eigenvector with the smallest eigenvalue for the set of
    coords """
    pca = decomposition.PCA()
    pca.fit_transform(coords)
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    min_value_idx = np.argmin(eigenvalues)
    min_vector = eigenvectors[min_value_idx, min_value_idx]
    return min_vector


def membrane_leaflet_identification(atoms, cutoff=50, lipid_head="P", periodic=False):
    r"""
    Identify membrane leaflets in membrane simulations

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        Models to calculate find leaflets in
    cutoff : float
        Cutoff distance for lipid headgroup neighbor search
    lipid_head : str
        Name of a headgroup atom
    periodic : bool
        Wether to take into account PBC or not

    Returns
    -------
    angles: ndarray, dtype=float
        For each lipid_head atom in each model, the id of its leaflet (either 0
        or 1)
    """
    if isinstance(atoms, AtomArray):
        atoms = stack([atoms])
    atoms = atoms[:, atoms.atom_name == lipid_head]

    leaflet = np.full((len(atoms), atoms.array_length()), np.nan)
    for frame in range(len(atoms)):
        frame_atoms = atoms[frame, :]

        # for every headgroup, get all other headgroups inside the cutoff
        cell_list = CellList(frame_atoms, cell_size=cutoff, periodic=periodic)
        neighbor_list = cell_list.get_atoms(frame_atoms.coord, radius=cutoff)

        # calculate the smallest eigenvector for each set of headgroups
        for lipid_id in range(len(neighbor_list)):
            neighbors = neighbor_list[lipid_id]
            neighbors = neighbors[neighbors >= 0]
            coords = frame_atoms[neighbors].coord
            leaflet[frame, lipid_id] = _smallest_eigenvector(coords)

    # normalize and return
    leaflet[leaflet > 0] = 1
    leaflet[leaflet < 0] = 0
    return leaflet

