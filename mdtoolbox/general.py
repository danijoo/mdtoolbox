import numpy as np
import biotite.structure as struct

def psi_iter(atoms):
    """ Iterate over all psi angles in a protein. """
    res_iter1 = struct.residue_iter(atoms)
    res_iter2 = struct.residue_iter(atoms)
    next(res_iter2)
    for res, next_res in zip(res_iter1, res_iter2):
        yield res[..., struct.filter_backbone(res)] \
            + next_res[..., next_res.atom_name == "N"]
    yield None # No psi for last AS

def phi_iter(atoms):
    """ Iterate over all phi angles in a protein. """
    res_iter1 = struct.residue_iter(atoms)
    res_iter2 = struct.residue_iter(atoms)
    next(res_iter1)
    yield None # No phi for first AS
    for res, prev_res in zip(res_iter1, res_iter2):
        yield prev_res[..., prev_res.atom_name == "C"] \
            + res[..., struct.filter_backbone(res)]

def chain_iter(atoms):
    """ Iterate chains """
    for chain_id in np.unique(atoms.chain_id):
        yield atoms[..., atoms.chain_id == chain_id]

def stack_iter(atoms):
    """ Iterate an array stack by frames """
    for stack in atoms:
        yield stack