import pytest
from pysidt.extensions import *
from molecule.molecule import Molecule
import logging

class TestExtensionGeneration:
    """
    Contains unit tests of the :class:`Arrhenius` class.
    """
    
    def test_molecular_transform_atom_extensions(self):
        mol = Molecule(smiles="CCC")
        
        exts = molecular_transform_atom_extensions(
            mol,
            0,
            "",
            r_full=[ATOMTYPES[x] for x in ["C","O","N"]],
            r_un_full=[0],
            r_site_full=[],
            r_morph_full=[],
            r_lone_pairs_full=[],
            tree=None,
            estimate_delta=False,
            assoc_decomposition_init_value_unc_dict=None,
        )
        
        ms = [ext[0] for ext in exts]
        msout = [Molecule(smiles=sm) for sm in ["CCC","CCO","CCN"]]
        
        for m in ms:
            for mout in msout:
                if m.is_isomorphic(mout,save_order=True):
                    break
            else:
                assert False, f"Extension {m.to_smiles()} not generated in molecular_transform_atom_extensions from propane"
    
    def test_molecular_specify_internal_new_bond_extensions(self):
        
        mol = Molecule().from_smiles("CCC")
        exts = molecular_specify_internal_new_bond_extensions(mol, 0, 1, 1, "", [1.0,2.0,3.0], 
                        max_ring_gen_size=5, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None)
        
        ms = [ext[0] for ext in exts]
        msout = [Molecule(smiles=sm) for sm in ["CCC","C1CC1C","C1CCC1C","C1CCCC1C"]]
        
        for m in ms:
            for mout in msout:
                if m.is_isomorphic(mout,save_order=True):
                    break
            else:
                assert False, f"Extension {m.to_smiles()} not generated in molecular_specify_internal_new_bond_extensions from propane"
    