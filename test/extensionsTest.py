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
    
    def test_molecular_generalize_remove_bridge_extensions(self):
        mol = Molecule(smiles="C1CCOC12CC2")
        
        exts = []
        for i in range(len(mol.atoms)):
            for j in range(len(mol.atoms)):
                if i > j:
                    exts.extend(molecular_generalize_remove_bridge_extensions(mol, i, j, n_strucs_max=1, basename="", r_bonds=[1.0,2.0,3.0], 
                                                    tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None))
                    
        sms = ['CCOC1(C)CC1',
            'CCC1(OC)CC1',
            'CCCC1(O)CC1',
            'CCCOC1CC1',
            'CCCC1CC1',
            'OCCCC1CC1',
            'CCC1CCCO1',
            'CCC1CCCO1',
            'CC1(C)CCCO1']
        
        correct_mols = [Molecule(smiles=sm) for sm in sms]
        
        for ext in exts:
            m = ext[0]
            for mout in correct_mols:
                if m.is_isomorphic(mout,save_order=True):
                    break
            else:
                assert False, f"Extension {m.to_smiles()} not generated in molecular_generalize_remove_bridge_extensions from C1CCOC12CC2"
                