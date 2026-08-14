import pytest
from pysidt.extensions import *
from pysidt.decomposition import *
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
            assoc_decomposition_init_value_unc=None,
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
                        max_ring_gen_size=5, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc=None)
        
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
                                                    tree=None, estimate_delta=False, assoc_decomposition_init_value_unc=None))
                    
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
                
    def test_molecular_specify_external_new_bond_extensions(self):
        mol = Molecule(smiles="CCC")
        
        exts = molecular_specify_external_new_bond_extensions(mol, 0, "", r=[ATOMTYPES[x] for x in ["C","O","N"]], r_bonds=[1.0,2.0,3.0], 
                        r_label=[''], tree=None, estimate_delta=False, assoc_decomposition_init_value_unc=None)
        
        m = [ext[0] for ext in exts][0]
        msout = Molecule(smiles="CCCC")
        
        assert m.is_isomorphic(msout,save_order=True), f"Extension {msout.to_smiles()} not generated in molecular_specify_external_new_bond_extensions from propane"
        
    def test_molecular_generalize_remove_atom_extensions(self):
        mol = Molecule(smiles="CCC")
        
        exts = molecular_generalize_remove_atom_extensions(mol, 0, "", 1, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc=None)
        
        assert exts[0][0].is_isomorphic(Molecule(smiles="CC"),save_order=True), f"Extension CC not generated in molecular_generalize_remove_atom_extensions from propane"

    def test_molecular_transform_bond_extensions(self):
        mol = Molecule(smiles="CCC")
        
        exts = molecular_transform_bond_extensions(mol, 0, 1, "", [1.0,2.0,3.0], [1.0,2.0,3.0], tree=None, estimate_delta=False, assoc_decomposition_init_value_unc=None)
        
        ms = [ext[0] for ext in exts]
        msouts = [Molecule(smiles=sm) for sm in ["CCC","CC=C","CC#C"]]
        
        for m in ms:
            for mout in msouts:
                if m.is_isomorphic(mout,save_order=True):
                    break
            else:
                assert False, f"Unexpected Extension {m.to_smiles()} generated in molecular_transform_bond_extensions from propane"
    
#     def test_generative_extensions_from_tree_node(self):
#         decomp = Group().from_adjacency_list("""1 * N u0 {2,D}
#         2 [C,O] u0 {1,D}""")
#         value,name = evaluate_single(tree, decomp, trace=True)
#         node = tree.nodes[name]
        
#         grpsout = [Group().from_adjacency_list("""1 N     u0 {2,D}
# 2 [C,O] u0 {1,D} {3,[S,D,T,B]}
# 3 Rx!H  ux {2,[S,D,T,B]}"""),
#                    Group().from_adjacency_list("""1 * N u0""")]
        
#         new_grps,nodes,delta,delta_unc = generative_extensions_from_tree_node(decomp,node,[ATOMTYPES[x] for x in ["C", "O", "N"]])
        
#         gmap = dict()
#         for gnew in new_grps:
#             for gout in grpsout:
#                 if gnew.is_isomorphic(gout,save_order=True):
#                     gmap[gnew] = gout
#                     break
        
#         assert len(gmap) == len(new_grps) and len(new_grps) == len(grpsout), "Group mismatch"
        
    # def test_get_molecular_extensions_for_generative_expansion(self):
        
    #     mol = Molecule(smiles="CCC")
        
    #     exts = get_molecular_extensions_for_generative_expansion(
    #         mol,
    #         None,
    #         atom_decomposition_noH,
    #         r_full = [ATOMTYPES[x] for x in ["C", "O", "N"]],
    #         r_bonds_full=[1, 2, 3], #, 1.5],
    #         r_un_full=[0],
    #         r_site_full=[],
    #         r_morph_full=[],
    #         r_ncoord_full=[],
    #         r_label=None,
    #         basename="Root",
    #         n_strucs_min=1,
    #         n_strucs_max=1,
    #         max_ring_gen_size=9,
    #         decomposition_associated=None,
    #         generate_extensions_from_tree=False,
    #         generate_local_extensions=True,
    #         )
        
    #     smsout = ['CCC',
    #         'CCO',
    #         'CCN',
    #         'CCCC',
    #         'CCCO',
    #         'CCCN',
    #         'CC',
    #         'CCC1CC1',
    #         'CCC1CCC1',
    #         'CCC1CCCC1',
    #         'CCC1CCCCC1',
    #         'CCC1CCCCCC1',
    #         'CCC1CCCCCCC1',
    #         'CCC1CCCCCCCC1',
    #         'COC',
    #         'CNC',
    #         'CC(C)C',
    #         'CC(C)O',
    #         'CC(N)C',
    #         'C=CC',
    #         'C#CC',
    #         'CC1(C)CC1',
    #         'CC1(C)CCC1',
    #         'CC1(C)CCCC1',
    #         'CC1(C)CCCCC1',
    #         'CC1(C)CCCCCC1',
    #         'CC1(C)CCCCCCC1',
    #         'CC1(C)CCCCCCCC1',
    #         'C1CC1',
    #         'C1CCC1',
    #         'C1CCCC1',
    #         'C1CCCCC1',
    #         'C1CCCCCC1',
    #         'C1CCCCCCC1',
    #         'C1CCCCCCCC1']
        
    #     msout = [Molecule(smiles=sm) for sm in smsout]
        
    #     for ext in exts:
    #         for m in msout:
    #             if ext[0].is_isomorphic(m,save_order=True):
    #                 break
    #         else:
    #             assert False, "Molecule {} generated was not expected".format(ext[0].to_smiles())
        
        
    #     for m in msout:
    #         for ext in exts:
    #             if ext[0].is_isomorphic(m,save_order=True):
    #                 break
    #         else:
    #             assert False, "Expected Molecule {} was not generated".format(m.to_smiles())
        