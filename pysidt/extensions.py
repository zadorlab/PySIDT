import logging
from copy import deepcopy

import numpy as np
import itertools

try:
    from molecule.molecule.atomtype import ATOMTYPES, allElements, get_atomtype
    from molecule.molecule.element import bde_elements, PeriodicSystem, get_element
    from molecule.molecule.group import GroupAtom, GroupBond, Group
    from molecule.molecule.molecule import Molecule, Atom, Bond
    from molecule.exceptions import UnexpectedChargeError, AtomTypeError
except:
    from rmgpy.molecule.atomtype import ATOMTYPES, allElements, get_atomtype
    from rmgpy.molecule.element import bde_elements, PeriodicSystem, get_element
    from rmgpy.molecule.group import GroupAtom, GroupBond, Group
    from rmgpy.molecule.molecule import Molecule, Atom, Bond
    from rmgpy.exceptions import UnexpectedChargeError, AtomTypeError

from pysidt.utils import find_shortest_paths, evaluate_single
from pysidt.mol import *

def split_mols(data, newgrp):
    """
    divides the reactions in rxns between the new
    group structure newgrp and the old structure with
    label oldlabel
    returns a list of reactions associated with the new group
    the list of reactions associated with the old group
    and a list of the indices of all of the reactions
    associated with the new group
    """
    if len(data) == 0:
        return [],[]
    
    new = []
    comp = []

    if isinstance(data[0], Molecule):
        for i, mol in enumerate(data):
            if mol.is_subgraph_isomorphic(
                newgrp, save_order=True, check_labels=True,
            ):
                new.append(mol)
            else:
                comp.append(mol)
    else:
        for i, datum in enumerate(data):
            if datum.mol.is_subgraph_isomorphic(
                newgrp, save_order=True, check_labels=True,
            ):
                new.append(datum)
            else:
                comp.append(datum)

    return new, comp


def get_extension_edge(
    group,
    items,
    node_children,
    basename,
    n_strucs_min,
    iter_item_cap=np.inf,
    r=None,
    r_bonds=None,
    r_un=None,
    r_site=None,
    r_morph=None,
    r_ncoord=None,
    r_label=None,
    r_lone_pairs=None,
    just_reg_dim=False, #determine reg_dims for group only
    max_ring_gen_size=None,
    soft_iter_max=np.inf,
    hard_iter_max=np.inf,
):
    """
    finds the set of all extension groups to parent such that
    1) the extension group divides the set of reactions under parent
    2) No generalization of the extension group divides the set of reactions under parent

    We find this by generating all possible extensions of the initial group.  Extensions that split reactions are added
    to the list.  All extensions that do not split reactions and do not create bonds are ignored
    (although those that match every reaction are labeled so we don't search them twice).  Those that match
    all reactions and involve bond creation undergo this process again.

    Principle:  Say you have two elementary changes to a group ext1 and ext2 if applying ext1 and ext2 results in a
    split at least one of ext1 and ext2 must result in a split

    Speed of this algorithm relies heavily on searching non bond creation dimensions once.
    """
    if r_bonds is None:
        r_bonds = [1, 2, 3, 1.5, 4]
    if r_un is None:
        r_un = [0, 1, 2, 3]
    if r_site is None:
        r_site = []
    if r_morph is None:
        r_morph = []
    if r_label is None:
        r_label = None
    if r_lone_pairs is None:
        r_lone_pairs = []

    out_exts = [[]]
    grps = [[group]]
    names = [basename]
    first_time = True
    gave_up_split = False

    iter = 0

    while grps[iter] != []:
        grp = grps[iter][-1]

        exts = get_extensions(
            grp,
            basename=names[-1],
            r_full=r,
            r_bonds_full=r_bonds,
            r_un_full=r_un,
            r_site_full=r_site,
            r_morph_full=r_morph,
            r_ncoord_full=r_ncoord,
            r_label=r_label,
            r_lone_pairs_full=r_lone_pairs,
            n_strucs_min=n_strucs_min,
            max_ring_gen_size=max_ring_gen_size,
        )

        reg_dict = dict()
        ext_inds = []
        for i, (grp2, grpc, name, typ, indc) in enumerate(exts):
            if (
                typ != "intNewBridgeExt"
                and typ != "extNewBondExt"
                and (typ, indc) not in reg_dict.keys()
            ):
                # first list is all extensions that match at least one reaction
                # second is extensions that match all reactions
                reg_dict[(typ, indc)] = ([], [])

            new, comp = split_mols(items, grp2)

            if len(new) == 0:
                val = np.inf
                boo = False
            elif len(comp) == 0:
                val = np.inf
                boo = True
            else:
                val = 1.0
                boo = True

            if val != np.inf:
                out_exts[-1].append(
                    exts[i]
                )  # this extension splits reactions (optimization dim)
                if typ == "atomExt":
                    reg_dict[(typ, indc)][0].extend(grp2.atoms[indc[0]].atomtype)
                elif typ == "elExt":
                    reg_dict[(typ, indc)][0].extend(
                        grp2.atoms[indc[0]].radical_electrons
                    )
                elif typ == "lonepairExt":
                    reg_dict[(typ, indc)][0].extend(
                        grp2.atoms[indc[0]].lone_pairs
                    )
                elif typ == "bondExt":
                    reg_dict[(typ, indc)][0].extend(
                        grp2.get_bond(grp2.atoms[indc[0]], grp2.atoms[indc[1]]).order
                    )
                elif typ == "coordExt":
                    reg_dict[(typ, indc)][0].extend(
                        grp2.atoms[indc[0]].props["Ncoord"]
                    )

            elif boo:  # this extension matches all reactions (regularization dim)
                if typ == "intNewBridgeExt" or typ == "extNewBondExt":
                    # these are bond formation extensions, we want to expand these until we get splits
                    ext_inds.append(i)
                elif typ == "atomExt":
                    reg_dict[(typ, indc)][0].extend(grp2.atoms[indc[0]].atomtype)
                    reg_dict[(typ, indc)][1].extend(grp2.atoms[indc[0]].atomtype)
                elif typ == "elExt":
                    reg_dict[(typ, indc)][0].extend(
                        grp2.atoms[indc[0]].radical_electrons
                    )
                    reg_dict[(typ, indc)][1].extend(
                        grp2.atoms[indc[0]].radical_electrons
                    )
                elif typ == "lonepairExt":
                    reg_dict[(typ, indc)][0].extend(
                        grp2.atoms[indc[0]].lone_pairs
                    )
                    reg_dict[(typ, indc)][1].extend(
                        grp2.atoms[indc[0]].lone_pairs
                    )
                elif typ == "bondExt":
                    reg_dict[(typ, indc)][0].extend(
                        grp2.get_bond(grp2.atoms[indc[0]], grp2.atoms[indc[1]]).order
                    )
                    reg_dict[(typ, indc)][1].extend(
                        grp2.get_bond(grp2.atoms[indc[0]], grp2.atoms[indc[1]]).order
                    )
                elif typ == "coordExt":
                    reg_dict[(typ, indc)][0].extend(
                        grp2.atoms[indc[0]].props["Ncoord"]
                    )
                    reg_dict[(typ, indc)][1].extend(
                        grp2.atoms[indc[0]].props["Ncoord"]
                    )
                elif typ == "ringExt":
                    reg_dict[(typ, indc)][1].append(True)
            else:
                # this extension matches no reactions
                if typ == "ringExt":
                    reg_dict[(typ, indc)][0].append(False)
                    reg_dict[(typ, indc)][1].append(False)

        for (
            typr,
            indcr,
        ) in (
            reg_dict.keys()
        ):  # have to label the regularization dimensions in all relevant groups
            reg_val = reg_dict[(typr, indcr)]

            if first_time and not node_children:
                # parent
                if (
                    typr != "intNewBridgeExt" and typr != "extNewBondExt"
                ):  # these dimensions should be regularized
                    if typr == "atomExt":
                        grp.atoms[indcr[0]].reg_dim_atm = list(reg_val)
                    elif typr == "elExt":
                        grp.atoms[indcr[0]].reg_dim_u = list(reg_val)
                    elif typr == "lonepairExt":
                        grp.atoms[indcr[0]].reg_dim_p = list(reg_val)
                    elif typr == "siteExt":
                        grp.atoms[indcr[0]].reg_dim_site = list(reg_val)
                    elif typr == "morphExt":
                        grp.atoms[indcr[0]].reg_dim_morphology = list(reg_val)
                    elif typr == "coordExt":
                        grp.atoms[indcr[0]].reg_dim_ncoord = list(reg_val)
                    elif typr == "ringExt":
                        grp.atoms[indcr[0]].reg_dim_r = list(reg_val)
                    elif typr == "bondExt":
                        atms = grp.atoms
                        bd = grp.get_bond(atms[indcr[0]], atms[indcr[1]])
                        bd.reg_dim = list(reg_val)

            # extensions being sent out
            if (
                typr != "intNewBridgeExt" and typr != "extNewBondExt"
            ):  # these dimensions should be regularized
                for grp2, grpc, name, typ, indc in out_exts[-1]:  # returned groups
                    if typr == "atomExt":
                        grp2.atoms[indcr[0]].reg_dim_atm = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_atm = list(reg_val)
                    elif typr == "elExt":
                        grp2.atoms[indcr[0]].reg_dim_u = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_u = list(reg_val)
                    elif typr == "lonepairExt":
                        grp2.atoms[indcr[0]].reg_dim_p = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_p = list(reg_val)
                    elif typr == "siteExt":
                        grp2.atoms[indcr[0]].reg_dim_site = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_site = list(reg_val)
                    elif typr == "morphExt":
                        grp2.atoms[indcr[0]].reg_dim_morphology = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_morphology = list(reg_val)
                    elif typr == "coordExt":
                        grp2.atoms[indcr[0]].reg_dim_ncoord = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_ncoord = list(reg_val)
                    elif typr == "ringExt":
                        grp2.atoms[indcr[0]].reg_dim_r = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_r = list(reg_val)
                    elif typr == "bondExt":
                        atms = grp2.atoms
                        bd = grp2.get_bond(atms[indcr[0]], atms[indcr[1]])
                        bd.reg_dim = [
                            list(set(bd.order) & set(reg_val[0])),
                            list(set(bd.order) & set(reg_val[1])),
                        ]
                        if grpc:
                            atms = grpc.atoms
                            bd = grpc.get_bond(atms[indcr[0]], atms[indcr[1]])
                            bd.reg_dim = [
                                list(set(bd.order) & set(reg_val[0])),
                                list(set(bd.order) & set(reg_val[1])),
                            ]

        # extensions being expanded
        for (
            typr,
            indcr,
        ) in (
            reg_dict.keys()
        ):  # have to label the regularization dimensions in all relevant groups
            reg_val = reg_dict[(typr, indcr)]
            if (
                typr != "intNewBridgeExt" and typr != "extNewBondExt"
            ):  # these dimensions should be regularized
                for ind2 in ext_inds:  # groups for expansion
                    grp2, grpc, name, typ, indc = exts[ind2]
                    if typr == "atomExt":
                        grp2.atoms[indcr[0]].reg_dim_atm = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_atm = list(reg_val)
                    elif typr == "elExt":
                        grp2.atoms[indcr[0]].reg_dim_u = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_u = list(reg_val)
                    elif typr == "lonepairExt":
                        grp2.atoms[indcr[0]].reg_dim_p = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_p = list(reg_val)
                    elif typr == "siteExt":
                        grp2.atoms[indcr[0]].reg_dim_site = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_site = list(reg_val)
                    elif typr == "morphExt":
                        grp2.atoms[indcr[0]].reg_dim_morphology = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_morphology = list(reg_val)
                    elif typr == "coordExt":
                        grp2.atoms[indcr[0]].reg_dim_ncoord = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_ncoord = list(reg_val)
                    elif typr == "ringExt":
                        grp2.atoms[indcr[0]].reg_dim_r = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_r = list(reg_val)
                    elif typr == "bondExt":
                        atms = grp2.atoms
                        bd = grp2.get_bond(atms[indcr[0]], atms[indcr[1]])
                        bd.reg_dim = [
                            list(set(bd.order) & set(reg_val[0])),
                            list(set(bd.order) & set(reg_val[1])),
                        ]
                        if grpc:
                            atms = grpc.atoms
                            bd = grpc.get_bond(atms[indcr[0]], atms[indcr[1]])
                            bd.reg_dim = [
                                list(set(bd.order) & set(reg_val[0])),
                                list(set(bd.order) & set(reg_val[1])),
                            ]

        out_exts.append([])
        grps[iter].pop()
        names.pop()

        if just_reg_dim:
            return True,None
        
        for ind in ext_inds:  # collect the groups to be expanded
            grpr, grpcr, namer, typr, indcr = exts[ind]
            if len(grps) == iter + 1:
                grps.append([])
            grps[iter + 1].append(grpr)
            names.append(namer)

        if first_time:
            first_time = False

        if (
            not grps[iter] #we've finished this iteration of groups
            and len(grps) != iter + 1 #there are groups to expand)
        ):
            iter += 1
            if not (any([len(x) > 0 for x in out_exts])) and len(grps[iter]) > iter_item_cap:
                logging.error(
                    "Recursion item cap hit not splitting {0} data at iter {1} with {2} items".format(
                        len(items), iter, len(grps[iter])
                    )
                )
                iter -= 1
                gave_up_split = True
            
            elif not (any([len(x) > 0 for x in out_exts])) and iter > hard_iter_max: #we have not found any extensions that split yet and hard_iter_max is violated 
                iter -= 1
                gave_up_split = True
                logging.error("hard_iter_max achieved giving up split")
                
            elif iter > soft_iter_max:
                iter -= 1
                logging.error("soft_iter_max achieved terminating early")
                
            elif any([len(x) > 0 for x in out_exts]):
                iter -= 1
                
    out = []
    # compile all of the valid extensions together
    # may be some duplicates here, but I don't think it's currently worth identifying them
    for x in out_exts:
        out.extend(x)

    return out, gave_up_split


def get_extensions(
    grp,
    r_full=None,
    r_bonds_full=[1, 2, 3, 1.5, 4],
    r_un_full=[0, 1, 2, 3],
    r_site_full=[],
    r_morph_full=[],
    r_ncoord_full=[],
    r_label=[],
    r_lone_pairs_full=[],
    basename="",
    atm_ind=None,
    atm_ind2=None,
    n_strucs_min=None,
    max_ring_gen_size=None,
):
    """
    generate all allowed group extensions and their complements
    note all atomtypes except for elements and r/r!H's must be removed
    """
    # cython.declare(atoms=list, atm=GroupAtom, atm2=GroupAtom, bd=GroupBond, i=int, j=int,
    #                 extents=list, RnH=list, typ=list)
    extents = []

    if n_strucs_min is None:
        n_strucs_min = len(grp.split())

    if isinstance(r_full[0],list):
        r = [x for y in r_full for x in y]
    else:
        r = r_full[:]
    
    if r_bonds_full:
        if isinstance(r_bonds_full[0],list):
            r_bonds = [x for y in r_bonds_full for x in y]
        else:
            r_bonds = r_bonds_full[:]
    
    if r_un_full:
        if isinstance(r_un_full[0],list):
            r_un = [x for y in r_un_full for x in y]
        else:
            r_un = r_un_full[:]
    
    if r_lone_pairs_full:
        if isinstance(r_lone_pairs_full[0],list):
            r_lone_pairs = [x for y in r_lone_pairs_full for x in y]
        else:
            r_lone_pairs = r_lone_pairs_full[:]
            
    if r_site_full:
        if isinstance(r_site_full[0],list):
            r_site = [x for y in r_site_full for x in y]
        else:
            r_site = r_site_full[:]
    
    if r_morph_full:
        if isinstance(r_morph_full[0],list):
            r_morph = [x for y in r_morph_full for x in y]
        else:
            r_morph = r_morph_full[:]
    
    if r_ncoord_full:
        if isinstance(r_ncoord_full[0],list):
            r_ncoord = [x for y in r_ncoord_full for x in y]
        else:
            r_ncoord = r_ncoord_full[:]
    
    if r_label == []:
        r_label = ['']
    
    # generate appropriate r and r!H
    if r is None:
        r = bde_elements  # set of possible r elements/atoms
        r = [ATOMTYPES[x] for x in r]

    if ATOMTYPES["X"] in r and ATOMTYPES["H"] in r:
        RxnH = r[:]
        RxnH.remove(ATOMTYPES["H"])
        R = r[:]
        R.remove(ATOMTYPES["X"])
        RnH = R[:]
        RnH.remove(ATOMTYPES["H"])
    elif ATOMTYPES["H"] in r:
        R = r[:]
        RnH = R[:]
        RnH.remove(ATOMTYPES["H"])
        RxnH = R[:]
        RxnH.remove(ATOMTYPES["H"])
    elif ATOMTYPES["X"] in r:
        RxnH = r[:]
        R = r[:]
        R.remove(ATOMTYPES["X"])
        RnH = R[:]
    else:
        R = r[:]
        RnH = r[:]
        RxnH = r[:]

    atoms = grp.atoms
    if atm_ind is None:
        for i, atm in enumerate(atoms):
            typ = atm.atomtype
            if not atm.reg_dim_atm[0]:
                if len(typ) == 1:
                    if typ[0].label == "R":
                        extents.extend(
                            specify_atom_extensions(grp, i, basename, R, r_full)
                        )  # specify types of atoms
                    elif typ[0].label == "R!H":
                        extents.extend(specify_atom_extensions(grp, i, basename, RnH, r_full))
                    elif typ[0].label == "Rx":
                        extents.extend(specify_atom_extensions(grp, i, basename, r, r_full))
                    elif typ[0].label == "Rx!H":
                        extents.extend(specify_atom_extensions(grp, i, basename, RxnH, r_full))
                else:
                    extents.extend(specify_atom_extensions(grp, i, basename, typ, r_full))
            else:
                if len(typ) == 1:
                    if typ[0].label == "R":
                        extents.extend(
                            specify_atom_extensions(
                                grp, i, basename, atm.reg_dim_atm[0], r_full
                            )
                        )  # specify types of atoms
                    elif typ[0].label == "R!H":
                        extents.extend(
                            specify_atom_extensions(
                                grp, i, basename, list(set(atm.reg_dim_atm[0]) & set(RnH)), r_full
                            )
                        )
                    elif typ[0].label == "Rx":
                        extents.extend(
                            specify_atom_extensions(
                                grp, i, basename, list(set(atm.reg_dim_atm[0]) & set(r)), r_full
                            )
                        )
                    elif typ[0].label == "Rx!H":
                        extents.extend(
                            specify_atom_extensions(
                                grp,
                                i,
                                basename,
                                list(set(atm.reg_dim_atm[0]) & set(RxnH)), r_full
                            )
                        )

                else:
                    extents.extend(
                        specify_atom_extensions(
                            grp, i, basename, list(set(typ) & set(atm.reg_dim_atm[0])), r_full
                        )
                    )
            if r_un_full:
                if not atm.reg_dim_u[0]:
                    if len(atm.radical_electrons) != 1:
                        if len(atm.radical_electrons) == 0:
                            extents.extend(
                                specify_unpaired_extensions(grp, i, basename, r_un, r_un_full)
                            )
                        else:
                            extents.extend(
                                specify_unpaired_extensions(
                                    grp, i, basename, atm.radical_electrons, r_un_full
                                )
                            )
                else:
                    if len(atm.radical_electrons) != 1 and len(atm.reg_dim_u[0]) != 1:
                        if len(atm.radical_electrons) == 0:
                            extents.extend(
                                specify_unpaired_extensions(
                                    grp, i, basename, atm.reg_dim_u[0], r_un_full
                                )
                            )
                        else:
                            extents.extend(
                                specify_unpaired_extensions(
                                    grp,
                                    i,
                                    basename,
                                    list(
                                        set(atm.radical_electrons) & set(atm.reg_dim_u[0])
                                    ),
                                    r_un_full,
                                )
                            )
            if r_lone_pairs_full:
                if not atm.reg_dim_p[0]:
                    if len(atm.lone_pairs) != 1:
                        if len(atm.lone_pairs) == 0:
                            extents.extend(
                                specify_lone_pair_extensions(grp, i, basename, r_lone_pairs, r_lone_pairs_full)
                            )
                        else:
                            extents.extend(
                                specify_lone_pair_extensions(
                                    grp, i, basename, atm.lone_pairs, r_lone_pairs_full
                                )
                            )
                else:
                    if len(atm.lone_pairs) != 1 and len(atm.reg_dim_p[0]) != 1:
                        if len(atm.lone_pairs) == 0:
                            extents.extend(
                                specify_lone_pair_extensions(
                                    grp, i, basename, atm.reg_dim_p[0], r_lone_pairs_full
                                )
                            )
                        else:
                            extents.extend(
                                specify_lone_pair_extensions(
                                    grp,
                                    i,
                                    basename,
                                    list(
                                        set(atm.lone_pairs) & set(atm.reg_dim_p[0])
                                    ),
                                    r_lone_pairs_full,
                                )
                            )
            if r_site_full:
                if not atm.reg_dim_site[0]:
                    if len(atm.site) != 1:
                        if len(atm.site) == 0:
                            extents.extend(
                                specify_site_extensions(grp, i, basename, r_site, r_site_full)
                            )
                        else:
                            extents.extend(
                                specify_site_extensions(grp, i, basename, atm.site, r_site_full)
                            )
                else:
                    if len(atm.site) != 1 and len(atm.reg_dim_site[0]) != 1:
                        if len(atm.site) == 0:
                            extents.extend(
                                specify_site_extensions(
                                    grp, i, basename, atm.reg_dim_site[0], r_site_full
                                )
                            )
                        else:
                            extents.extend(
                                specify_site_extensions(
                                    grp,
                                    i,
                                    basename,
                                    list(set(atm.site) & set(atm.reg_dim_site[0])), r_site_full
                                )
                            )
            if r_morph_full:
                if not atm.reg_dim_morphology[0]:
                    if len(atm.morphology) != 1:
                        if len(atm.morphology) == 0:
                            extents.extend(
                                specify_morphology_extensions(grp, i, basename, r_morph, r_morph_full)
                            )
                        else:
                            extents.extend(
                                specify_morphology_extensions(
                                    grp, i, basename, atm.morphology, r_morph_full
                                )
                            )
                else:
                    if len(atm.morphology) != 1 and len(atm.reg_dim_morphology[0]) != 1:
                        if len(atm.morphology) == 0:
                            extents.extend(
                                specify_morphology_extensions(
                                    grp, i, basename, atm.reg_dim_morphology[0], r_morph_full
                                )
                            )
                        else:
                            extents.extend(
                                specify_morphology_extensions(
                                    grp,
                                    i,
                                    basename,
                                    list(
                                        set(atm.morphology)
                                        & set(atm.reg_dim_morphology[0])
                                    ),
                                    r_morph_full,
                                )
                            )
            if r_ncoord_full:
                if not atm.reg_dim_ncoord[0]:
                    if "Ncoord" not in atm.props.keys() or len(atm.props["Ncoord"]) != 1:
                        if "Ncoord" not in atm.props.keys() or len(atm.props["Ncoord"]) == 0:
                            extents.extend(
                                specify_ncoord_extensions(grp, i, basename, r_ncoord, r_ncoord_full)
                            )
                        else:
                            extents.extend(
                                specify_ncoord_extensions(
                                    grp, i, basename, atm.props["Ncoord"], r_ncoord_full
                                )
                            )
                else:
                    if "Ncoord" not in atm.props.keys() or (len(atm.props["Ncoord"]) != 1 and len(atm.reg_dim_ncoord[0]) != 1):
                        if "Ncoord" not in atm.props.keys() or len(atm.props["Ncoord"]) == 0:
                            extents.extend(
                                specify_ncoord_extensions(
                                    grp, i, basename, atm.reg_dim_ncoord[0], r_ncoord_full
                                )
                            )
                        else:
                            extents.extend(
                                specify_ncoord_extensions(
                                    grp,
                                    i,
                                    basename,
                                    list(
                                        set(atm.props["Ncoord"]) & set(atm.reg_dim_ncoord[0])
                                    ),
                                    r_ncoord_full,
                                )
                            )
            if not atm.reg_dim_r[0] and "inRing" not in atm.props:
                extents.extend(specify_ring_extensions(grp, i, basename))

            extents.extend(
                specify_external_new_bond_extensions(grp, i, basename, r_bonds, r_label)
            )
            for j, atm2 in enumerate(atoms):
                if j <= i and not grp.has_bond(atm, atm2):
                    extents.extend(
                        specify_internal_new_bond_extensions(
                            grp, i, j, n_strucs_min, basename, r_bonds, max_ring_gen_size=max_ring_gen_size,
                        )
                    )
                elif j < i:
                    bd = grp.get_bond(atm, atm2)
                    if len(bd.order) > 1 and not bd.reg_dim[0]:
                        extents.extend(
                            specify_bond_extensions(grp, i, j, basename, bd.order, r_bonds_full)
                        )
                    elif (
                        len(bd.order) > 1
                        and len(bd.reg_dim[0]) > 1
                        and len(bd.reg_dim[0]) > len(bd.reg_dim[1])
                    ):
                        extents.extend(
                            specify_bond_extensions(grp, i, j, basename, bd.reg_dim[0], r_bonds_full)
                        )

    elif (
        atm_ind is not None and atm_ind2 is not None
    ):  # if both atm_ind and atm_ind2 are defined only look at the bonds between them
        i = atm_ind
        j = atm_ind2
        atm = atoms[i]
        atm2 = atoms[j]
        if j <= i and not grp.has_bond(atm, atm2):
            extents.extend(
                specify_internal_new_bond_extensions(
                    grp, i, j, n_strucs_min, basename, r_bonds, max_ring_gen_size=max_ring_gen_size,
                )
            )
        if grp.has_bond(atm, atm2):
            bd = grp.get_bond(atm, atm2)
            if len(bd.order) > 1 and not bd.reg_dim[0]:
                extents.extend(specify_bond_extensions(grp, i, j, basename, bd.order, r_bonds_full))
            elif (
                len(bd.order) > 1
                and len(bd.reg_dim[0]) > 1
                and len(bd.reg_dim[0]) > len(bd.reg_dim[1])
            ):
                extents.extend(
                    specify_bond_extensions(grp, i, j, basename, bd.reg_dim[0], r_bonds_full)
                )

    elif atm_ind is not None:  # look at the atom at atm_ind
        i = atm_ind
        atm = atoms[i]
        typ = atm.atomtype
        if not atm.reg_dim_atm[0]:
            if len(typ) == 1:
                if typ[0].label == "R":
                    extents.extend(
                        specify_atom_extensions(grp, i, basename, R)
                    )  # specify types of atoms
                elif typ[0].label == "R!H":
                    extents.extend(specify_atom_extensions(grp, i, basename, RnH, r_full))
                elif typ[0].label == "Rx":
                    extents.extend(specify_atom_extensions(grp, i, basename, r, r_full))
                elif typ[0].label == "Rx!H":
                    extents.extend(specify_atom_extensions(grp, i, basename, RxnH, r_full))
            else:
                extents.extend(specify_atom_extensions(grp, i, basename, typ, r_full))
        else:
            if len(typ) == 1:
                if typ[0].label == "R":
                    extents.extend(
                        specify_atom_extensions(grp, i, basename, atm.reg_dim_atm[0], r_full)
                    )  # specify types of atoms
                elif typ[0].label == "R!H":
                    extents.extend(
                        specify_atom_extensions(
                            grp, i, basename, list(set(atm.reg_dim_atm[0]) & set(RnH)), r_full
                        )
                    )
                elif typ[0].label == "Rx":
                    extents.extend(
                        specify_atom_extensions(
                            grp, i, basename, list(set(atm.reg_dim_atm[0]) & set(r)), r_full
                        )
                    )
                elif typ[0].label == "Rx!H":
                    extents.extend(
                        specify_atom_extensions(
                            grp, i, basename, list(set(atm.reg_dim_atm[0]) & set(RxnH)), r_full
                        )
                    )
            else:
                extents.extend(
                    specify_atom_extensions(
                        grp, i, basename, list(set(typ) & set(atm.reg_dim_atm[0])), r_full
                    )
                )
        if r_un_full:
            if not atm.reg_dim_u:
                if len(atm.radical_electrons) != 1:
                    if len(atm.radical_electrons) == 0:
                        extents.extend(specify_unpaired_extensions(grp, i, basename, r_un, r_un_full))
                    else:
                        extents.extend(
                            specify_unpaired_extensions(
                                grp, i, basename, atm.radical_electrons, r_un_full
                            )
                        )
            else:
                if len(atm.radical_electrons) != 1 and len(atm.reg_dim_u[0]) != 1:
                    if len(atm.radical_electrons) == 0:
                        extents.extend(
                            specify_unpaired_extensions(grp, i, basename, atm.reg_dim_u[0], r_un_full)
                        )
                    else:
                        extents.extend(
                            specify_unpaired_extensions(
                                grp,
                                i,
                                basename,
                                list(set(atm.radical_electrons) & set(atm.reg_dim_u[0])),
                                r_un_full,
                            )
                        )
        if r_lone_pairs_full:
            if not atm.reg_dim_p:
                if len(atm.lone_pairs) != 1:
                    if len(atm.lone_pairs) == 0:
                        extents.extend(specify_lone_pair_extensions(grp, i, basename, r_lone_pairs, r_lone_pairs_full))
                    else:
                        extents.extend(
                            specify_lone_pair_extensions(
                                grp, i, basename, atm.lone_pairs, r_lone_pairs_full
                            )
                        )
            else:
                if len(atm.lone_pairs) != 1 and len(atm.reg_dim_p[0]) != 1:
                    if len(atm.lone_pairs) == 0:
                        extents.extend(
                            specify_lone_pair_extensions(grp, i, basename, atm.reg_dim_p[0], r_lone_pairs_full)
                        )
                    else:
                        extents.extend(
                            specify_lone_pair_extensions(
                                grp,
                                i,
                                basename,
                                list(set(atm.lone_pairs) & set(atm.reg_dim_p[0])),
                                r_lone_pairs_full,
                            )
                        )
        if r_site_full:
            if not atm.reg_dim_site:
                if len(atm.site) != 1:
                    if len(atm.site) == 0:
                        extents.extend(
                            specify_site_extensions(grp, i, basename, r_site, r_site_full)
                        )
                    else:
                        extents.extend(
                            specify_site_extensions(grp, i, basename, atm.site, r_site_full)
                        )
            else:
                if len(atm.site) != 1 and len(atm.reg_dim_site[0]) != 1:
                    if len(atm.site) == 0:
                        extents.extend(
                            specify_site_extensions(
                                grp, i, basename, atm.reg_dim_site[0], r_site_full
                            )
                        )
                    else:
                        extents.extend(
                            specify_site_extensions(
                                grp,
                                i,
                                basename,
                                list(set(atm.site) & set(atm.reg_dim_site[0])),
                                r_site_full,
                            )
                        )
        if r_morph_full:
            if not atm.reg_dim_morphology:
                if len(atm.morphology) != 1:
                    if len(atm.morphology) == 0:
                        extents.extend(
                            specify_morphology_extensions(grp, i, basename, r_morph, r_morph_full)
                        )
                    else:
                        extents.extend(
                            specify_morphology_extensions(
                                grp, i, basename, atm.morphology, r_morph_full
                            )
                        )
            else:
                if len(atm.morphology) != 1 and len(atm.reg_dim_morphology[0]) != 1:
                    if len(atm.morphology) == 0:
                        extents.extend(
                            specify_morphology_extensions(
                                grp, i, basename, atm.reg_dim_morphology[0], r_morph_full
                            )
                        )
                    else:
                        extents.extend(
                            specify_morphology_extensions(
                                grp,
                                i,
                                basename,
                                list(
                                    set(atm.morphology) & set(atm.reg_dim_morphology[0])
                                ),
                                r_morph_full,
                            )
                        )
        if r_ncoord_full:
            if not atm.reg_dim_ncoord:
                if "Ncoord" not in atm.props.keys() or len(atm.props["Ncoord"]) != 1:
                    if "Ncoord" not in atm.props.keys() or len(atm.props["Ncoord"]) == 0:
                        extents.extend(specify_ncoord_extensions(grp, i, basename, r_ncoord, r_ncoord_full))
                    else:
                        extents.extend(
                            specify_ncoord_extensions(
                                grp, i, basename, atm.props["Ncoord"], r_ncoord_full
                            )
                        )
            else:
                if "Ncoord" not in atm.props.keys() or (len(atm.props["Ncoord"]) != 1 and len(atm.reg_dim_ncoord[0]) != 1):
                    if "Ncoord" not in atm.props.keys() or len(atm.props["Ncoord"]) == 0:
                        extents.extend(
                            specify_ncoord_extensions(grp, i, basename, atm.reg_dim_ncoord[0], r_ncoord_full)
                        )
                    else:
                        extents.extend(
                            specify_ncoord_extensions(
                                grp,
                                i,
                                basename,
                                list(set(atm.props["Ncoord"]) & set(atm.reg_dim_ncoord[0])),
                                r_ncoord_full,
                            )
                        )
        if not atm.reg_dim_r[0] and "inRing" not in atm.props:
            extents.extend(specify_ring_extensions(grp, i, basename))

        extents.extend(specify_external_new_bond_extensions(grp, i, basename, r_bonds, r_label))
        for j, atm2 in enumerate(atoms):
            if j <= i and not grp.has_bond(atm, atm2):
                extents.extend(
                    specify_internal_new_bond_extensions(
                        grp, i, j, n_strucs_min, basename, r_bonds, max_ring_gen_size=max_ring_gen_size,
                    )
                )
            elif j < i:
                bd = grp.get_bond(atm, atm2)
                if len(bd.order) > 1 and not bd.reg_dim:
                    extents.extend(
                        specify_bond_extensions(grp, i, j, basename, bd.order, r_bonds_full)
                    )
                elif (
                    len(bd.order) > 1
                    and len(bd.reg_dim[0]) > 1
                    and len(bd.reg_dim[0]) > len(bd.reg_dim[1])
                ):
                    extents.extend(
                        specify_bond_extensions(grp, i, j, basename, bd.reg_dim[0], r_bonds_full)
                    )

    else:
        raise ValueError("atm_ind must be defined if atm_ind2 is defined")

    for ex in extents:
        ex[0].update_fingerprint()
        if ex[1]:
            ex[1].update_fingerprint()

    return extents

def get_extensions_for_generative_expansion(
    grp,
    tree,
    decomposition,
    r_full,
    r_bonds_full=[1, 2, 3, 1.5, 4],
    r_un_full=[0, 1, 2, 3],
    r_site_full=[],
    r_morph_full=[],
    r_ncoord_full=[],
    r_label=None,
    r_lone_pairs_full=[],
    basename="",
    n_strucs_min=None,
    n_strucs_max=None,
    max_ring_gen_size=None,
    decomposition_associated=None,
    specification_extensions_only=False):
    """
    generate all possible extensions that can be applied to a group structure and roughly estimate changes in prediction
    decomposition must preserve atom ordering relative to grp
    decomposition_associated (f(decomp, i, j=None)): by default we choose "associated" decompositions to estimate based on whether a tagged atom in the decomposition would be changed, this allows "associated" decompositions to be specified by a different function of the decomposition and the atom indexes 
    """

    if n_strucs_min is None:
        n_strucs_min = len(grp.split())
        
    if n_strucs_max is None:
        n_strucs_max = len(grp.split())

    if isinstance(r_full[0],list):
        r = [x for y in r_full for x in y]
    else:
        r = r_full[:]
    
    if r_bonds_full:
        if isinstance(r_bonds_full[0],list):
            r_bonds = [x for y in r_bonds_full for x in y]
        else:
            r_bonds = r_bonds_full[:]
    
    if r_un_full:
        if isinstance(r_un_full[0],list):
            r_un = [x for y in r_un_full for x in y]
        else:
            r_un = r_un_full[:]
    
    if r_lone_pairs_full:
        if isinstance(r_lone_pairs_full[0],list):
            r_lone_pairs = [x for y in r_lone_pairs_full for x in y]
        else:
            r_lone_pairs = r_lone_pairs_full[:]
            
    if r_site_full:
        if isinstance(r_site_full[0],list):
            r_site = [x for y in r_site_full for x in y]
        else:
            r_site = r_site_full[:]
    
    if r_morph_full:
        if isinstance(r_morph_full[0],list):
            r_morph = [x for y in r_morph_full for x in y]
        else:
            r_morph = r_morph_full[:]
    
    if r_ncoord_full:
        if isinstance(r_ncoord_full[0],list):
            r_ncoord = [x for y in r_ncoord_full for x in y]
        else:
            r_ncoord = r_ncoord_full[:]
    
    if r_label is None or r_label == []:
        r_label = ['']
    
    # generate appropriate r and r!H
    if r is None:
        r = bde_elements  # set of possible r elements/atoms
        r = [ATOMTYPES[x] for x in r]

    if ATOMTYPES["X"] in r and ATOMTYPES["H"] in r:
        RxnH = r[:]
        RxnH.remove(ATOMTYPES["H"])
        R = r[:]
        R.remove(ATOMTYPES["X"])
        RnH = R[:]
        RnH.remove(ATOMTYPES["H"])
    elif ATOMTYPES["H"] in r:
        R = r[:]
        RnH = R[:]
        RnH.remove(ATOMTYPES["H"])
        RxnH = R[:]
        RxnH.remove(ATOMTYPES["H"])
    elif ATOMTYPES["X"] in r:
        RxnH = r[:]
        R = r[:]
        R.remove(ATOMTYPES["X"])
        RnH = R[:]
    else:
        R = r[:]
        RnH = r[:]
        RxnH = r[:]

    
    decomps = decomposition(grp)
    assoc_decomposition_init_value_unc_dict = {decomp: evaluate_single(tree, decomp, estimate_uncertainty=True) for decomp in decomps}
    
    atoms = grp.atoms
    
    extents = []
    
    for i, atm in enumerate(atoms):
        
        #find decompositions impacted most by changing this atom and estimate value and uncertainty
        assoc_decomposition_init_i = {decomp:vunc for decomp,vunc in assoc_decomposition_init_value_unc_dict.items() if (decomposition_associated is not None and decomposition_associated(decomp, i)) or (decomposition_associated is None and decomp.atoms[i].label not in ["","*S"])}

        typ = atm.atomtype
        
        if len(typ) == 1:
            if typ[0].label == "R":
                extents.extend(
                    specify_atom_extensions(grp, i, basename, R, r_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
                )  # specify types of atoms
            elif typ[0].label == "R!H":
                extents.extend(specify_atom_extensions(grp, i, basename, RnH, r_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i))
            elif typ[0].label == "Rx":
                extents.extend(specify_atom_extensions(grp, i, basename, r, r_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i))
            elif typ[0].label == "Rx!H":
                extents.extend(specify_atom_extensions(grp, i, basename, RxnH, r_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i))
        else:
            extents.extend(specify_atom_extensions(grp, i, basename, typ, r_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i))
            if not specification_extensions_only and len(typ) < len(r_full):
                extents.extend(generalize_atom_extensions(grp, i, basename, r_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i))
        
        if r_un_full:
            if len(atm.radical_electrons) != 1:
                if len(atm.radical_electrons) == 0:
                    extents.extend(
                        specify_unpaired_extensions(grp, i, basename, r_un, r_un_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
                    )
                else:
                    extents.extend(
                        specify_unpaired_extensions(
                            grp, i, basename, atm.radical_electrons, r_un_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i
                        )
                    )
            if len(atm.radical_electrons) != 0 and len(atm.radical_electrons) < len(r_un_full) and not specification_extensions_only:
                extents.extend(
                    generalize_unpaired_extensions(grp, i, basename, r_un_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
                )

        if r_lone_pairs_full:
            if len(atm.lone_pairs) != 1:
                if len(atm.lone_pairs) == 0:
                    extents.extend(
                        specify_lone_pair_extensions(grp, i, basename, r_lone_pairs, r_lone_pairs_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
                    )
                else:
                    extents.extend(
                        specify_lone_pair_extensions(
                            grp, i, basename, atm.lone_pairs, r_lone_pairs_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i
                        )
                    )
            if len(atm.lone_pairs) != 0 and len(atm.lone_pairs) < len(r_lone_pairs_full) and not specification_extensions_only:
                extents.extend(
                    generalize_lone_pair_extensions(grp, i, basename, r_lone_pairs, r_lone_pairs_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
                )
            
        if r_site_full:
            if len(atm.site) != 1:
                if len(atm.site) == 0:
                    extents.extend(
                        specify_site_extensions(grp, i, basename, r_site, r_site_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
                    )
                else:
                    extents.extend(
                        specify_site_extensions(grp, i, basename, atm.site, r_site_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
                    )
            if len(atm.site) != 0 and len(atm.site) < len(r_site_full) and not specification_extensions_only:
                extents.extend(
                    generalize_site_extensions(grp, i, basename, r_site_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
                )
                
        if r_morph_full:
            if len(atm.morphology) != 1:
                if len(atm.morphology) == 0:
                    extents.extend(
                        specify_morphology_extensions(grp, i, basename, r_morph, r_morph_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
                    )
                else:
                    extents.extend(
                        specify_morphology_extensions(
                            grp, i, basename, atm.morphology, r_morph_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i
                        )
                    )
            if len(atm.morphology) != 0 and len(atm.morphology) < len(r_morph_full) and not specification_extensions_only:
                extents.extend(
                    generalize_morphology_extensions(grp, i, basename, r_morph_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
                )
                
        if r_ncoord_full:
            if "Ncoord" not in atm.props.keys() or len(atm.props["Ncoord"]) != 1:
                if "Ncoord" not in atm.props.keys() or len(atm.props["Ncoord"]) == 0:
                    extents.extend(
                        specify_ncoord_extensions(grp, i, basename, r_ncoord, r_ncoord_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
                    )
                else:
                    extents.extend(
                        specify_ncoord_extensions(
                            grp, i, basename, atm.props["Ncoord"], r_ncoord_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i
                        )
                    )
            if "Ncoord" in atm.props.keys() and (len(atm.props["Ncoord"]) < len(r_ncoord_full)) and len(atm.props["Ncoord"]) != 0 and not specification_extensions_only:
                extents.extend(
                    generalize_ncoord_extensions(grp, i, basename, r_ncoord_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
                )
                
        if "inRing" not in atm.props:
            extents.extend(specify_ring_extensions(grp, i, basename, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i))
        elif not specification_extensions_only:
            extents.extend(generalize_ring_extensions(grp, i, basename, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i))
            
        extents.extend(
            specify_external_new_bond_extensions(grp, i, basename, r_bonds, r_label, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
        )
        
        if not specification_extensions_only:
            extents.extend(
                generalize_remove_atom_extensions(grp, i, basename, n_struc_max=n_strucs_max, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i)
            )
        
        for j, atm2 in enumerate(atoms):
            assoc_decomposition_init_i_j = {decomp:vunc for decomp,vunc in assoc_decomposition_init_value_unc_dict.items() if (decomposition_associated is not None and decomposition_associated(decomp, i, j)) or (decomposition_associated is None and (decomp.atoms[i].label not in ["","*S"] or decomp.atoms[j].label not in ["","*S"]))}
            if j <= i:
                if grp.has_bond(atm, atm2):
                    bd = grp.get_bond(atm, atm2)
                    if len(bd.order) > 1:
                        extents.extend(
                            specify_bond_extensions(grp, i, j, basename, bd.order, r_bonds_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i_j)
                        )
                if not specification_extensions_only:
                    extents.extend(
                        specify_internal_new_bond_extensions(
                            grp, i, j, n_strucs_min, basename, r_bonds, max_ring_gen_size=max_ring_gen_size, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i_j
                        )
                    )
                
                
                    extents.extend(
                        generalize_remove_bridge_extensions(grp, i, j, n_strucs_max=n_strucs_max, basename=basename, r_bonds=r_bonds, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc_dict=assoc_decomposition_init_i_j)
                    )


    for ex in extents:
        ex[0].update_fingerprint()
        if ex[1]:
            ex[1].update_fingerprint()

    return extents

def get_molecular_extensions_for_generative_expansion(
    mol,
    tree,
    decomposition,
    r_full,
    r_bonds_full=[1, 2, 3, 1.5, 4],
    r_un_full=[0, 1, 2, 3],
    r_site_full=[],
    r_morph_full=[],
    r_ncoord_full=[],
    r_label=None,
    r_lone_pairs_full=[],
    basename="",
    n_strucs_min=None,
    n_strucs_max=None,
    max_ring_gen_size=None,
    decomposition_associated=None,
    generate_extensions_from_tree=True,
    generate_local_extensions=True,
    max_heavy_atoms=np.inf,
    max_fused_cluster_rings=np.inf,
    enforce_bredts_rule=False,
    ):
    """
    generate all possible extensions that can be applied to a group structure and roughly estimate changes in prediction
    decomposition must preserve atom ordering relative to grp
    decomposition_associated (f(decomp, i, j=None)): by default we choose "associated" decompositions to estimate based on whether a tagged atom in the decomposition would be changed, this allows "associated" decompositions to be specified by a different function of the decomposition and the atom indexes 
    """

    if n_strucs_min is None:
        n_strucs_min = len(mol.split())
        
    if n_strucs_max is None:
        n_strucs_max = len(mol.split())

    if isinstance(r_full[0],list):
        r = [x for y in r_full for x in y]
    else:
        r = r_full[:]
    
    if r_bonds_full:
        if isinstance(r_bonds_full[0],list):
            r_bonds = [x for y in r_bonds_full for x in y]
        else:
            r_bonds = r_bonds_full[:]
    
    if r_un_full:
        if isinstance(r_un_full[0],list):
            r_un = [x for y in r_un_full for x in y]
        else:
            r_un = r_un_full[:]
    
    if r_lone_pairs_full:
        if isinstance(r_lone_pairs_full[0],list):
            r_lone_pairs = [x for y in r_lone_pairs_full for x in y]
        else:
            r_lone_pairs = r_lone_pairs_full[:]
            
    if r_site_full:
        if isinstance(r_site_full[0],list):
            r_site = [x for y in r_site_full for x in y]
        else:
            r_site = r_site_full[:]
    
    if r_morph_full:
        if isinstance(r_morph_full[0],list):
            r_morph = [x for y in r_morph_full for x in y]
        else:
            r_morph = r_morph_full[:]
    
    if r_ncoord_full:
        if isinstance(r_ncoord_full[0],list):
            r_ncoord = [x for y in r_ncoord_full for x in y]
        else:
            r_ncoord = r_ncoord_full[:]
    
    if r_label is None or r_label == []:
        r_label = ['']
    
    # generate appropriate r and r!H
    if r is None:
        r = bde_elements  # set of possible r elements/atoms
        r = [ATOMTYPES[x] for x in r]

    if ATOMTYPES["X"] in r and ATOMTYPES["H"] in r:
        RxnH = r[:]
        RxnH.remove(ATOMTYPES["H"])
        R = r[:]
        R.remove(ATOMTYPES["X"])
        RnH = R[:]
        RnH.remove(ATOMTYPES["H"])
    elif ATOMTYPES["H"] in r:
        R = r[:]
        RnH = R[:]
        RnH.remove(ATOMTYPES["H"])
        RxnH = R[:]
        RxnH.remove(ATOMTYPES["H"])
    elif ATOMTYPES["X"] in r:
        RxnH = r[:]
        R = r[:]
        R.remove(ATOMTYPES["X"])
        RnH = R[:]
    else:
        R = r[:]
        RnH = r[:]
        RxnH = r[:]

    
    decomps = decomposition(mol)
    assoc_decomposition_init_value_unc = [(decomp,)+evaluate_single(tree, decomp, estimate_uncertainty=True, trace=True) for decomp in decomps]
    atoms = mol.atoms
    
    extents = []
    
    if generate_local_extensions:
        for i, atm in enumerate(atoms):
            
            #find decompositions impacted most by changing this atom and estimate value and uncertainty
            assoc_decomposition_init_i = [tup for tup in assoc_decomposition_init_value_unc if (decomposition_associated is not None and decomposition_associated(tup[0], i)) or (decomposition_associated is None and tup[0].atoms[i].label not in ["","*S"])]
            
            if len(assoc_decomposition_init_i) == 0:
                continue
            
            typ = atm.atomtype
            
            extents.extend(
                molecular_transform_atom_extensions(mol, i, basename, r_full=r_full, r_un_full=r_un_full, r_site_full=r_site_full, r_morph_full=r_morph_full, r_lone_pairs_full=r_lone_pairs_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc=assoc_decomposition_init_i)
            )  # transform atoms
            extents.extend(
                molecular_specify_external_new_bond_extensions(mol, i, basename, r, r_bonds=r_bonds, r_label=r_label, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc=assoc_decomposition_init_i)
            )
            extents.extend(
                molecular_generalize_remove_atom_extensions(mol, i, basename, n_struc_max=n_strucs_max, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc=assoc_decomposition_init_i)
            )
            
            for j, atm2 in enumerate(atoms):
                
                assoc_decomposition_init_i_j = [tup for tup in assoc_decomposition_init_value_unc if (decomposition_associated is not None and decomposition_associated(tup[0], i, j)) or (decomposition_associated is None and (tup[0].atoms[i].label not in ["","*S"] or tup[0].atoms[j].label not in ["","*S"]))]
                
                if len(assoc_decomposition_init_i_j) == 0:
                    continue
                
                if j <= i: 
                    if mol.has_bond(atm, atm2):
                        extents.extend(
                            molecular_transform_bond_extensions(mol, i, j, basename, r_bonds, r_bonds_full, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc=assoc_decomposition_init_i_j)
                        )
                        extents.extend(
                        molecular_generalize_remove_bridge_extensions(mol, i, j, n_strucs_max, basename, r_bonds=r_bonds, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc=assoc_decomposition_init_i_j)
                        )
                    else:
                        extents.extend(
                            molecular_specify_internal_new_bond_extensions(
                                mol, i, j, n_strucs_min, basename, r_bonds, max_ring_gen_size=max_ring_gen_size, tree=tree, estimate_delta=True, assoc_decomposition_init_value_unc=assoc_decomposition_init_i_j
                            )
                        )
    
    if generate_extensions_from_tree:
        for decomp,v,unc,tr in assoc_decomposition_init_value_unc:
            node = tree.nodes[tr]
            structs,tree_nodes,delta,delta_var = molecular_generative_extensions_from_tree_node(decomp,node,element_atomtypes=r)
            extents.extend([(st,None,node.name+"_Treegen","Treegen",None,delta[k],delta_var[k]) for k,st in enumerate(structs)])
    
    unique_extents = []
    for ext in extents:
        if (not np.isinf(max_heavy_atoms) and len([a for a in ext[0].atoms if not a.is_hydrogen()]) > max_heavy_atoms) or (not np.isinf(max_fused_cluster_rings) and get_ring_count_in_largest_fused_ring_system(ext[0]) > max_fused_cluster_rings):
            continue
        if enforce_bredts_rule and invalidated_by_bredts_rule(ext[0]):
            continue
        if ext[0].is_isomorphic(mol,save_order=True,strict=False,check_labels=True,):
            continue
        for uext in unique_extents:
            if ext[0].is_isomorphic(uext[0],save_order=True,strict=False,check_labels=True,):
                break
        else:
            unique_extents.append(ext)

    ext_classes = np.unique([x[-4] for x in unique_extents])
    ext_class_dict = {ext_class:0 for ext_class in ext_classes}
    
    for i,ext in enumerate(unique_extents):
        ext_class_dict[ext[-4]] += 1
    logging.error("Extension class counts:")
    logging.error(ext_class_dict)
    
    return unique_extents

def molecular_transform_atom_extensions(
    mol,
    i,
    basename,
    r_full=None,
    r_un_full=[0, 1, 2, 3],
    r_site_full=[],
    r_morph_full=[],
    r_lone_pairs_full=[],
    tree=None,
    estimate_delta=False,
    assoc_decomposition_init_value_unc=None,
):
    """
    Generate single-atom transform extensions by changing one atom in the original
    group into any other fully specified valid atom configuration.

    This generates transform extensions for Atoms
    """
    extents = []

    if r_full is None:
        r = bde_elements
        r = [ATOMTYPES[x] for x in r]
    elif isinstance(r_full[0], list):
        r = [x for y in r_full for x in y]
    else:
        r = r_full[:]
    
    if r_un_full == []:
        r_un = ['x']
    elif isinstance(r_un_full[0], list):
        r_un = [x for y in r_un_full for x in y]
    else:
        r_un = r_un_full[:]
        
    if r_site_full == []:
        r_site = ['x']
    elif isinstance(r_site_full[0], list):
        r_site = [x for y in r_site_full for x in y]
    else:
        r_site = r_site_full[:]
    if r_morph_full == []:
        r_morph = ['x']
    elif isinstance(r_morph_full[0], list):
        r_morph = [x for y in r_morph_full for x in y]
    else:
        r_morph = r_morph_full[:]
    if r_lone_pairs_full == []:
        r_lone_pairs = ['x']
    elif isinstance(r_lone_pairs_full[0], list):
        r_lone_pairs = [x for y in r_lone_pairs_full for x in y]
    else:
        r_lone_pairs = r_lone_pairs_full[:]
    
    for atomtype,un,site,morph,lone_pairs in itertools.product(r,r_un,r_site,r_morph,r_lone_pairs):
        element = None
        for element_label in allElements:
            if atomtype is ATOMTYPES[element_label] or atomtype in ATOMTYPES[element_label].specific:
                element = element_label
                break
        
        
        atom = mol.atoms[i]
        
        if atom.element.symbol == element and (atom.radical_electrons == un or un == 'x') and (atom.site == site or site == 'x') and (atom.morphology == morph or morph == 'x') and (atom.lone_pairs == lone_pairs or lone_pairs == 'x'):
            continue #same as atom
        
        old_atom_type_str = atom.atomtype.label
        
        m = mol.copy(deep=True)
        
        mapping = {k:m.atoms[k] for k in range(len(m.atoms))}
        
        atom = m.atoms[i]
        
        newatom = Atom(element=element,radical_electrons=un if un != 'x' else 0, lone_pairs=lone_pairs if lone_pairs != 'x' else PeriodicSystem.lone_pairs[element], charge=0,
                       site=site if site != 'x' else '', morphology=morph if morph != 'x' else '')
        
        mapping[i] = newatom
        atom_bonds = {a:bd.order for a,bd in atom.bonds.items()}
        
        atomind = m.atoms.index(atom)
        m.remove_atom(atom)
        m.vertices.insert(atomind,newatom)
        
        for a,order in atom_bonds.items():
            m.add_bond(Bond(newatom,a,order=order))
        
        octet_deviation = get_octet_deviation(newatom) #2/8 minus the number of total electrons

        lone_bonded_atoms = [a for a in newatom.bonds.keys() if len(a.bonds) == 1]
            
        if octet_deviation < 0 and len(lone_bonded_atoms)*2 < abs(octet_deviation):
            continue #we cannot remove enough lone bonded atoms to satisfy
        
        if octet_deviation > 0: #need to add bonds
            while octet_deviation > 1: #1 here because we may leave a radical
                H = Atom('H', radical_electrons=0, lone_pairs=0, charge=0)
                bd = Bond(H,newatom,order=1)
                m.add_atom(H)
                m.add_bond(bd)
                octet_deviation -= 2
        elif octet_deviation < 0: #need to remove bonds
            lone_bond_ind = 0
            while octet_deviation < 0:
                m.remove_atom(lone_bonded_atoms[lone_bond_ind])
                lone_bond_ind += 1
                octet_deviation += 2
        
        
        m.update(sort_atoms=False)
        
        if estimate_delta:
            assert assoc_decomposition_init_value_unc is not None and len(assoc_decomposition_init_value_unc) > 0, "Must provide assoc_decomposition_init_value_unc to estimate delta values for atom extensions"
            assert tree is not None, "Must provide tree to estimate delta values for atom extensions"
            delta_v = None
            delta_var = None
            for decomp,v_init,unc_init,tr in assoc_decomposition_init_value_unc:
                for k,a in mapping.items():
                    a.label = decomp.atoms[k].label
                v,unc = evaluate_single(tree, m, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc**2 - unc_init**2
                else:
                    delta_v += v - v_init
                    delta_var += unc**2 - unc_init**2
            
            m.clear_labeled_atoms()
            
            extents.append(
            (
                m,
                None,
                basename + "_" + str(i + 1) + old_atom_type_str + "->" + str(atomtype),
                "atomTransformExt",
                (i,),
                delta_v,
                delta_var,
            )
            )
                
        else:
            extents.append(
                (
                    m,
                    None,
                    basename + "_" + str(i + 1) + old_atom_type_str + "->" + str(atomtype),
                    "atomTransfromExt",
                    (i,),
                )
            )

    return extents

def specify_atom_extensions(grp, i, basename, r, r_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for specification of the type of atom defined by a given atomtype
    or set of atomtypes
    """
    # cython.declare(grps=list, labelList=list, Rset=set, item=AtomType, grp=Group, grpc=Group, k=AtomType, p=str)

    grps = []
    Rset = set(r)
    if isinstance(r_full[0],list):
        r_spc_full = [[y for y in x if y in r] for x in r_full]
        if len(r_spc_full) == 1:
            r_spc_full = [[x] for x in r_spc_full[0]]
        else:
            r_spc_full += [[x] for x in sum(r_spc_full,[]) if [x] not in r_spc_full]
    else:
        r_spc_full = [[x] for x in r_full if x in r]
    for item in r_spc_full:
        g = deepcopy(grp)
        grpc = deepcopy(grp)
        old_atom_type = g.atoms[i].atomtype
        g.atoms[i].atomtype = item
        grpc.atoms[i].atomtype = list(Rset - set(item))

        if len(grpc.atoms[i].atomtype) == 0:
            grpc = None

        if len(old_atom_type) > 1:
            labelList = []
            old_atom_type_str = ""
            for k in old_atom_type:
                labelList.append(k.label)
            for p in sorted(labelList):
                old_atom_type_str += p
        elif len(old_atom_type) == 0:
            old_atom_type_str = ""
        else:
            old_atom_type_str = old_atom_type[0].label

        if estimate_delta:
            assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for atom extensions"
            assert tree is not None, "Must provide tree to estimate delta values for atom extensions"
            delta_v = None
            delta_var = None
            for decomp,d in assoc_decomposition_init_value_unc_dict.items():
                for i,a in enumerate(decomp.atoms):
                    g.atoms[i].label = a.label
                v_init,unc_init = d
                v,unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc**2 - unc_init**2
                else:
                    delta_v += v - v_init
                    delta_var += unc**2 - unc_init**2
            
            g.clear_labeled_atoms()
            
            grps.append(
            (
                g,
                grpc,
                basename + "_" + str(i + 1) + old_atom_type_str + "->" + "".join([x.label for x in item]),
                "atomExt",
                (i,),
                delta_v,
                delta_var,
            )
        )
                
        else:
            grps.append(
                (
                    g,
                    grpc,
                    basename + "_" + str(i + 1) + old_atom_type_str + "->" + "".join([x.label for x in item]),
                    "atomExt",
                    (i,),
                )
            )

    return grps

def generalize_atom_extensions(grp, i, basename, r, r_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for generalization (making less specific) of the type of atom defined by a given atomtype
    or set of atomtypes. This is the inverse of specify_atom_extensions.
    
    Instead of splitting a general type into specific ones, this combines specific types into less specific groups.
    """
    # cython.declare(grps=list, labelList=list, Rset=set, item=AtomType, grp=Group, grpc=Group, k=AtomType, p=str)

    grps = []
    if isinstance(r_full[0],list):
        for L in r_full:
            if all(a in L for a in grp.atoms[i].atomtype):
                if len(L) > len(grp.atoms[i].atomtype):
                    r_gen = L
                else:
                    r_gen = r
                break
        else:
            r_gen = r
    else:
        r_gen = r
    
    g = deepcopy(grp)
    old_atom_type = g.atoms[i].atomtype
    g.atoms[i].atomtype = r_gen

    if len(old_atom_type) > 1:
        labelList = []
        old_atom_type_str = ""
        for k in old_atom_type:
            labelList.append(k.label)
        for p in sorted(labelList):
            old_atom_type_str += p
    elif len(old_atom_type) == 0:
        old_atom_type_str = ""
    else:
        old_atom_type_str = old_atom_type[0].label

    if estimate_delta:
        assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for atom generalizations"
        assert tree is not None, "Must provide tree to estimate delta values for atom extensions"
        delta_v = None
        delta_var = None
        for decomp,d in assoc_decomposition_init_value_unc_dict.items():
            for i,a in enumerate(decomp.atoms):
                g.atoms[i].label = a.label
            v_init,unc_init = d
            v,unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_var = unc**2 - unc_init**2
                else:
                    delta_var = unc**2 - unc_init**2
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_var += unc**2 - unc_init**2
                else:
                    delta_var += unc**2 - unc_init**2
        
        g.clear_labeled_atoms()
            
        grps.append(
            (
                g,
                None,
                basename + "_" + str(i + 1) + old_atom_type_str + "->" + "".join([x.label for x in r_gen]),
                "atomGen",
                (i,),
                delta_v,
                delta_var,
            )
        )
    else:
        grps.append(
            (
                g,
                None,
                basename + "_" + str(i + 1) + old_atom_type_str + "->" + "".join([x.label for x in r_gen]),
                "atomGen",
                (i,),
            )
        )

    return grps

def specify_ring_extensions(grp, i, basename, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for specifying if an atom is in a ring
    """
    # cython.declare(grps=list, label_list=list, grp=Group, grpc=Group, atom_type=list, atom_type_str=str, k=AtomType,
    #                 p=str)

    grps = []
    label_list = []

    g = deepcopy(grp)
    grpc = deepcopy(grp)
    g.atoms[i].props["inRing"] = True
    grpc.atoms[i].props["inRing"] = False

    atom_type = g.atoms[i].atomtype

    if len(atom_type) > 1:
        atom_type_str = ""
        for k in atom_type:
            label_list.append(k.label)
        for p in sorted(label_list):
            atom_type_str += p
    elif len(atom_type) == 0:
        atom_type_str = ""
    else:
        atom_type_str = atom_type[0].label

    if estimate_delta:
        assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for ring extensions"
        assert tree is not None, "Must provide tree to estimate delta values for atom extensions"
        delta_v = None
        delta_var = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                delta_var = unc ** 2 - unc_init ** 2
            else:
                delta_v += v - v_init
                delta_var += unc ** 2 - unc_init ** 2
                
        g.clear_labeled_atoms()
            
        grps.append(
            (
                g,
                grpc,
                basename + "_" + str(i + 1) + atom_type_str + "-inRing",
                "ringExt",
                (i,),
                delta_v,
                delta_var,
            )
        )
    else:
        grps.append(
            (
                g,
                grpc,
                basename + "_" + str(i + 1) + atom_type_str + "-inRing",
                "ringExt",
                (i,),
            )
        )

    return grps

def generalize_ring_extensions(grp, i, basename, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates generalizations for ring membership of a given atom.
    """
    grps = []
    g = deepcopy(grp)
    if "inRing" not in g.atoms[i].props:
        return []
    old_atom_type = g.atoms[i].atomtype
    del g.atoms[i].props["inRing"]
    grpc = None

    if len(old_atom_type) > 1:
        labelList = [k.label for k in old_atom_type]
        old_atom_type_str = "".join(sorted(labelList))
    elif len(old_atom_type) == 0:
        old_atom_type_str = ""
    else:
        old_atom_type_str = old_atom_type[0].label

    if estimate_delta:
        assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for ring generalizations"
        assert tree is not None, "Must provide tree to estimate delta values for ring generalizations"
        delta_v = None
        delta_var = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_var = unc ** 2 - unc_init ** 2
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_var += unc ** 2 - unc_init ** 2
                else:
                    delta_var += unc ** 2 - unc_init ** 2
                    
        g.clear_labeled_atoms()
            
        grps.append(
            (
                g,
                grpc,
                basename + "_" + str(i + 1) + old_atom_type_str + "->anyRing",
                "ringGen",
                (i,),
                delta_v,
                delta_var,
            )
        )
    else:
        grps.append(
            (
                g,
                grpc,
                basename + "_" + str(i + 1) + old_atom_type_str + "->anyRing",
                "ringGen",
                (i,),
            )
        )

    return grps

def specify_unpaired_extensions(grp, i, basename, r_un, r_un_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for specification of the number of electrons on a given atom
    """

    grps = []
    label_list = []

    Rset = set(r_un)
    if isinstance(r_un_full[0],list):
        r_spc_un_full = [[y for y in x if y in r_un] for x in r_un_full]
        if len(r_spc_un_full) == 1:
            r_spc_un_full = [[x] for x in r_spc_un_full[0]]
        else:
            r_spc_un_full += [[x] for x in sum(r_spc_un_full,[]) if [x] not in r_spc_un_full]
    else:
        r_spc_un_full = [[x] for x in r_un_full if x in r_un]
    for item in r_spc_un_full:
        g = deepcopy(grp)
        grpc = deepcopy(grp)
        g.atoms[i].radical_electrons = item
        grpc.atoms[i].radical_electrons = list(Rset - set(item))

        if len(grpc.atoms[i].radical_electrons) == 0:
            grpc = None

        atom_type = g.atoms[i].atomtype

        if len(atom_type) > 1:
            atom_type_str = ""
            for k in atom_type:
                label_list.append(k.label)
            for p in sorted(label_list):
                atom_type_str += p
        elif len(atom_type) == 0:
            atom_type_str = ""
        else:
            atom_type_str = atom_type[0].label

        if estimate_delta:
            assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for unpaired extensions"
            assert tree is not None, "Must provide tree to estimate delta values for atom extensions"
            delta_v = None
            delta_var = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    g.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_v += v - v_init
                    delta_var += unc ** 2 - unc_init ** 2
            
            g.clear_labeled_atoms()
            
            grps.append((g, grpc, basename + "_" + str(i + 1) + "-u" + "".join([str(x) for x in item]), "elExt", (i,), delta_v, delta_var))
        else:
            grps.append(
                (g, grpc, basename + "_" + str(i + 1) + "-u" + "".join([str(x) for x in item]), "elExt", (i,))
            )

    return grps


def generalize_unpaired_extensions(grp, i, basename, r_un, r_un_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates generalizations for radical electron specification on a given atom.
    """
    grps = []
    if isinstance(r_un_full[0], list):
        for L in r_un_full:
            if all(a in L for a in grp.atoms[i].radical_electrons):
                if len(L) > len(grp.atoms[i].radical_electrons):
                    r_gen = L
                else:
                    r_gen = r_un
                break
        else:
            r_gen = r_un
    else:
        r_gen = r_un

    g = deepcopy(grp)
    grpc = None
    g.atoms[i].radical_electrons = r_gen

    atom_type = g.atoms[i].atomtype
    label_list = []
    if len(atom_type) > 1:
        atom_type_str = ""
        for k in atom_type:
            label_list.append(k.label)
        for p in sorted(label_list):
            atom_type_str += p
    elif len(atom_type) == 0:
        atom_type_str = ""
    else:
        atom_type_str = atom_type[0].label

    if estimate_delta:
        assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for unpaired generalizations"
        assert tree is not None, "Must provide tree to estimate delta values for unpaired generalizations"
        delta_v = None
        delta_var = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_var = unc ** 2 - unc_init ** 2
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_var += unc ** 2 - unc_init ** 2
                else:
                    delta_var += unc ** 2 - unc_init ** 2
        
        g.clear_labeled_atoms()
        
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-u" + "".join([str(x) for x in r_gen]), "elGen", (i,), delta_v, delta_var))
    else:
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-u" + "".join([str(x) for x in r_gen]), "elGen", (i,)))

    return grps

def specify_lone_pair_extensions(grp, i, basename, r_lone_pairs, r_lone_pairs_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for specification of the number of lone pairs on a given atom
    """

    grps = []
    label_list = []

    Rset = set(r_lone_pairs)
    if isinstance(r_lone_pairs_full[0],list):
        r_spc_lone_pairs_full = [[y for y in x if y in r_lone_pairs] for x in r_lone_pairs_full]
        if len(r_spc_lone_pairs_full) == 1:
            r_spc_lone_pairs_full = [[x] for x in r_spc_lone_pairs_full[0]]
        else:
            r_spc_lone_pairs_full += [[x] for x in sum(r_spc_lone_pairs_full,[]) if [x] not in r_spc_lone_pairs_full]
    else:
        r_spc_lone_pairs_full = [[x] for x in r_lone_pairs_full if x in r_lone_pairs]
    for item in r_spc_lone_pairs_full:
        g = deepcopy(grp)
        grpc = deepcopy(grp)
        g.atoms[i].lone_pairs = item
        grpc.atoms[i].lone_pairs = list(Rset - set(item))

        if len(grpc.atoms[i].lone_pairs) == 0:
            grpc = None

        atom_type = g.atoms[i].atomtype

        if len(atom_type) > 1:
            atom_type_str = ""
            for k in atom_type:
                label_list.append(k.label)
            for p in sorted(label_list):
                atom_type_str += p
        elif len(atom_type) == 0:
            atom_type_str = ""
        else:
            atom_type_str = atom_type[0].label

        if estimate_delta:
            assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for lone-pair extensions"
            assert tree is not None, "Must provide tree to estimate delta values for atom extensions"
            delta_v = None
            delta_var = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    g.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_v += v - v_init
                    delta_var += unc ** 2 - unc_init ** 2
                    
            g.clear_labeled_atoms()
            
            grps.append((g, grpc, basename + "_" + str(i + 1) + "-p" + "".join([str(x) for x in item]), "lonepairExt", (i,), delta_v, delta_var))
        else:
            grps.append(
                (g, grpc, basename + "_" + str(i + 1) + "-p" + "".join([str(x) for x in item]), "lonepairExt", (i,))
            )

    return grps

def generalize_lone_pair_extensions(grp, i, basename, r_lone_pairs, r_lone_pairs_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates generalizations for lone pair specification on a given atom.
    """
    grps = []
    if isinstance(r_lone_pairs_full[0], list):
        for L in r_lone_pairs_full:
            if all(a in L for a in grp.atoms[i].lone_pairs):
                if len(L) > len(grp.atoms[i].lone_pairs):
                    r_gen = L
                else:
                    r_gen = r_lone_pairs
                break
        else:
            r_gen = r_lone_pairs
    else:
        r_gen = r_lone_pairs

    g = deepcopy(grp)
    grpc = None
    g.atoms[i].lone_pairs = r_gen

    atom_type = g.atoms[i].atomtype
    label_list = []
    if len(atom_type) > 1:
        atom_type_str = ""
        for k in atom_type:
            label_list.append(k.label)
        for p in sorted(label_list):
            atom_type_str += p
    elif len(atom_type) == 0:
        atom_type_str = ""
    else:
        atom_type_str = atom_type[0].label

    if estimate_delta:
        assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for lone-pair generalizations"
        assert tree is not None, "Must provide tree to estimate delta values for lone-pair generalizations"
        delta_v = None
        delta_var = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_var = unc ** 2 - unc_init ** 2
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_var += unc ** 2 - unc_init ** 2
                else:
                    delta_var += unc ** 2 - unc_init ** 2
        
        g.clear_labeled_atoms()
        
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-p" + "".join([str(x) for x in r_gen]), "lonepairGen", (i,), delta_v, delta_var))
    else:
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-p" + "".join([str(x) for x in r_gen]), "lonepairGen", (i,)))

    return grps

def specify_site_extensions(grp, i, basename, r_site, r_site_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for specification of the number of electrons on a given atom
    """
    # x = ATOMTYPES["X"]
    # if (
    #     not any([s.is_specific_case_of(x) for s in grp.atoms[i].atomtype])
    #     and grp.atoms[i].atomtype[0] != ATOMTYPES["Rx"]
    # ):
    #     return []

    grps = []
    label_list = []

    Rset = set(r_site)
    if isinstance(r_site_full[0],list):
        r_spc_site_full = [[y for y in x if y in r_site] for x in r_site_full]
        if len(r_spc_site_full) == 1:
            r_spc_site_full = [[x] for x in r_spc_site_full[0]]
        else:
            r_spc_site_full += [[x] for x in sum(r_spc_site_full,[]) if [x] not in r_spc_site_full]
    else:
        r_spc_site_full = [[x] for x in r_site_full if x in r_site]
    for item in r_spc_site_full:
        g = deepcopy(grp)
        grpc = deepcopy(grp)
        g.atoms[i].site = item
        grpc.atoms[i].site = list(Rset - set(item))

        if len(grpc.atoms[i].site) == 0:
            grpc = None

        atom_type = g.atoms[i].atomtype

        if len(atom_type) > 1:
            atom_type_str = ""
            for k in atom_type:
                label_list.append(k.label)
            for p in sorted(label_list):
                atom_type_str += p
        elif len(atom_type) == 0:
            atom_type_str = ""
        else:
            atom_type_str = atom_type[0].label

        if estimate_delta:
            assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for site extensions"
            assert tree is not None, "Must provide tree to estimate delta values for atom extensions"
            delta_v = None
            delta_var = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    g.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_v += v - v_init
                    delta_var += unc ** 2 - unc_init ** 2
                    
            g.clear_labeled_atoms()
            
            grps.append((g, grpc, basename + "_" + str(i + 1) + "-s" + "".join([str(x) for x in item]), "siteExt", (i,), delta_v, delta_var))
        else:
            grps.append(
                (g, grpc, basename + "_" + str(i + 1) + "-s" + "".join([str(x) for x in item]), "siteExt", (i,))
            )

    return grps

def generalize_site_extensions(grp, i, basename, r_site, r_site_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates generalizations for site specification on a given atom.
    """
    grps = []
    if isinstance(r_site_full[0], list):
        for L in r_site_full:
            if all(a in L for a in grp.atoms[i].site):
                if len(L) > len(grp.atoms[i].site):
                    r_gen = L
                else:
                    r_gen = r_site
                break
        else:
            r_gen = r_site
    else:
        r_gen = r_site

    g = deepcopy(grp)
    grpc = None
    g.atoms[i].site = r_gen

    atom_type = g.atoms[i].atomtype
    label_list = []
    if len(atom_type) > 1:
        atom_type_str = ""
        for k in atom_type:
            label_list.append(k.label)
        for p in sorted(label_list):
            atom_type_str += p
    elif len(atom_type) == 0:
        atom_type_str = ""
    else:
        atom_type_str = atom_type[0].label

    if estimate_delta:
        assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for site generalizations"
        assert tree is not None, "Must provide tree to estimate delta values for site generalizations"
        delta_v = None
        delta_var = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_var = unc ** 2 - unc_init ** 2
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_var += unc ** 2 - unc_init ** 2
                else:
                    delta_var += unc ** 2 - unc_init ** 2
        
        g.clear_labeled_atoms()
        
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-s" + "".join([str(x) for x in r_gen]), "siteGen", (i,), delta_v, delta_var))
    else:
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-s" + "".join([str(x) for x in r_gen]), "siteGen", (i,)))

    return grps

def specify_morphology_extensions(grp, i, basename, r_morph, r_morph_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for specification of the number of electrons on a given atom
    """
    # x = ATOMTYPES["X"]
    # if (
    #     not any([s.is_specific_case_of(x) for s in grp.atoms[i].atomtype])
    #     and grp.atoms[i].atomtype[0] != ATOMTYPES["Rx"]
    # ):
    #     return []

    grps = []
    label_list = []

    Rset = set(r_morph)
    if isinstance(r_morph_full[0],list):
        r_spc_morph_full = [[y for y in x if y in r_morph] for x in r_morph_full]
        if len(r_spc_morph_full) == 1:
            r_spc_morph_full = [[x] for x in r_spc_morph_full[0]]
        else:
            r_spc_morph_full += [[x] for x in sum(r_spc_morph_full,[]) if [x] not in r_spc_morph_full]
    else:
        r_spc_morph_full = [[x] for x in r_morph_full if x in r_morph]
    for item in r_spc_morph_full:
        g = deepcopy(grp)
        grpc = deepcopy(grp)
        g.atoms[i].morphology = item
        grpc.atoms[i].morphology = list(Rset - set(item))

        if len(grpc.atoms[i].morphology) == 0:
            grpc = None

        atom_type = g.atoms[i].atomtype

        if len(atom_type) > 1:
            atom_type_str = ""
            for k in atom_type:
                label_list.append(k.label)
            for p in sorted(label_list):
                atom_type_str += p
        elif len(atom_type) == 0:
            atom_type_str = ""
        else:
            atom_type_str = atom_type[0].label

        if estimate_delta:
            assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for morphology extensions"
            assert tree is not None, "Must provide tree to estimate delta values for morphology extensions"
            delta_v = None
            delta_var = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    g.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_v += v - v_init
                    delta_var += unc ** 2 - unc_init ** 2
            
            g.clear_labeled_atoms()
            
            grps.append((g, grpc, basename + "_" + str(i + 1) + "-m" + "".join([str(x) for x in item]), "morphExt", (i,), delta_v, delta_var))
        else:
            grps.append(
                (
                    g,
                    grpc,
                    basename + "_" + str(i + 1) + "-m" + "".join([str(x) for x in item]),
                    "morphExt",
                    (i,),
                )
            )

    return grps

def generalize_morphology_extensions(grp, i, basename, r_morph, r_morph_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates generalizations for morphology specification on a given atom.
    """
    grps = []
    if isinstance(r_morph_full[0], list):
        for L in r_morph_full:
            if all(a in L for a in grp.atoms[i].morphology):
                if len(L) > len(grp.atoms[i].morphology):
                    r_gen = L
                else:
                    r_gen = r_morph
                break
        else:
            r_gen = r_morph
    else:
        r_gen = r_morph

    g = deepcopy(grp)
    grpc = None
    g.atoms[i].morphology = r_gen

    atom_type = g.atoms[i].atomtype
    label_list = []
    if len(atom_type) > 1:
        atom_type_str = ""
        for k in atom_type:
            label_list.append(k.label)
        for p in sorted(label_list):
            atom_type_str += p
    elif len(atom_type) == 0:
        atom_type_str = ""
    else:
        atom_type_str = atom_type[0].label

    if estimate_delta:
        assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for morphology generalizations"
        assert tree is not None, "Must provide tree to estimate delta values for morphology generalizations"
        delta_v = None
        delta_var = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_var = unc ** 2 - unc_init ** 2
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_var += unc ** 2 - unc_init ** 2
                else:
                    delta_var += unc ** 2 - unc_init ** 2
        
        g.clear_labeled_atoms()

        grps.append((g, grpc, basename + "_" + str(i + 1) + "-m" + "".join([str(x) for x in r_gen]), "morphGen", (i,), delta_v, delta_var))
    else:
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-m" + "".join([str(x) for x in r_gen]), "morphGen", (i,)))

    return grps

def specify_ncoord_extensions(grp, i, basename, r_ncoord, r_ncoord_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for specification of the number of electrons on a given atom
    """

    grps = []
    label_list = []

    Rset = set(r_ncoord)
    if isinstance(r_ncoord_full,list):
        r_spc_ncoord_full = [[y for y in x if y in r_ncoord] for x in r_ncoord_full]
        if len(r_spc_ncoord_full) == 1:
            r_spc_ncoord_full = [[x] for x in r_spc_ncoord_full[0]]
        else:
            r_spc_ncoord_full += [[x] for x in sum(r_spc_ncoord_full,[]) if [x] not in r_spc_ncoord_full]
    else:
        r_spc_ncoord_full = [[x] for x in r_ncoord_full if x in r_ncoord]
    for item in r_spc_ncoord_full:
        g = deepcopy(grp)
        grpc = deepcopy(grp)
        g.atoms[i].props["Ncoord"] = item
        grpc.atoms[i].props["Ncoord"] = list(Rset - set(item))

        if len(grpc.atoms[i].props["Ncoord"]) == 0:
            grpc = None

        atom_type = g.atoms[i].atomtype

        if len(atom_type) > 1:
            atom_type_str = ""
            for k in atom_type:
                label_list.append(k.label)
            for p in sorted(label_list):
                atom_type_str += p
        elif len(atom_type) == 0:
            atom_type_str = ""
        else:
            atom_type_str = atom_type[0].label

        if estimate_delta:
            assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for ncoord extensions"
            assert tree is not None, "Must provide tree to estimate delta values for ncoord extensions"
            delta_v = None
            delta_var = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    g.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_v += v - v_init
                    delta_var += unc ** 2 - unc_init ** 2
            
            g.clear_labeled_atoms()
            
            grps.append((g, grpc, basename + "_" + str(i + 1) + "-n" + "".join([str(x) for x in item]), "coordExt", (i,), delta_v, delta_var))
        else:
            grps.append(
                (g, grpc, basename + "_" + str(i + 1) + "-n" + "".join([str(x) for x in item]), "coordExt", (i,))
            )

    return grps

def generalize_ncoord_extensions(grp, i, basename, r_ncoord, r_ncoord_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates generalizations for coordination number specification on a given atom.
    """
    grps = []
    if isinstance(r_ncoord_full[0], list):
        for L in r_ncoord_full:
            if all(a in L for a in grp.atoms[i].props.get("Ncoord", [])):
                if len(L) > len(grp.atoms[i].props.get("Ncoord", [])):
                    r_gen = L
                else:
                    r_gen = r_ncoord
                break
        else:
            r_gen = r_ncoord
    else:
        r_gen = r_ncoord

    g = deepcopy(grp)
    grpc = None
    g.atoms[i].props["Ncoord"] = r_gen

    atom_type = g.atoms[i].atomtype
    label_list = []
    if len(atom_type) > 1:
        atom_type_str = ""
        for k in atom_type:
            label_list.append(k.label)
        for p in sorted(label_list):
            atom_type_str += p
    elif len(atom_type) == 0:
        atom_type_str = ""
    else:
        atom_type_str = atom_type[0].label

    if estimate_delta:
        assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for ncoord generalizations"
        assert tree is not None, "Must provide tree to estimate delta values for ncoord generalizations"
        delta_v = None
        delta_var = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_var = unc ** 2 - unc_init ** 2
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_var += unc ** 2 - unc_init ** 2
                else:
                    delta_var += unc ** 2 - unc_init ** 2
                    
        g.clear_labeled_atoms()
        
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-n" + "".join([str(x) for x in r_gen]), "coordGen", (i,), delta_v, delta_var))
    else:
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-n" + "".join([str(x) for x in r_gen]), "coordGen", (i,)))

    return grps

def specify_internal_new_bond_extensions(grp, i, j, n_strucs_min, basename, r_bonds, max_ring_gen_size=None, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for creation of a bond (of undefined order)
    between two atoms indexed i,j that already exist in the group and are unbonded
    """
    # cython.declare(newgrp=Group)
    if i == j:
        if max_ring_gen_size is None:
            return []
        pathlen = 1
    else:
        paths = find_shortest_paths(grp.atoms[i],grp.atoms[j])
        if paths is None:
            pathlen = None
        else:
            pathlen = len(paths[0])
    
    if pathlen is None and n_strucs_min == len(grp.split()): #internal bridge will reduce below minimum number of independent structures
        return []
    
    atom_type_i = grp.atoms[i].atomtype
    atom_type_j = grp.atoms[j].atomtype

    if len(atom_type_i) > 1:
        atom_type_i_str = ""
        label_list_i = [k.label for k in atom_type_i]
        for k in sorted(label_list_i):
            atom_type_i_str += k
    elif len(atom_type_i) == 0:
        atom_type_i_str = ""
    else:
        atom_type_i_str = atom_type_i[0].label
    if len(atom_type_j) > 1:
        atom_type_j_str = ""
        label_list_j = [k.label for k in atom_type_j]
        for p in sorted(label_list_j):
            atom_type_j_str += p
    elif len(atom_type_j) == 0:
        atom_type_j_str = ""
    else:
        atom_type_j_str = atom_type_j[0].label
        
    if max_ring_gen_size is None: #just internal bond extensions
        newgrp = deepcopy(grp)
        newgrp.add_bond(GroupBond(newgrp.atoms[i], newgrp.atoms[j], r_bonds))

        if estimate_delta:
            assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for internal new-bond extensions"
            assert tree is not None, "Must provide tree to estimate delta values for internal new-bond extensions"
            delta_v = None
            delta_var = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    newgrp.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, newgrp, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_v += v - v_init
                    delta_var += unc ** 2 - unc_init ** 2

            newgrp.clear_labeled_atoms()
            
            return [
                (
                    newgrp,
                    None,
                    basename
                    + "_Int-"
                    + str(i + 1)
                    + atom_type_i_str
                    + "-"
                    + str(j + 1)
                    + atom_type_j_str
                    + "-Br0",
                    "intNewBridgeExt",
                    (i, j),
                    delta_v,
                    delta_var,
                )
            ]
        else:
            return [
                (
                    newgrp,
                    None,
                    basename
                    + "_Int-"
                    + str(i + 1)
                    + atom_type_i_str
                    + "-"
                    + str(j + 1)
                    + atom_type_j_str
                    + "-Br0",
                    "intNewBridgeExt",
                    (i, j),
                )
            ]
    else:
        grps = []
        for bridgelen in range(max_ring_gen_size-pathlen+1): #includes bridgelen == 0
            if i == j and bridgelen < 2: #this is no change from the original group or external bond creation
                continue
            if bridgelen == 0 and grp.has_bond(grp.atoms[i],grp.atoms[j]): #the bridging bond already exists
                continue
            newgrp = deepcopy(grp)
            tail_atom = newgrp.atoms[i]
            head_atom = newgrp.atoms[j]
            for k in range(bridgelen): #create first ring
                newatm = GroupAtom([ATOMTYPES['R!H']])
                newgrp.add_atom(newatm)
                bd = GroupBond(tail_atom,newatm,order=r_bonds)
                newgrp.add_bond(bd)
                tail_atom = newatm
            else:
                bd = GroupBond(tail_atom,head_atom,order=r_bonds)
                newgrp.add_bond(bd)
            
            if estimate_delta:
                assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for internal new-bond extensions"
                assert tree is not None, "Must provide tree to estimate delta values for internal new-bond extensions"
                delta_v = None
                delta_var = None
                for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                    for k, a in enumerate(decomp.atoms):
                        newgrp.atoms[k].label = a.label
                    v_init, unc_init = d
                    v, unc = evaluate_single(tree, newgrp, estimate_uncertainty=True)
                    if delta_v is None:
                        delta_v = v - v_init
                        delta_var = unc ** 2 - unc_init ** 2
                    else:
                        delta_v += v - v_init
                        delta_var += unc ** 2 - unc_init ** 2

                newgrp.clear_labeled_atoms()

                grps.append((
                    newgrp,
                    None,
                    basename
                    + "_Int-"
                    + str(i + 1)
                    + atom_type_i_str
                    + "-"
                    + str(j + 1)
                    + atom_type_j_str
                    + "-Br"+str(bridgelen),
                    "intNewBridgeExt",
                    (i, j),
                    delta_v,
                    delta_var,
                ))
            else:
                grps.append((
                    newgrp,
                    None,
                    basename
                    + "_Int-"
                    + str(i + 1)
                    + atom_type_i_str
                    + "-"
                    + str(j + 1)
                    + atom_type_j_str
                    + "-Br"+str(bridgelen),
                    "intNewBridgeExt",
                    (i, j),
                ))
    
    return grps    

def molecular_specify_internal_new_bond_extensions(mol, i, j, n_strucs_min, basename, r_bonds, max_ring_gen_size=None, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc=None):
    """
    generates extensions for creation of a bond (of undefined order)
    between two atoms indexed i,j that already exist in the group and are unbonded
    """
    # cython.declare(newgrp=Group)
    if i == j:
        if max_ring_gen_size is None:
            return []
        pathlen = 1
    else:
        paths = find_shortest_paths(mol.atoms[i],mol.atoms[j])
        if paths is None:
            pathlen = None
        else:
            pathlen = len(paths[0])
    
    if pathlen is None and n_strucs_min == len(mol.split()): #internal bridge will reduce below minimum number of independent structures
        return []
    elif pathlen is None and max_ring_gen_size != 2: #this choice allows it to create the connection, but at no length
        pathlen = max_ring_gen_size 
    elif pathlen is None and max_ring_gen_size == 2:
        pathlen = 1
    
    atom_type_i = mol.atoms[i].atomtype
    atom_i_lone_bonded_atoms = [a for a in mol.atoms[i].bonds.keys() if len(a.bonds)==1]
    atom_type_j = mol.atoms[j].atomtype
    atom_j_lone_bonded_atoms = [a for a in mol.atoms[j].bonds.keys() if len(a.bonds)==1]

    if atom_i_lone_bonded_atoms == [] or atom_j_lone_bonded_atoms == []: #cannot easily remove H or Val7 to make bond
        return []
    
    atom_type_i_str = atom_type_i.label
    atom_type_j_str = atom_type_j.label
        
    grps = []
    for bridgelen in range(max_ring_gen_size-pathlen+1): #includes bridgelen == 0
        if i == j and bridgelen < 2: #this is no change from the original group or external bond creation
            continue
        if bridgelen == 0 and pathlen == 2: #bond already exists
            continue
        newmol = mol.copy(deep=True)
        mapping = {k: newmol.atoms[k] for k in range(len(newmol.atoms))}
        
        tail_atom = newmol.atoms[i]
        head_atom = newmol.atoms[j]
        
        tail_atom_remove_atom = [a for a in tail_atom.bonds.keys() if len(a.bonds) == 1]
        if len(tail_atom_remove_atom) == 0:
            break
        else:
            tail_atom_remove_atom = tail_atom_remove_atom[0]
            
        newmol.remove_atom(tail_atom_remove_atom)
        
        head_atom_remove_atom = [a for a in head_atom.bonds.keys() if len(a.bonds) == 1]
        if len(head_atom_remove_atom) == 0:
            break
        else:
            head_atom_remove_atom = head_atom_remove_atom[0]

        for k in range(bridgelen): #create first ring
            newatm = Atom('C',radical_electrons=0,lone_pairs=0,charge=0)
            bd = Bond(tail_atom,newatm,order=1)
            H1 = Atom('H', radical_electrons=0, lone_pairs=0, charge=0)
            bdH1 = Bond(H1,newatm,order=1)
            H2 = Atom('H', radical_electrons=0, lone_pairs=0, charge=0)
            bdH2 = Bond(H2,newatm,order=1)
            newmol.add_atom(newatm)
            newmol.add_bond(bd)
            newmol.add_atom(H1)
            newmol.add_bond(bdH1)
            newmol.add_atom(H2)
            newmol.add_bond(bdH2)
            tail_atom = newatm
        else:
            newmol.remove_atom(head_atom_remove_atom)
            bd = Bond(tail_atom,head_atom,order=1)
            newmol.add_bond(bd)
        
        newmol.update(sort_atoms=False)
        
        if estimate_delta:
            assert assoc_decomposition_init_value_unc is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for internal new-bond extensions"
            assert tree is not None, "Must provide tree to estimate delta values for internal new-bond extensions"
            delta_v = None
            delta_var = None
            for decomp,v_init,unc_init,tr in assoc_decomposition_init_value_unc:
                for k, anew in mapping.items(): 
                    anew.label = decomp.atoms[k].label
                v, unc = evaluate_single(tree, newmol, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_v += v - v_init
                    delta_var += unc ** 2 - unc_init ** 2

            newmol.clear_labeled_atoms()

            grps.append((
                newmol,
                None,
                basename
                + "_Int-"
                + str(i + 1)
                + atom_type_i_str
                + "-"
                + str(j + 1)
                + atom_type_j_str
                + "-Br"+str(bridgelen),
                "intNewBridgeExt",
                (i, j),
                delta_v,
                delta_var,
            ))
        else:
            grps.append((
                newmol,
                None,
                basename
                + "_Int-"
                + str(i + 1)
                + atom_type_i_str
                + "-"
                + str(j + 1)
                + atom_type_j_str
                + "-Br"+str(bridgelen),
                "intNewBridgeExt",
                (i, j),
            ))
    
    return grps

def generalize_remove_bridge_extensions(grp, i, j, n_strucs_max, basename, r_bonds, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generalizes extensions by removing the shortest path (atoms/bonds)
    between two atoms indexed i,j that already exist in the group
    """
    # cython.declare(newgrp=Group)
    
    paths = find_shortest_paths(grp.atoms[i],grp.atoms[j])
    
    if paths is None:
        return []
    else:
        paths = [[grp.atoms.index(a) for a in p] for p in paths] #convert paths from lists of atoms to lists of atom indices
    
    atom_type_i = grp.atoms[i].atomtype
    atom_type_j = grp.atoms[j].atomtype

    if len(atom_type_i) > 1:
        atom_type_i_str = ""
        label_list_i = [k.label for k in atom_type_i]
        for k in sorted(label_list_i):
            atom_type_i_str += k
    elif len(atom_type_i) == 0:
        atom_type_i_str = ""
    else:
        atom_type_i_str = atom_type_i[0].label
    if len(atom_type_j) > 1:
        atom_type_j_str = ""
        label_list_j = [k.label for k in atom_type_j]
        for p in sorted(label_list_j):
            atom_type_j_str += p
    elif len(atom_type_j) == 0:
        atom_type_j_str = ""
    else:
        atom_type_j_str = atom_type_j[0].label
    
    grps = []
    for path in paths:
        newgrp = deepcopy(grp)
        mapping = {a:newgrp.atoms[q] for q,a in enumerate(grp.atoms)}
        tail_atom = newgrp.atoms[i]
        head_atom = newgrp.atoms[j]
        if len(path) == 2: #just remove bond
            newgrp.remove_bond(newgrp.get_bond(newgrp.atoms[i], newgrp.atoms[j]))
        else: #remove internal atoms and bonds
            for a in [newgrp.atoms[q] for q in path]:
                if a is not tail_atom and a is not head_atom:
                    newgrp.remove_atom(a)
        
        if len(newgrp.split()) > n_strucs_max: #removing that path creates too many separate structures
            continue
        
        if estimate_delta:
            assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for internal new-bond extensions"
            assert tree is not None, "Must provide tree to estimate delta values for internal new-bond extensions"
            delta_v = None
            delta_var = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                missing = False
                for k, a in enumerate(decomp.atoms):
                    if grp.atoms[k] in mapping.keys():                
                        mapping[grp.atoms[k]].label = a.label
                    elif a.label not in ["","*S"]: #we cannot map an important label for this decomposition
                        missing = True
                        break
                
                v_init, unc_init = d
                
                if missing:
                    v, unc = 0.0,0.0
                else:
                    v, unc = evaluate_single(tree, newgrp, estimate_uncertainty=True)

                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_v += v - v_init
                    delta_var += unc ** 2 - unc_init ** 2
                
            newgrp.clear_labeled_atoms()

            grps.append((
                newgrp,
                None,
                basename
                + "_Int-"
                + str(i + 1)
                + atom_type_i_str
                + "-"
                + str(j + 1)
                + atom_type_j_str
                + "-Br"+str(len(path)),
                "genRemoveBridgeExt",
                (i, j),
                delta_v,
                delta_var,
            ))
        else:
            grps.append((
                newgrp,
                None,
                basename
                + "_Int-"
                + str(i + 1)
                + atom_type_i_str
                + "-"
                + str(j + 1)
                + atom_type_j_str
                + "-Br"+str(len(path)),
                "genRemoveBridgeExt",
                (i, j),
            ))
    
    return grps

def molecular_generalize_remove_bridge_extensions(mol, i, j, n_strucs_max, basename, r_bonds, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc=None):
    """
    generalizes extensions by removing the shortest path (atoms/bonds)
    between two atoms indexed i,j that already exist in the group
    """
    # cython.declare(newgrp=Group)
    
    paths = find_shortest_paths(mol.atoms[i],mol.atoms[j])
    
    if paths is None:
        return []
    else:
        paths = [[mol.atoms.index(a) for a in p] for p in paths] #convert paths from lists of atoms to lists of atom indices
    
    atom_type_i = mol.atoms[i].atomtype
    atom_type_j = mol.atoms[j].atomtype

    
    atom_type_i_str = atom_type_i.label
    atom_type_j_str = atom_type_j.label
    
    mols = []
    for path in paths:
        newmol = mol.copy(deep=True)
        mapping = {a:newmol.atoms[q] for q,a in enumerate(mol.atoms)}
        tail_atom = newmol.atoms[i]
        head_atom = newmol.atoms[j]
        if len(path) == 2: #just remove bond
            newmol.remove_bond(newmol.get_bond(newmol.atoms[i], newmol.atoms[j]))
        else: #remove internal atoms and bonds
            for a in [newmol.atoms[q] for q in path]:
                if a is not tail_atom and a is not head_atom:
                    newmol.remove_atom(a)
        
        if len(newmol.split()) > n_strucs_max: #removing that path creates too many separate structures
            continue
        
        for endatom in [head_atom,tail_atom]:
            octet = get_octet_deviation(endatom)
            assert octet > 0
            while octet > 0:
                H = Atom('H', radical_electrons=0, lone_pairs=0, charge=0)
                bd = Bond(endatom,H)
                newmol.add_atom(H)
                newmol.add_bond(bd)
                octet -= 2
        
        newmol.update(sort_atoms=False)
        
        if estimate_delta:
            assert assoc_decomposition_init_value_unc is not None, "Must provide assoc_decomposition_init_value_unc to estimate delta values for internal new-bond extensions"
            assert tree is not None, "Must provide tree to estimate delta values for internal new-bond extensions"
            delta_v = None
            delta_var = None
            for decomp,v_init,unc_init,tr in assoc_decomposition_init_value_unc:
                missing = False
                for k, a in enumerate(decomp.atoms):
                    if mol.atoms[k] in mapping.keys():                
                        mapping[mol.atoms[k]].label = a.label
                    elif a.label not in ["","*S"]: #we cannot map an important label for this decomposition
                        missing = True
                        break
                
                if missing:
                    v, unc = 0.0,0.0
                else:
                    v, unc = evaluate_single(tree, newmol, estimate_uncertainty=True)

                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_v += v - v_init
                    delta_var += unc ** 2 - unc_init ** 2
                
            newmol.clear_labeled_atoms()

            mols.append((
                newmol,
                None,
                basename
                + "_Int-"
                + str(i + 1)
                + atom_type_i_str
                + "-"
                + str(j + 1)
                + atom_type_j_str
                + "-Br"+str(len(path)),
                "genRemoveBridgeExt",
                (i, j),
                delta_v,
                delta_var,
            ))
        else:
            mols.append((
                newmol,
                None,
                basename
                + "_Int-"
                + str(i + 1)
                + atom_type_i_str
                + "-"
                + str(j + 1)
                + atom_type_j_str
                + "-Br"+str(len(path)),
                "genRemoveBridgeExt",
                (i, j),
            ))
    
    return mols

def specify_external_new_bond_extensions(grp, i, basename, r_bonds, r_label, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for the creation of a bond (of undefined order) between
    an atom and a new atom that is not H
    """
    # cython.declare(ga=GroupAtom, newgrp=Group, j=int)
    grps = []
    for alabel in r_label:
        label_list = []
        ga = GroupAtom([ATOMTYPES["Rx!H"]])
        ga.label = alabel
        newgrp = deepcopy(grp)
        newgrp.add_atom(ga)
        j = newgrp.atoms.index(ga)
        newgrp.add_bond(GroupBond(newgrp.atoms[i], newgrp.atoms[j], r_bonds))
        atom_type = newgrp.atoms[i].atomtype
        if len(atom_type) > 1:
            atom_type_str = ""
            for k in atom_type:
                label_list.append(k.label)
            for p in sorted(label_list):
                atom_type_str += p
        elif len(atom_type) == 0:
            atom_type_str = ""
        else:
            atom_type_str = atom_type[0].label

        if estimate_delta:
            assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for external new-bond extensions"
            assert tree is not None, "Must provide tree to estimate delta values for external new-bond extensions"
            delta_v = None
            delta_var = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    newgrp.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, newgrp, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_v += v - v_init
                    delta_var += unc ** 2 - unc_init ** 2
            
            newgrp.clear_labeled_atoms()
            
            grps.append(
                (
                    newgrp,
                    None,
                    basename + "_Ext-" + str(i + 1) + atom_type_str + "-R" + alabel,
                    "extNewBondExt",
                    (len(newgrp.atoms) - 1,),
                    delta_v,
                    delta_var,
                )
            )
        else:
            grps.append(
                (
                    newgrp,
                    None,
                    basename + "_Ext-" + str(i + 1) + atom_type_str + "-R" + alabel,
                    "extNewBondExt",
                    (len(newgrp.atoms) - 1,),
                )
            )
    return grps

def molecular_specify_external_new_bond_extensions(mol, i, basename, r, r_bonds, r_label, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc=None):
    """
    generates extensions for the creation of a bond (of undefined order) between
    an atom and a new atom that is not H
    """
    # cython.declare(ga=GroupAtom, newgrp=Group, j=int)
    mols = []
    for at in r:
        for alabel in r_label:
            single_bonded_atomstrs = [asing.element.symbol for (asing,bd) in mol.atoms[i].bonds.items() if len(asing.bonds) == 1]
            unique_atomstr_indices = np.unique(single_bonded_atomstrs, return_index=True)[1]
            single_bonded_atomind_bds = [(mol.atoms.index(asing),bd) for (asing,bd) in mol.atoms[i].bonds.items() if len(asing.bonds) == 1]
            single_bonded_atomind_bds = [single_bonded_atomind_bds[q] for q in unique_atomstr_indices]

            for (asingind,bd) in single_bonded_atomind_bds:
                newmol = mol.copy(deep=True)
                asing = newmol.atoms[asingind]
                atom = newmol.atoms[i]
                newmol.remove_atom(asing)
                a = Atom(at.label, radical_electrons=0, lone_pairs=PeriodicSystem.lone_pairs[at.label], charge=0)
                a.label = alabel
                newmol.add_atom(a)
                newmol.add_bond(Bond(atom, a, order=bd.order))
                atom_type = atom.atomtype
                octet = get_octet_deviation(a)
                assert octet >= 0, (octet,newmol.to_adjacency_list(),at,asingind)
                while octet > 0:
                    H = Atom('H', radical_electrons=0, lone_pairs=0, charge=0)
                    bd = Bond(a,H)
                    newmol.add_atom(H)
                    newmol.add_bond(bd)
                    octet -= 2

                atom_type_str = atom_type.label

                newmol.update(sort_atoms=False)
                
                if estimate_delta:
                    assert assoc_decomposition_init_value_unc is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for external new-bond extensions"
                    assert tree is not None, "Must provide tree to estimate delta values for external new-bond extensions"
                    delta_v = None
                    delta_var = None
                    for decomp,v_init,unc_init,tr in assoc_decomposition_init_value_unc:
                        for k, a in enumerate(decomp.atoms):
                            newmol.atoms[k].label = a.label
                        v, unc = evaluate_single(tree, newmol, estimate_uncertainty=True)
                        if delta_v is None:
                            delta_v = v - v_init
                            delta_var = unc ** 2 - unc_init ** 2
                        else:
                            delta_v += v - v_init
                            delta_var += unc ** 2 - unc_init ** 2
                    
                    newmol.clear_labeled_atoms()
                    
                    mols.append(
                        (
                            newmol,
                            None,
                            basename + "_Ext-" + str(i + 1) + atom_type_str + "-R" + alabel,
                            "extNewBondExt",
                            (len(newmol.atoms) - 1,),
                            delta_v,
                            delta_var,
                        )
                    )
                else:
                    mols.append(
                        (
                            newmol,
                            None,
                            basename + "_Ext-" + str(i + 1) + atom_type_str + "-R" + alabel,
                            "extNewBondExt",
                            (len(newmol.atoms) - 1,),
                        )
                    )
    return mols

def generalize_remove_atom_extensions(grp, i, basename, n_struc_max, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for the removal of an atom 
    """
    # cython.declare(ga=GroupAtom, newgrp=Group, j=int)
    if len(grp.atoms) < 2:
        return []
    label_list = []
    grps = []
    newgrp = deepcopy(grp)
    mapping = {a:newgrp.atoms[q] for q,a in enumerate(grp.atoms)}
    
    atom_type = newgrp.atoms[i].atomtype
    if len(atom_type) > 1:
        atom_type_str = ""
        for k in atom_type:
            label_list.append(k.label)
        for p in sorted(label_list):
            atom_type_str += p
    elif len(atom_type) == 0:
        atom_type_str = ""
    else:
        atom_type_str = atom_type[0].label
        
    newgrp.remove_atom(newgrp.atoms[i])
    
    if len(newgrp.split()) > n_struc_max: #removing that atom creates too many separate structures
        return []
        

    if estimate_delta:
        assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for external new-bond extensions"
        assert tree is not None, "Must provide tree to estimate delta values for external new-bond extensions"
        delta_v = None
        delta_var = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            missing = False
            for k, a in enumerate(decomp.atoms):
                if grp.atoms[k] in mapping.keys():                
                    mapping[grp.atoms[k]].label = a.label
                elif a.label not in ["","*S"]: #we cannot map an important label for this decomposition
                    missing = True
                    break
              
            v_init, unc_init = d
            if missing:
                v, unc = 0.0,0.0
            else:
                v, unc = evaluate_single(tree, newgrp, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                delta_var = unc ** 2 - unc_init ** 2
            else:
                delta_v += v - v_init
                delta_var += unc ** 2 - unc_init ** 2
        
        newgrp.clear_labeled_atoms()
        
        grps.append(
            (
                newgrp,
                None,
                basename + "_Ext-" + str(i + 1) + atom_type_str + "-R",
                "genAtomRemovalExt",
                (len(newgrp.atoms) - 1,),
                delta_v,
                delta_var,
            )
        )
    else:
        grps.append(
            (
                newgrp,
                None,
                basename + "_Ext-" + str(i + 1) + atom_type_str + "-R",
                "genAtomRemovalExt",
                (len(newgrp.atoms) - 1,),
            )
        )
    return grps

def molecular_generalize_remove_atom_extensions(mol, i, basename, n_struc_max, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc=None):
    """
    generates extensions for the removal of an atom 
    """
    # cython.declare(ga=GroupAtom, newgrp=Group, j=int)
    if len(mol.atoms) < 2:
        return []
    
    mols = []
    newmol = mol.copy(deep=True)
    mapping = {a:newmol.atoms[q] for q,a in enumerate(mol.atoms)}
    
    atom_type = newmol.atoms[i].atomtype
    
    atom_type_str = atom_type.label
    
    adjacent_atoms = newmol.atoms[i].bonds.keys()
    newmol.remove_atom(newmol.atoms[i])
    
    nsplit = newmol.split()
    if len([x for x in nsplit if len(x.atoms) > 1]) > n_struc_max: #removing that atom creates too many separate structures
        return []
    newmol = [x for x in nsplit if len(x.atoms) > 1]
    if len(newmol) == 0:
        return []
    else:
        newmol = newmol[0]
    for a in adjacent_atoms:
        if a in newmol.atoms:
            octet = get_octet_deviation(a)
            while octet > 0:
                H = Atom('H', radical_electrons=0, lone_pairs=0, charge=0)
                bd = Bond(a,H)
                newmol.add_atom(H)
                newmol.add_bond(bd)
                octet -= 2
           
    newmol.update(sort_atoms=False)
    
    if estimate_delta:
        assert assoc_decomposition_init_value_unc is not None, "Must provide assoc_decomposition_init_value_unc to estimate delta values for external new-bond extensions"
        assert tree is not None, "Must provide tree to estimate delta values for external new-bond extensions"
        delta_v = None
        delta_var = None
        for decomp,v_init,unc_init,tr in assoc_decomposition_init_value_unc:
            missing = False
            for k, a in enumerate(decomp.atoms):
                if mol.atoms[k] in mapping.keys():                
                    mapping[mol.atoms[k]].label = a.label
                elif a.label not in ["","*S"]: #we cannot map an important label for this decomposition
                    missing = True
                    break
              
            if missing:
                v, unc = 0.0,0.0
            else:
                v, unc = evaluate_single(tree, newmol, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                delta_var = unc ** 2 - unc_init ** 2
            else:
                delta_v += v - v_init
                delta_var += unc ** 2 - unc_init ** 2
        
        newmol.clear_labeled_atoms()
        
        mols.append(
            (
                newmol,
                None,
                basename + "_Ext-" + str(i + 1) + atom_type_str + "-R",
                "genAtomRemovalExt",
                (len(newmol.atoms) - 1,),
                delta_v,
                delta_var,
            )
        )
    else:
        mols.append(
            (
                newmol,
                None,
                basename + "_Ext-" + str(i + 1) + atom_type_str + "-R",
                "genAtomRemovalExt",
                (len(newmol.atoms) - 1,),
            )
        )
    return mols

def specify_bond_extensions(grp, i, j, basename, r_bonds, r_bonds_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for the specification of bond order for a given bond
    """
    # cython.declare(grps=list, label_list=list, Rbset=set, bd=float, grp=Group, grpc=Group)
    grps = []
    label_list = []
    Rbset = set(r_bonds)
    bdict = {1: "-", 2: "=", 3: "#", 1.5: "-=", 4: "$", 0.05: "..", 0: "--"}
    bstrdict = {"S": 1, "D": 2, "T": 3, "B": 1.5, "Q": 4, "R": 0.05, "vdW": 0}
    if isinstance(r_bonds_full[0],list):
        r_spc_bonds_full = [[y for y in x if y in r_bonds] for x in r_bonds_full]
        if len(r_spc_bonds_full) == 1:
            r_spc_bonds_full = [[x] for x in r_spc_bonds_full[0]]
        else:
            r_spc_bonds_full += [[x] for x in sum(r_spc_bonds_full,[]) if [x] not in r_spc_bonds_full]
    else:
        r_spc_bonds_full = [[x] for x in r_bonds_full if x in r_bonds]
    for bd in r_spc_bonds_full:
        g = deepcopy(grp)
        grpc = deepcopy(grp)
        g.atoms[i].bonds[g.atoms[j]].order = bd
        g.atoms[j].bonds[g.atoms[i]].order = bd
        grpc.atoms[i].bonds[grpc.atoms[j]].order = list(Rbset - set(bd))
        grpc.atoms[j].bonds[grpc.atoms[i]].order = list(Rbset - set(bd))

        if len(list(Rbset - set(bd))) == 0:
            grpc = None

        atom_type_i = g.atoms[i].atomtype
        atom_type_j = g.atoms[j].atomtype

        if len(atom_type_i) > 1:
            atom_type_i_str = ""
            for k in atom_type_i:
                label_list.append(k.label)
            for p in sorted(label_list):
                atom_type_i_str += p
        elif len(atom_type_i) == 0:
            atom_type_i_str = ""
        else:
            atom_type_i_str = atom_type_i[0].label
        if len(atom_type_j) > 1:
            atom_type_j_str = ""
            for k in atom_type_j:
                label_list.append(k.label)
            for p in sorted(label_list):
                atom_type_j_str += p
        elif len(atom_type_j) == 0:
            atom_type_j_str = ""
        else:
            atom_type_j_str = atom_type_j[0].label

        b = ""
        for v in bdict.keys():
            if any(abs(v - x) < 1e-4 for x in bd):
                b += bdict[v]
        if estimate_delta:
            assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for bond extensions"
            assert tree is not None, "Must provide tree to estimate delta values for bond extensions"
            delta_v = None
            delta_var = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    g.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_v += v - v_init
                    delta_var += unc ** 2 - unc_init ** 2
                    
            g.clear_labeled_atoms()
            
            grps.append(
                (
                    g,
                    grpc,
                    basename
                    + "_Sp-"
                    + str(i + 1)
                    + atom_type_i_str
                    + b
                    + str(j + 1)
                    + atom_type_j_str,
                    "bondExt",
                    (i, j),
                    delta_v,
                    delta_var,
                )
            )
        else:
            grps.append(
                (
                    g,
                    grpc,
                    basename
                    + "_Sp-"
                    + str(i + 1)
                    + atom_type_i_str
                    + b
                    + str(j + 1)
                    + atom_type_j_str,
                    "bondExt",
                    (i, j),
                )
            )
    return grps

def molecular_transform_bond_extensions(mol, i, j, basename, r_bonds, r_bonds_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc=None):
    """
    generates extensions changing the order of a bond
    """
    # cython.declare(grps=list, label_list=list, Rbset=set, bd=float, grp=Group, grpc=Group)
    mols = []
    label_list = []
    Rbset = set(r_bonds)
    bdict = {1: "-", 2: "=", 3: "#", 1.5: "-=", 4: "$", 0.05: "..", 0: "--"}
    bstrdict = {"S": 1, "D": 2, "T": 3, "B": 1.5, "Q": 4, "R": 0.05, "vdW": 0}
    if isinstance(r_bonds_full[0],list):
        r_spc_bonds_full = [[y for y in x if y in r_bonds] for x in r_bonds_full]
        if len(r_spc_bonds_full) == 1:
            r_spc_bonds_full = [x for x in r_spc_bonds_full[0]]
        else:
            r_spc_bonds_full += [x for x in sum(r_spc_bonds_full,[]) if [x] not in r_spc_bonds_full]
    else:
        r_spc_bonds_full = [x for x in r_bonds_full if x in r_bonds]
        
    for order in r_spc_bonds_full:
        lone_bonded_atom_inds_i = [mol.atoms.index(a) for a in mol.atoms[i].bonds.keys() if len(a.bonds) == 1 and mol.atoms.index(a) != j]
        lone_bonded_atom_inds_j = [mol.atoms.index(a) for a in mol.atoms[j].bonds.keys() if len(a.bonds) == 1 and mol.atoms.index(a) != i]
        bd = mol.get_bond(mol.atoms[i],mol.atoms[j])

        if order > (bd.order + min(len(lone_bonded_atom_inds_i),len(lone_bonded_atom_inds_j))): #we can't create this bond without deleting more than local lone bonded atoms
            continue
        
        newmol = mol.copy(deep=True)
        
        atom_i = newmol.atoms[i]
        atom_j = newmol.atoms[j]
        if assoc_decomposition_init_value_unc:
            decomp_to_decomp_newmol_atom_map = {decomp: {decomp.atoms[q]:newmol.atoms[q] for q in range(len(newmol.atoms))} for decomp,v_init,unc_init,tr in assoc_decomposition_init_value_unc}
        atom_i.bonds[atom_j].order = order
        lone_bonded_atoms_i = [newmol.atoms[q] for q in lone_bonded_atom_inds_i]
        lone_bonded_atoms_j = [newmol.atoms[q] for q in lone_bonded_atom_inds_j]
        
        for k,bdatoms in enumerate([atom_i,atom_j]):
            octet = get_octet_deviation(bdatoms)
            if octet > 0:
                while octet > 1:
                    H = Atom('H', radical_electrons=0, lone_pairs=0, charge=0)
                    Hbd = Bond(bdatoms,H)
                    newmol.add_atom(H)
                    newmol.add_bond(Hbd)
                    octet -= 2
            else:
                lone_bonded_ind = 0
                while octet < 0:
                    if k == 0:
                        at_remove = lone_bonded_atoms_i[lone_bonded_ind]
                    else:
                        at_remove = lone_bonded_atoms_j[lone_bonded_ind]
                    newmol.remove_atom(at_remove)
                    octet += 2
                    lone_bonded_ind += 1
        
        atom_type_i = atom_i.atomtype
        atom_type_j = atom_j.atomtype
        atom_type_i_str = atom_type_i.label
        atom_type_j_str = atom_type_j.label

        b = ""
        for v in bdict.keys():
            if any(abs(v - x) < 1e-4 for x in [order]):
                b += bdict[v]
        
        newmol.update(sort_atoms=False)
        
        if estimate_delta:
            assert assoc_decomposition_init_value_unc is not None, "Must provide assoc_decomposition_init_value_unc to estimate delta values for bond extensions"
            assert tree is not None, "Must provide tree to estimate delta values for bond extensions"
            delta_v = None
            delta_var = None
            for decomp,v_init,unc_init,tr in assoc_decomposition_init_value_unc:
                for da,na in  decomp_to_decomp_newmol_atom_map[decomp].items():
                    na.label = da.label
                v, unc = evaluate_single(tree, newmol, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_v += v - v_init
                    delta_var += unc ** 2 - unc_init ** 2
                    
            newmol.clear_labeled_atoms()
            
            mols.append(
                (
                    newmol,
                    None,
                    basename
                    + "_Sp-"
                    + str(i + 1)
                    + atom_type_i_str
                    + b
                    + str(j + 1)
                    + atom_type_j_str,
                    "bondExt",
                    (i, j),
                    delta_v,
                    delta_var,
                )
            )
        else:
            mols.append(
                (
                    newmol,
                    None,
                    basename
                    + "_Sp-"
                    + str(i + 1)
                    + atom_type_i_str
                    + b
                    + str(j + 1)
                    + atom_type_j_str,
                    "bondExt",
                    (i, j),
                )
            )
    return mols

def generalize_bond_extensions(grp, i, j, basename, r_bonds, r_bonds_full, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates generalizations for bond order specification for a given bond.
    """
    grps = []
    Rbset = set(r_bonds)
    if isinstance(r_bonds_full[0], list):
        for L in r_bonds_full:
            if all(a in L for a in grp.get_bond(grp.atoms[i], grp.atoms[j]).order):
                if len(L) > len(grp.get_bond(grp.atoms[i], grp.atoms[j]).order):
                    r_gen = L
                else:
                    r_gen = r_bonds
                break
        else:
            r_gen = r_bonds
    else:
        r_gen = r_bonds

    g = deepcopy(grp)
    grpc = None
    g.atoms[i].bonds[g.atoms[j]].order = r_gen
    g.atoms[j].bonds[g.atoms[i]].order = r_gen

    atom_type_i = g.atoms[i].atomtype
    atom_type_j = g.atoms[j].atomtype
    if len(atom_type_i) > 1:
        atom_type_i_str = ""
        label_list_i = [k.label for k in atom_type_i]
        for p in sorted(label_list_i):
            atom_type_i_str += p
    elif len(atom_type_i) == 0:
        atom_type_i_str = ""
    else:
        atom_type_i_str = atom_type_i[0].label

    if len(atom_type_j) > 1:
        atom_type_j_str = ""
        label_list_j = [k.label for k in atom_type_j]
        for p in sorted(label_list_j):
            atom_type_j_str += p
    elif len(atom_type_j) == 0:
        atom_type_j_str = ""
    else:
        atom_type_j_str = atom_type_j[0].label

    b = ""
    for v in {1: "-", 2: "=", 3: "#", 1.5: "-=", 4: "$", 0.05: "..", 0: "--"}.keys():
        if any(abs(v - x) < 1e-4 for x in r_gen):
            b += {1: "-", 2: "=", 3: "#", 1.5: "-=", 4: "$", 0.05: "..", 0: "--"}[v]

    if estimate_delta:
        assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for bond generalizations"
        assert tree is not None, "Must provide tree to estimate delta values for bond generalizations"
        delta_v = None
        delta_var = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_var = unc ** 2 - unc_init ** 2
                else:
                    delta_var = unc ** 2 - unc_init ** 2
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_var += unc ** 2 - unc_init ** 2
                else:
                    delta_var += unc ** 2 - unc_init ** 2
        
        g.clear_labeled_atoms()
        
        grps.append(
            (
                g,
                grpc,
                basename + "_Sp-" + str(i + 1) + atom_type_i_str + b + str(j + 1) + atom_type_j_str,
                "bondGen",
                (i, j),
                delta_v,
                delta_var,
            )
        )
    else:
        grps.append(
            (
                g,
                grpc,
                basename + "_Sp-" + str(i + 1) + atom_type_i_str + b + str(j + 1) + atom_type_j_str,
                "bondGen",
                (i, j),
            )
        )

    return grps


def generate_extensions_reverse(grp,structs):
    """
    This function is designed to generate extensions by reverse engineering the structures being split rather than extending the original group
    This should be a reliable fallback when traditional extension generation becomes to expensive
    """
    exts = []
    if isinstance(structs[0], Molecule):
        temp_structs = structs
    else: #Datum
        temp_structs = [x.mol for x in structs]

    for st in temp_structs:
        new_struct = st.to_group()
        #fix ring membership
        rc = new_struct.get_relevant_cycles()
        for atom in new_struct.atoms:
            atom.props['inRing'] = False
            for ring in rc:
                if atom in ring:
                    atom.props['inRing'] = True
                    break
        new,comp = split_mols(temp_structs, new_struct)
        if len(new) > 0 and len(comp) > 0:
            aexts = [new_struct]
            matches = [frozenset([st])]
        else: #this almost means all structures are duplicates...they likely are
            aexts = []
            matches = []
        st_inds_to_not_remove = []
        st_inds = list(range(len(st.atoms)))
        while True:
            old_struct = new_struct
            new_struct = old_struct.copy(deep=True)
            scores = [score_atom_reverse_extension_generation(new_struct,a) for a in new_struct.atoms]
            inds = np.argsort(scores)[::-1]
            ind = None
            index = 0
            assert len(st_inds) == len(new_struct.atoms), (len(st_inds),len(new_struct.atoms))
            while ind is None or st_inds[ind] in st_inds_to_not_remove or scores[ind] == -np.inf:
                ind = inds[index]
                index += 1
                if index == len(inds): #tried every atom
                    ind = None 
                    break
                
            if ind is None:
                break
            
            at = new_struct.atoms[ind]
            if at.label: #don't remove any labeled atoms
                st_inds_to_not_remove.append(st_inds[ind])
                continue
            new_struct.remove_atom(at)
            new_struct.update()
            
            if not new_struct.is_subgraph_isomorphic(grp, save_order=True, check_labels=True): #removing that atom broke isomorphism with original group so don't delete that atom
                new_struct = old_struct
                st_inds_to_not_remove.append(st_inds[ind])
                continue
            else:
                new,comp = split_mols(temp_structs, new_struct)
                boos = np.array([item.is_subgraph_isomorphic(new_struct, save_order=True, check_labels=True) for item in temp_structs])
                if len(comp) == 0: #suddenly matches all groups...don't remove that atom
                    new_struct = old_struct
                    st_inds_to_not_remove.append(st_inds[ind])
                    continue
                elif len(comp) > 0 and len(new) > 0: #splits groups
                    del st_inds[ind]
                    aexts.append(new_struct)
                    matches.append(frozenset(s for i,s in enumerate(temp_structs) if boos[i]))
                else:
                    del st_inds[ind]
                    continue

        best_split_to_ext = dict()
        for i,g in enumerate(aexts):
            sts = matches[i]
            if sts in best_split_to_ext.keys() and len(best_split_to_ext[sts].atoms) < len(g.atoms):
                continue
            else:
                best_split_to_ext[sts] = g

        exts.extend(list(best_split_to_ext.values()))

    for ext in exts:
        ext.multiplicity = grp.multiplicity
    
    return exts

def score_atom_reverse_extension_generation(g,atm):
    s = 0
    if len(atm.atomtype) == 1 and atm.atomtype[0].label == 'H':
        s += 1
    s -= len(atm.bonds)
    iatm = g.atoms.index(atm)
    gtemp = g.copy(deep=True)
    gtemp.remove_atom(gtemp.atoms[iatm])
    if len(gtemp.split()) > len(g.split()):
        s -= np.inf
    return s

def extend_structure_from_group_to_specific_group(struct,grp,grpspec,element_atomtypes,struct_to_node_isomorphisms=None):
    gen_structs = []
    
    if struct_to_node_isomorphisms is None:
        struct_to_node_isomorphisms = struct.find_subgraph_isomorphisms(grp,save_order=True,check_labels=True)
    struct_to_grp_index_isomorphisms = [{struct.atoms.index(a):grp.atoms.index(b) for a,b in iso.items()} for iso in struct_to_node_isomorphisms]
    node_to_struct_index_isomorphisms = [{grp.atoms.index(v):struct.atoms.index(k) for k,v in iso.items()} for iso in struct_to_node_isomorphisms]
    
    for i,struct_node_iso in enumerate(struct_to_grp_index_isomorphisms):
        node_struct_iso = node_to_struct_index_isomorphisms[i]
        child_to_node_isomorphisms = grpspec.find_intersection_isomorphisms(grp,save_order=True,check_labels=True)
        node_to_child_isomorphisms = [{v:k for k,v in d.items()} for d in child_to_node_isomorphisms]
        node_to_child_index_isomorphisms = [{grp.atoms.index(node_at):grpspec.atoms.index(child_at) for node_at,child_at in iso.items()} for iso in node_to_child_isomorphisms]
        for node_child_iso in node_to_child_index_isomorphisms:
            child_node_iso = {v:k for k,v in node_child_iso.items()}
            new_struct = struct.copy(deep=True)
            atoms_to_remove = []
            for struct_index,node_index in struct_node_iso.items(): #find node mapped atoms intersection, on these mappings we try to make every atom as specific as the child/specific group
                if struct.atoms[struct_index].has_intersection_with(grpspec.atoms[node_child_iso[node_index]]):
                    set_intersection_with_atom(new_struct.atoms[struct_index],grpspec.atoms[node_child_iso[node_index]],element_atomtypes=element_atomtypes)
                else: #if an atom cannot be made as specific as the child group we have to give up
                    break #cannot make viable new_struct
            else:
                continuing = False
                for bd_child in grpspec.get_all_edges(): #find node mapped bonds intersection, on these mappings we try to make every bond as specific as the child/specific group
                    ind1 = grpspec.atoms.index(bd_child.vertex1)
                    ind2 = grpspec.atoms.index(bd_child.vertex2)
                    if ind1 in child_node_iso.keys() and ind2 in child_node_iso.keys():
                        if new_struct.has_bond(new_struct.atoms[node_struct_iso[child_node_iso[ind1]]],new_struct.atoms[node_struct_iso[child_node_iso[ind2]]]):
                            bd_struct = new_struct.get_bond(new_struct.atoms[node_struct_iso[child_node_iso[ind1]]],new_struct.atoms[node_struct_iso[child_node_iso[ind2]]])
                        else:
                            bd_struct = None

                        if bd_struct and bd_child:
                            if bd_struct.has_intersection_with(bd_child):
                                set_intersection_with_bond(bd_struct,bd_child)
                            else:
                                continuing = True
                                break
                        else: #bd_struct None => bd_node None so this is a new bond...so add that to struct
                            new_bd = GroupBond(new_struct.atoms[node_struct_iso[child_node_iso[ind1]]],new_struct.atoms[node_struct_iso[child_node_iso[ind2]]],order=bd_child.order)
                            new_struct.add_bond(new_bd)
                if continuing:
                    continue
                    
                #now add the missing atoms and associated bonds from the child to new_struct
                child_struct_iso = {node_child_iso[k]:v for k,v in node_struct_iso.items()}
                continuing = False
                while len(child_struct_iso) < len(grpspec.atoms):
                    map_len = len(child_struct_iso)
                    for i,a in enumerate(grpspec.atoms):
                        if i in child_struct_iso.keys(): #if that atom is already mapped to struct skip
                            continue
                        else:
                            newga = grpspec.atoms[i].copy()
                            bonded_inds = [grpspec.atoms.index(at) for at in a.bonds.keys()]
                            mapped_bonded_inds = list(set(bonded_inds).intersection(set(child_struct_iso.keys())))
                            new_struct.add_atom(newga)
                            child_struct_iso[i] = new_struct.atoms.index(newga)
                            for mapped_bonded_ind in mapped_bonded_inds:
                                child_bond = grpspec.get_bond(grpspec.atoms[i],grpspec.atoms[mapped_bonded_ind])
                                st_atom = new_struct.atoms[child_struct_iso[mapped_bonded_ind]]
                                if st_atom.lone_pairs:
                                    lone_pairs = min(st_atom.lone_pairs)
                                else:
                                    lone_pairs = 0
                                if st_atom.radical_electrons:
                                    radical_electrons = min(st_atom.radical_electrons)
                                else:
                                    radical_electrons = 0
                                minimum_octet =  sum(min(bd.order) for bd in st_atom.bonds.values())*2 + lone_pairs + radical_electrons + 2*min(child_bond.order)
                                if minimum_octet > 8: #if adding this bond would violate octet rule try to remove Hydrogen's
                                    lone_bonded_atoms = [a for a in st_atom.bonds.keys() if len(a.bonds) == 1 and new_struct.atoms.index(a) not in child_struct_iso.values() and a not in atoms_to_remove]
                                    lone_bond_ind = 0
                                    octet_change = 0
                                    while minimum_octet + octet_change > 8:
                                        if lone_bond_ind >= len(lone_bonded_atoms): #unable to fix octet violation
                                            continuing = True
                                            break
                                        lone_order = min(st_atom.bonds[lone_bonded_atoms[lone_bond_ind]].order)
                                        if new_struct.atoms.index(lone_bonded_atoms[lone_bond_ind]) not in child_struct_iso.values():
                                            atoms_to_remove.append(lone_bonded_atoms[lone_bond_ind])
                                            lone_bond_ind += 1
                                            octet_change -= lone_order*2
                                if continuing:
                                    break
                                gbd = GroupBond(st_atom,newga,order=child_bond.order)
                                new_struct.add_bond(gbd)

                    if continuing:
                        break
                       
                if not continuing:
                    for a in atoms_to_remove:
                        new_struct.remove_atom(a)

                    if not new_struct.is_subgraph_isomorphic(grp, save_order=True, check_labels=True) or not new_struct.is_subgraph_isomorphic(grpspec, save_order=True, check_labels=True):
                        raise ValueError("Generated structure is not isomorphic to a group.")

                    new_struct.clear_labeled_atoms()
                    gen_structs.append(new_struct)
    
    return gen_structs

def extend_structure_from_group_to_general_group(struct,grp,grpgen,struct_to_node_isomorphisms=None,max_recursion_depth=2,recursion_depth=0):
    if max_recursion_depth <= recursion_depth:
        return [],[]
    
    gen_structs = []
    node_to_struct_index_isomorphism_record = []
    if struct_to_node_isomorphisms is None:
        struct_to_node_isomorphisms = struct.find_subgraph_isomorphisms(grp,save_order=True,check_labels=True)
        
    node_to_struct_index_isomorphisms = [{grp.atoms.index(v):struct.atoms.index(k) for k,v in iso.items()} for iso in struct_to_node_isomorphisms]
    
    node_to_parent_isomorphisms = grp.find_subgraph_isomorphisms(grpgen,save_order=True,check_labels=True)
    node_to_parent_index_isomorphisms = [{grp.atoms.index(a):grpgen.atoms.index(b) for a,b in iso.items()} for iso in node_to_parent_isomorphisms]
    
    for node_to_parent_index_isomorphism in node_to_parent_index_isomorphisms:
        for node_to_struct_index_isomorphism in node_to_struct_index_isomorphisms:
            unmapped_node_indices = []
            for node_index,node_at in enumerate(grp.atoms): #if the node is mapped generalize it to the parent
                if node_index in node_to_parent_index_isomorphism.keys():
                    parent_index = node_to_parent_index_isomorphism[node_index]
                    p_at = grpgen.atoms[parent_index]
                    if not node_at.equivalent(p_at):
                        struct_index = node_to_struct_index_isomorphism[node_index]
                        new_struct = struct.copy(deep=True)
                        st_at = new_struct.atoms[struct_index]
                        newst_at = GroupAtom(atomtype=p_at.atomtype,radical_electrons=p_at.radical_electrons,charge=p_at.charge,label=p_at.label,
                                             lone_pairs=p_at.lone_pairs,site=p_at.site,morphology=p_at.morphology,props=p_at.props.copy())
                        gbds = []
                        for a,bd in st_at.bonds.items():
                            gbds.append(GroupBond(newst_at,a,order=bd.order))
                        new_struct.remove_atom(st_at)
                        new_struct.vertices.insert(struct_index,newst_at)
                        for bd in gbds:
                            new_struct.add_bond(bd)
                        if not new_struct.is_subgraph_isomorphic(grp,save_order=True,check_labels=True): #if the new struct is not isomorphic to the node then this individual change is enough
                            gen_structs.append(new_struct)
                            node_to_struct_index_isomorphism_record.append(node_to_struct_index_isomorphism)
                        else: #otherwise this individual change is not enough due to isomorphic degeneracy...recurse
                            out_grps,node_to_struct_index_isomorphism_record_local = extend_structure_from_group_to_general_group(new_struct,grp,grpgen,max_recursion_depth=max_recursion_depth,recursion_depth=recursion_depth+1)
                            gen_structs.extend(out_grps)
                            node_to_struct_index_isomorphism_record.extend(node_to_struct_index_isomorphism_record_local)
                else: #if the node is unmapped note it for further analysis
                    unmapped_node_indices.append(node_index)
            
            missing_bond_indices = []
            for bd in grp.get_all_edges(): #go through all node group bonds
                missing = False
                node_index1 = grp.atoms.index(bd.vertex1)
                node_index2 = grp.atoms.index(bd.vertex2)
                if node_index1 in node_to_parent_index_isomorphism.keys() and node_index2 in node_to_parent_index_isomorphism.keys() and grpgen.has_bond(grpgen.atoms[node_to_parent_index_isomorphism[node_index1]],grpgen.atoms[node_to_parent_index_isomorphism[node_index2]]): #if both 
                    parent_index1 = node_to_parent_index_isomorphism[node_index1]
                    parent_index2 = node_to_parent_index_isomorphism[node_index2]
                else:
                    missing = True #bond in node is not present in parent
                if not missing and not grpgen.has_bond(grpgen.atoms[parent_index1],grpgen.atoms[parent_index2]):
                    missing = True
                
                if missing:
                    missing_bond_indices.append((node_index1,node_index2))
                else:
                    parent_bd = grpgen.get_bond(grpgen.atoms[parent_index1],grpgen.atoms[parent_index2])
                    node_bd = grp.get_bond(grp.atoms[node_index1],grp.atoms[node_index2])
                    if not node_bd.equivalent(parent_bd): #if the bonds aren't the same adjust bond order to match parent, but not node
                        struct_index1 = node_to_struct_index_isomorphism[node_index1]
                        struct_index2 = node_to_struct_index_isomorphism[node_index2]
                        new_struct = struct.copy(deep=True)
                        bd = new_struct.get_bond(new_struct.atoms[struct_index1],new_struct.atoms[struct_index2])
                        bd.order = parent_bd.order
                        gen_structs.append(new_struct)
                        node_to_struct_index_isomorphism_record.append(node_to_struct_index_isomorphism)
            #remove atoms/bonds, do not change the split of structures...remove all separate sets of connected atoms/bonds
            #cluster atoms/bonds
            unmapped_node_index_clusters = []
            unmapped_node_indices_left = unmapped_node_indices[:]
            
            while len(unmapped_node_indices_left) > 0:
                ind = unmapped_node_indices_left[0]
                a = grp.atoms[ind]
                cluster = [ind]
                new_ats = [a]
                while len(new_ats) > 0:
                    ats = new_ats[:]
                    new_ats = []
                    for at in ats:
                        for at2 in at.bonds.keys():
                            ind2 = grp.atoms.index(at2)
                            if ind2 in unmapped_node_indices and ind2 not in cluster:
                                new_ats.append(at2)
                                cluster.append(ind2)
                
                unmapped_node_index_clusters.append(cluster)
                for cind in cluster:
                    unmapped_node_indices_left.remove(cind)
            
            for node_index_cluster in unmapped_node_index_clusters:
                new_struct = struct.copy(deep=True)
                gen_atoms = [new_struct.atoms[node_to_struct_index_isomorphism[index]] for index in node_index_cluster]
                for a in gen_atoms:
                    new_struct.remove_atom(a)
                if len(new_struct.split()) == 1:
                    gen_structs.append(new_struct)
                node_to_struct_index_isomorphism_record.append(node_to_struct_index_isomorphism)
                
            for missing_bond_inds in missing_bond_indices: #if they involved a removed atom we don't need to worry about them
                if missing_bond_inds[0] in node_to_struct_index_isomorphism.keys() and missing_bond_inds[1] in node_to_struct_index_isomorphism.keys():
                    new_struct = struct.copy(deep=True)
                    bd = new_struct.get_bond(new_struct.atoms[node_to_struct_index_isomorphism[missing_bond_inds[0]]],new_struct.atoms[node_to_struct_index_isomorphism[missing_bond_inds[1]]])
                    new_struct.remove_bond(bd)
                    if len(new_struct.split()) == 1:
                        gen_structs.append(new_struct)
                        node_to_struct_index_isomorphism_record.append(node_to_struct_index_isomorphism)
    
    return gen_structs,node_to_struct_index_isomorphism_record

def generative_extensions_from_tree_node(decomp,node,element_atomtypes):
    """Generates generative extensions by following the tree up or down on individual decompositions
    Currently assumes that the decompositions are labeling only (do not remove, modify, or add atoms apart from adding labels)
    Args:
        decomp: a decomposition of the generative group structure
        node: the node the decomposition matches in the tree
    """
    #break this into two functions for modifying struct that matches a group to another group, one for when the other group is more specific
    # and one for when the other group is more general, this will allow internal recursion of these algorithms when we hit mapping degeneracy
    gen_structs = []
    tree_target_nodes = []
    delta = []
    delta_var = []
    
    struct_to_node_isomorphisms = decomp.find_subgraph_isomorphisms(node.group,save_order=True,check_labels=True)
    
    for child in node.children:
        out_structs = extend_structure_from_group_to_specific_group(decomp,node.group,child.group,element_atomtypes,struct_to_node_isomorphisms=struct_to_node_isomorphisms)
        gen_structs.extend(out_structs)
        tree_target_nodes.extend([child]*len(out_structs))
        delta.extend([child.rule.value-node.rule.value]*len(out_structs))
        delta_var.extend([child.rule.uncertainty - node.rule.uncertainty]*len(out_structs))
    
    if node.parent.group is not None:
        out_structs,_ = extend_structure_from_group_to_general_group(decomp,node.group,node.parent.group,struct_to_node_isomorphisms=struct_to_node_isomorphisms)
        gen_structs.extend(out_structs)
        tree_target_nodes.extend([node.parent]*len(out_structs))
        delta.extend([node.parent.rule.value-node.rule.value]*len(out_structs))
        delta_var.extend([node.parent.rule.uncertainty - node.rule.uncertainty]*len(out_structs))
    
    return gen_structs,tree_target_nodes,delta,delta_var

def molecular_generative_extensions_from_tree_node(decomp,node,element_atomtypes):
    """Generates generative extensions by following the tree up or down on individual decompositions
    Currently assumes that the decompositions are labeling only (do not remove, modify, or add atoms apart from adding labels)
    Args:
        decomp: a decomposition of the generative group structure
        node: the node the decomposition matches in the tree
    """
    #break this into two functions for modifying struct that matches a group to another group, one for when the other group is more specific
    # and one for when the other group is more general, this will allow internal recursion of these algorithms when we hit mapping degeneracy
    if node.group is None:
        return [],[],[],[]
    
    gen_structs = []
    tree_target_nodes = []
    delta = []
    delta_var = []
    
    if not isinstance(decomp,Group):
        struct = decomp.to_group()
    else:
        struct = decomp
    struct_to_node_isomorphisms = struct.find_subgraph_isomorphisms(node.group,save_order=True,check_labels=True)
    
    for child in node.children:
        output_structs = extend_structure_from_group_to_specific_group(struct,node.group,child.group,element_atomtypes,struct_to_node_isomorphisms=struct_to_node_isomorphisms)
        out_structs = []
        for st in output_structs:
            st.clear_labeled_atoms()
            try:
                out_structs.append(st.make_sample_molecule()) #molecularizing the group makes it more specific so this is okay
            except (AtomTypeError,UnexpectedChargeError):
                continue

        gen_structs.extend(out_structs)
        tree_target_nodes.extend([child]*len(out_structs))
        delta.extend([child.rule.value-node.rule.value]*len(out_structs))
        delta_var.extend([child.rule.uncertainty - node.rule.uncertainty]*len(out_structs))
    
    if node.parent.group is not None:
        out_structs,node_to_struct_index_isomorphism_record = extend_structure_from_group_to_general_group(struct,node.group,node.parent.group,struct_to_node_isomorphisms=struct_to_node_isomorphisms)
        out_structs = sum([make_constrained_sample_molecule(st,node.group,node_to_struct_index_isomorphism_record[i],element_atomtypes) for i,st in enumerate(out_structs)],[])
        gen_structs.extend(out_structs)
        tree_target_nodes.extend([node.parent]*len(out_structs))
        delta.extend([node.parent.rule.value-node.rule.value]*len(out_structs))
        delta_var.extend([node.parent.rule.uncertainty - node.rule.uncertainty]*len(out_structs))
        
    return gen_structs,tree_target_nodes,delta,delta_var #delta and delta_var here are probably terrible estimates...

def make_constrained_sample_molecule(struct,grp,node_to_struct_index_isomorphism,element_atomtypes):
    atom_to_differentiating_atomtypes_map = dict()
    for node_index,struct_index in node_to_struct_index_isomorphism.items():
        if struct_index >= len(struct.atoms):
            continue
        target_atom = struct.atoms[struct_index]
        other_atom = grp.atoms[node_index]
        target_element_atomtypes = {el for atyp in target_atom.atomtype for el in get_atomtype_elements(atyp,element_atomtypes)}
        other_element_atomtypes = {el for atyp in other_atom.atomtype for el in get_atomtype_elements(atyp,element_atomtypes)}
        target_diff = list(target_element_atomtypes - other_element_atomtypes)
        if len(target_diff) == len(target_element_atomtypes):
            continue
        else:
            atom_to_differentiating_atomtypes_map[struct_index] = target_diff
    
    bond_to_differentiating_orders_map = dict()
    for bdgen in grp.get_all_edges():
        node_index1 = grp.atoms.index(bdgen.vertex1)
        node_index2 = grp.atoms.index(bdgen.vertex2)
        struct_index1 = node_to_struct_index_isomorphism[node_index1]
        struct_index2 = node_to_struct_index_isomorphism[node_index2]
        if struct_index1 >= len(struct.atoms) or struct_index2 >= len(struct.atoms) or not struct.has_bond(struct.atoms[struct_index1],struct.atoms[struct_index2]):
            continue
        bdst = struct.get_bond(struct.atoms[struct_index1],struct.atoms[struct_index2])
        order_diff = list(set(bdst.order) - set(bdgen.order))
        if len(order_diff) == len(bdst.order):
            continue
        else:
            bond_to_differentiating_orders_map[(struct_index1,struct_index2)] = order_diff
    
    out_structs = []
    for struct_index,target_diff in atom_to_differentiating_atomtypes_map.items():
        for atyp in target_diff:
            new_struct = struct.copy(deep=True)
            new_struct.atoms[struct_index].atomtype = [atyp]
            new_struct.clear_labeled_atoms()
            try:
                out_structs.append(new_struct.make_sample_molecule())
            except (UnexpectedChargeError, AtomTypeError) as e:
                continue
    
    for struct_inds,order_diff in bond_to_differentiating_orders_map.items():
        for order in order_diff:
            new_struct = struct.copy(deep=True)
            bdnew = new_struct.get_bond(new_struct.atoms[struct_inds[0]],new_struct.atoms[struct_inds[1]])
            bdnew.order = [order]
            new_struct.clear_labeled_atoms()
            try:
                out_structs.append(new_struct.make_sample_molecule())
            except (UnexpectedChargeError, AtomTypeError) as e:
                continue
    
    return out_structs
    
def set_intersection_with_atom(target_atom,other_atom,element_atomtypes):
    #atomtype
    target_element_atomtypes = {el for atyp in target_atom.atomtype for el in get_atomtype_elements(atyp,element_atomtypes)}
    other_element_atomtypes = {el for atyp in other_atom.atomtype for el in get_atomtype_elements(atyp,element_atomtypes)}
    target_atom.atomtype = list(target_element_atomtypes.intersection(other_element_atomtypes))
    
    #radical electrons
    if target_atom.radical_electrons and other_atom.radical_electrons:
        target_atom.radical_electrons = list(set(target_atom.radical_electrons).intersection(set(other_atom.radical_electrons)))
    elif other_atom.radical_electrons:
        target_atom.radical_electrons = other_atom.radical_electrons

    # lone pairs
    if target_atom.lone_pairs and other_atom.lone_pairs:
        target_atom.lone_pairs = list(set(target_atom.lone_pairs).intersection(set(other_atom.lone_pairs)))
    elif other_atom.lone_pairs:
        target_atom.lone_pairs = other_atom.lone_pairs

    # charge
    if target_atom.charge and other_atom.charge:
        target_atom.charge = list(set(target_atom.charge).intersection(set(other_atom.charge)))
    elif other_atom.charge:
        target_atom.charge = other_atom.charge

    # site
    if target_atom.site and other_atom.site:
        target_atom.site = list(set(target_atom.site).intersection(set(other_atom.site)))
    elif other_atom.site:
        target_atom.site = other_atom.site

    # morphology
    if target_atom.morphology and other_atom.morphology:
        target_atom.morphology = list(set(target_atom.morphology).intersection(set(other_atom.morphology)))
    elif other_atom.morphology:
        target_atom.morphology = other_atom.morphology

    if not target_atom.label:
        target_atom.label = other_atom.label
    
    if "Ncoord" in target_atom.props.keys() and target_atom.props["Ncoord"] and "Ncoord" in other_atom.props.keys() and other_atom.props["Ncoord"]:
        target_atom.props["Ncoord"] = list(set(target_atom.props["Ncoord"]).intersection(set(other_atom.props["Ncoord"])))
    elif "Ncoord" in other_atom.props.keys() and other_atom.props["Ncoord"] is not None:
        target_atom.props["Ncoord"] = other_atom.props["Ncoord"]
    
    if "inRing" in other_atom.props.keys() and other_atom.props["inRing"] is not None:
        target_atom.props["inRing"] = other_atom.props["inRing"]
    
def set_intersection_with_bond(target_bond,other_bond):
    target_bond.order = list(set(target_bond.order).intersection(set(other_bond.order)))