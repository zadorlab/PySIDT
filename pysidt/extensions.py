import logging
from copy import deepcopy

import numpy as np

try:
    from molecule.molecule.atomtype import ATOMTYPES
    from molecule.molecule.element import bde_elements
    from molecule.molecule.group import GroupAtom, GroupBond
    from molecule.molecule.molecule import Molecule
except:
    from rmgpy.molecule.atomtype import ATOMTYPES
    from rmgpy.molecule.element import bde_elements
    from rmgpy.molecule.group import GroupAtom, GroupBond
    from rmgpy.molecule.molecule import Molecule

from pysidt.utils import find_shortest_paths, evaluate_single


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
                newgrp, generate_initial_map=True, save_order=True
            ):
                new.append(mol)
            else:
                comp.append(mol)
    else:
        for i, datum in enumerate(data):
            if datum.mol.is_subgraph_isomorphic(
                newgrp, generate_initial_map=True, save_order=True
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
    iter_max=np.inf,
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
            not grps[iter]
            and len(grps) != iter + 1
            and (not (any([len(x) > 0 for x in out_exts])))
        ):
            iter += 1
            if len(grps[iter]) > iter_item_cap:
                logging.error(
                    "Recursion item cap hit not splitting {0} data at iter {1} with {2} items".format(
                        len(items), iter, len(grps[iter])
                    )
                )
                iter -= 1
                gave_up_split = True

        elif (
            not grps[iter]
            and len(grps) != iter + 1
            and (any([len(x) > 0 for x in out_exts]) and iter + 1 > iter_max)
        ):
            logging.error("iter_max achieved terminating early")

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
            delta_unc = None
            for decomp,d in assoc_decomposition_init_value_unc_dict.items():
                for i,a in enumerate(decomp.atoms):
                    g.atoms[i].label = a.label
                v_init,unc_init = d
                v,unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_unc = np.sqrt(unc**2 - unc_init**2)
                else:
                    delta_v += v - v_init
                    delta_unc += np.sqrt(unc**2 - unc_init**2)
            
            g.clear_labeled_atoms()
            
            grps.append(
            (
                g,
                grpc,
                basename + "_" + str(i + 1) + old_atom_type_str + "->" + "".join([x.label for x in item]),
                "atomExt",
                (i,),
                delta_v,
                delta_unc,
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
        delta_unc = None
        for decomp,d in assoc_decomposition_init_value_unc_dict.items():
            for i,a in enumerate(decomp.atoms):
                g.atoms[i].label = a.label
            v_init,unc_init = d
            v,unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_unc = -np.sqrt(unc**2 - unc_init**2)
                else:
                    delta_unc = np.sqrt(unc**2 - unc_init**2)
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_unc += -np.sqrt(unc**2 - unc_init**2)
                else:
                    delta_unc += np.sqrt(unc**2 - unc_init**2)

        grps.append(
            (
                g,
                None,
                basename + "_" + str(i + 1) + old_atom_type_str + "->" + "".join([x.label for x in r_gen]),
                "atomGen",
                (i,),
                delta_v,
                delta_unc,
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
        delta_unc = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_unc = -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_unc += -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
        grps.append(
            (
                g,
                grpc,
                basename + "_" + str(i + 1) + old_atom_type_str + "->anyRing",
                "ringGen",
                (i,),
                delta_v,
                delta_unc,
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
        delta_unc = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_unc = -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_unc += -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-u" + "".join([str(x) for x in r_gen]), "elGen", (i,), delta_v, delta_unc))
    else:
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-u" + "".join([str(x) for x in r_gen]), "elGen", (i,)))

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
        delta_unc = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_unc = -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_unc += -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-p" + "".join([str(x) for x in r_gen]), "lonepairGen", (i,), delta_v, delta_unc))
    else:
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-p" + "".join([str(x) for x in r_gen]), "lonepairGen", (i,)))

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
        delta_unc = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_unc = -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_unc += -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-s" + "".join([str(x) for x in r_gen]), "siteGen", (i,), delta_v, delta_unc))
    else:
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-s" + "".join([str(x) for x in r_gen]), "siteGen", (i,)))

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
        delta_unc = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_unc = -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_unc += -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-m" + "".join([str(x) for x in r_gen]), "morphGen", (i,), delta_v, delta_unc))
    else:
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-m" + "".join([str(x) for x in r_gen]), "morphGen", (i,)))

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
        delta_unc = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_unc = -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_unc += -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-n" + "".join([str(x) for x in r_gen]), "coordGen", (i,), delta_v, delta_unc))
    else:
        grps.append((g, grpc, basename + "_" + str(i + 1) + "-n" + "".join([str(x) for x in r_gen]), "coordGen", (i,)))

    return grps


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
        delta_unc = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                if unc < unc_init:
                    delta_unc = -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            else:
                delta_v += v - v_init
                if unc < unc_init:
                    delta_unc += -np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
        grps.append(
            (
                g,
                grpc,
                basename + "_Sp-" + str(i + 1) + atom_type_i_str + b + str(j + 1) + atom_type_j_str,
                "bondGen",
                (i, j),
                delta_v,
                delta_unc,
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
        delta_unc = None
        for decomp, d in assoc_decomposition_init_value_unc_dict.items():
            for k, a in enumerate(decomp.atoms):
                g.atoms[k].label = a.label
            v_init, unc_init = d
            v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
            if delta_v is None:
                delta_v = v - v_init
                delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            else:
                delta_v += v - v_init
                delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
        grps.append(
            (
                g,
                grpc,
                basename + "_" + str(i + 1) + atom_type_str + "-inRing",
                "ringExt",
                (i,),
                delta_v,
                delta_unc,
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
            delta_unc = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    g.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_v += v - v_init
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            grps.append((g, grpc, basename + "_" + str(i + 1) + "-u" + "".join([str(x) for x in item]), "elExt", (i,), delta_v, delta_unc))
        else:
            grps.append(
                (g, grpc, basename + "_" + str(i + 1) + "-u" + "".join([str(x) for x in item]), "elExt", (i,))
            )

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
            delta_unc = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    g.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_v += v - v_init
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            grps.append((g, grpc, basename + "_" + str(i + 1) + "-p" + "".join([str(x) for x in item]), "lonepairExt", (i,), delta_v, delta_unc))
        else:
            grps.append(
                (g, grpc, basename + "_" + str(i + 1) + "-p" + "".join([str(x) for x in item]), "lonepairExt", (i,))
            )

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
            delta_unc = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    g.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_v += v - v_init
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            grps.append((g, grpc, basename + "_" + str(i + 1) + "-s" + "".join([str(x) for x in item]), "siteExt", (i,), delta_v, delta_unc))
        else:
            grps.append(
                (g, grpc, basename + "_" + str(i + 1) + "-s" + "".join([str(x) for x in item]), "siteExt", (i,))
            )

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
            delta_unc = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    g.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_v += v - v_init
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            grps.append((g, grpc, basename + "_" + str(i + 1) + "-m" + "".join([str(x) for x in item]), "morphExt", (i,), delta_v, delta_unc))
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
            delta_unc = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    g.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_v += v - v_init
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            grps.append((g, grpc, basename + "_" + str(i + 1) + "-n" + "".join([str(x) for x in item]), "coordExt", (i,), delta_v, delta_unc))
        else:
            grps.append(
                (g, grpc, basename + "_" + str(i + 1) + "-n" + "".join([str(x) for x in item]), "coordExt", (i,))
            )

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
            delta_unc = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    newgrp.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, newgrp, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_v += v - v_init
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))

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
                    delta_unc,
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
                delta_unc = None
                for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                    for k, a in enumerate(decomp.atoms):
                        newgrp.atoms[k].label = a.label
                    v_init, unc_init = d
                    v, unc = evaluate_single(tree, newgrp, estimate_uncertainty=True)
                    if delta_v is None:
                        delta_v = v - v_init
                        delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                    else:
                        delta_v += v - v_init
                        delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))

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
                    delta_unc,
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

def specify_external_new_bond_extensions(grp, i, basename, r_bonds, r_label, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for the creation of a bond (of undefined order) between
    an atom and a new atom that is not H
    """
    # cython.declare(ga=GroupAtom, newgrp=Group, j=int)
    label_list = []
    grps = []
    for alabel in r_label:
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
            delta_unc = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    newgrp.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, newgrp, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_v += v - v_init
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            grps.append(
                (
                    newgrp,
                    None,
                    basename + "_Ext-" + str(i + 1) + atom_type_str + "-R" + alabel,
                    "extNewBondExt",
                    (len(newgrp.atoms) - 1,),
                    delta_v,
                    delta_unc,
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
            delta_unc = None
            for decomp, d in assoc_decomposition_init_value_unc_dict.items():
                for k, a in enumerate(decomp.atoms):
                    g.atoms[k].label = a.label
                v_init, unc_init = d
                v, unc = evaluate_single(tree, g, estimate_uncertainty=True)
                if delta_v is None:
                    delta_v = v - v_init
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_v += v - v_init
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
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
                    delta_unc,
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

def generalize_internal_new_bond_extensions(grp, i, j, n_strucs_max, basename, r_bonds, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generalizes extensions by removing the shortest path (atoms/bonds)
    between two atoms indexed i,j that already exist in the group
    """
    # cython.declare(newgrp=Group)
    
    paths = find_shortest_paths(grp.atoms[i],grp.atoms[j])
    
    if paths is None:
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
    
    grps = []
    for path in paths:
        newgrp = deepcopy(grp)
        mapping = {a:newgrp.atoms[q] for q,a in enumerate(grp.atoms)}
        tail_atom = newgrp.atoms[i]
        head_atom = newgrp.atoms[j]
        if len(path) == 2: #just remove bond
            newgrp.remove_bond(newgrp.atoms[i],newgrp.atoms[j])
        else: #remove internal atoms and bonds
            for a in path:
                if a is not tail_atom and a is not head_atom:
                    newgrp.remove_atom(a)
        
        if len(newgrp.split()) > n_strucs_max: #removing that path creates too many separate structures
            continue
        
        if estimate_delta:
            assert assoc_decomposition_init_value_unc_dict is not None, "Must provide assoc_decomposition_init_value_unc_dict to estimate delta values for internal new-bond extensions"
            assert tree is not None, "Must provide tree to estimate delta values for internal new-bond extensions"
            delta_v = None
            delta_unc = None
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
                    delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                else:
                    delta_v += v - v_init
                    delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
                
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
                delta_unc,
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

def generalize_external_new_bond_extensions(grp, i, basename, n_struc_max, r_bonds, r_label, tree=None, estimate_delta=False, assoc_decomposition_init_value_unc_dict=None):
    """
    generates extensions for the removal of an atom 
    """
    # cython.declare(ga=GroupAtom, newgrp=Group, j=int)
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
        delta_unc = None
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
                delta_unc = np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
            else:
                delta_v += v - v_init
                delta_unc += np.sqrt(max(0.0, unc ** 2 - unc_init ** 2))
        
        newgrp.clear_labeled_atoms()
        
        grps.append(
            (
                newgrp,
                None,
                basename + "_Ext-" + str(i + 1) + atom_type_str + "-R",
                "genAtomRemovalExt",
                (len(newgrp.atoms) - 1,),
                delta_v,
                delta_unc,
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
            
            if not new_struct.is_subgraph_isomorphic(grp, generate_initial_map=True, save_order=True): #removing that atom broke isomorphism with original group so don't delete that atom
                new_struct = old_struct
                st_inds_to_not_remove.append(st_inds[ind])
                continue
            else:
                new,comp = split_mols(temp_structs, new_struct)
                boos = np.array([item.is_subgraph_isomorphic(new_struct, generate_initial_map=True, save_order=True) for item in temp_structs])
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