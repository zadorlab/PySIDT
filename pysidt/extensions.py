import logging
from copy import deepcopy

import numpy as np

from molecule.molecule.atomtype import ATOMTYPES
from molecule.molecule.element import bde_elements
from molecule.molecule.group import GroupAtom, GroupBond
from molecule.molecule.molecule import Molecule


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
    parent,
    n_strucs_min,
    iter_max=np.inf,
    iter_item_cap=np.inf,
    r=None,
    r_bonds=None,
    r_un=None,
    r_site=None,
    r_morph=None,
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

    out_exts = [[]]
    grps = [[parent.group]]
    names = [parent.name]
    first_time = True
    gave_up_split = False

    iter = 0

    while grps[iter] != []:
        grp = grps[iter][-1]

        exts = get_extensions(
            grp,
            basename=names[-1],
            r=r,
            r_bonds=r_bonds,
            r_un=r_un,
            r_site=r_site,
            r_morph=r_morph,
            n_strucs_min=n_strucs_min,
        )

        reg_dict = dict()
        ext_inds = []
        for i, (grp2, grpc, name, typ, indc) in enumerate(exts):
            if (
                typ != "intNewBondExt"
                and typ != "extNewBondExt"
                and (typ, indc) not in reg_dict.keys()
            ):
                # first list is all extensions that match at least one reaction
                # second is extensions that match all reactions
                reg_dict[(typ, indc)] = ([], [])

            new, comp = split_mols(parent.items, grp2)

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
                elif typ == "bondExt":
                    reg_dict[(typ, indc)][0].extend(
                        grp2.get_bond(grp2.atoms[indc[0]], grp2.atoms[indc[1]]).order
                    )

            elif boo:  # this extension matches all reactions (regularization dim)
                if typ == "intNewBondExt" or typ == "extNewBondExt":
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
                elif typ == "bondExt":
                    reg_dict[(typ, indc)][0].extend(
                        grp2.get_bond(grp2.atoms[indc[0]], grp2.atoms[indc[1]]).order
                    )
                    reg_dict[(typ, indc)][1].extend(
                        grp2.get_bond(grp2.atoms[indc[0]], grp2.atoms[indc[1]]).order
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

            if first_time and not parent.children:
                # parent
                if (
                    typr != "intNewBondExt" and typr != "extNewBondExt"
                ):  # these dimensions should be regularized
                    if typr == "atomExt":
                        grp.atoms[indcr[0]].reg_dim_atm = list(reg_val)
                    elif typr == "elExt":
                        grp.atoms[indcr[0]].reg_dim_u = list(reg_val)
                    elif typr == "siteExt":
                        grp.atoms[indcr[0]].reg_dim_site = list(reg_val)
                    elif typr == "morphExt":
                        grp.atoms[indcr[0]].reg_dim_morphology = list(reg_val)
                    elif typr == "ringExt":
                        grp.atoms[indcr[0]].reg_dim_r = list(reg_val)
                    elif typr == "bondExt":
                        atms = grp.atoms
                        bd = grp.get_bond(atms[indcr[0]], atms[indcr[1]])
                        bd.reg_dim = list(reg_val)

            # extensions being sent out
            if (
                typr != "intNewBondExt" and typr != "extNewBondExt"
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
                    elif typr == "siteExt":
                        grp2.atoms[indcr[0]].reg_dim_site = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_site = list(reg_val)
                    elif typr == "morphExt":
                        grp2.atoms[indcr[0]].reg_dim_morphology = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_morphology = list(reg_val)
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
                typr != "intNewBondExt" and typr != "extNewBondExt"
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
                    elif typr == "siteExt":
                        grp2.atoms[indcr[0]].reg_dim_site = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_site = list(reg_val)
                    elif typr == "morphExt":
                        grp2.atoms[indcr[0]].reg_dim_morphology = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_morphology = list(reg_val)
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
                    "Recursion item cap hit not splitting {0} reactions at iter {1} with {2} items".format(
                        len(parent.items), iter, len(grps[iter])
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
    r=None,
    r_bonds=[1, 2, 3, 1.5, 4],
    r_un=[0, 1, 2, 3],
    r_site=[],
    r_morph=[],
    basename="",
    atm_ind=None,
    atm_ind2=None,
    n_strucs_min=None,
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
                            specify_atom_extensions(grp, i, basename, R)
                        )  # specify types of atoms
                    elif typ[0].label == "R!H":
                        extents.extend(specify_atom_extensions(grp, i, basename, RnH))
                    elif typ[0].label == "Rx":
                        extents.extend(specify_atom_extensions(grp, i, basename, r))
                    elif typ[0].label == "Rx!H":
                        extents.extend(specify_atom_extensions(grp, i, basename, RxnH))
                else:
                    extents.extend(specify_atom_extensions(grp, i, basename, typ))
            else:
                if len(typ) == 1:
                    if typ[0].label == "R":
                        extents.extend(
                            specify_atom_extensions(
                                grp, i, basename, atm.reg_dim_atm[0]
                            )
                        )  # specify types of atoms
                    elif typ[0].label == "R!H":
                        extents.extend(
                            specify_atom_extensions(
                                grp, i, basename, list(set(atm.reg_dim_atm[0]) & set(R))
                            )
                        )
                    elif typ[0].label == "Rx":
                        extents.extend(
                            specify_atom_extensions(
                                grp, i, basename, list(set(atm.reg_dim_atm[0]) & set(r))
                            )
                        )
                    elif typ[0].label == "Rx!H":
                        extents.extend(
                            specify_atom_extensions(
                                grp,
                                i,
                                basename,
                                list(set(atm.reg_dim_atm[0]) & set(RxnH)),
                            )
                        )

                else:
                    extents.extend(
                        specify_atom_extensions(
                            grp, i, basename, list(set(typ) & set(atm.reg_dim_atm[0]))
                        )
                    )
            if not atm.reg_dim_u[0]:
                if len(atm.radical_electrons) != 1:
                    if len(atm.radical_electrons) == 0:
                        extents.extend(
                            specify_unpaired_extensions(grp, i, basename, r_un)
                        )
                    else:
                        extents.extend(
                            specify_unpaired_extensions(
                                grp, i, basename, atm.radical_electrons
                            )
                        )
            else:
                if len(atm.radical_electrons) != 1 and len(atm.reg_dim_u[0]) != 1:
                    if len(atm.radical_electrons) == 0:
                        extents.extend(
                            specify_unpaired_extensions(
                                grp, i, basename, atm.reg_dim_u[0]
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
                            )
                        )
            if r_site:
                if not atm.reg_dim_site[0]:
                    if len(atm.site) != 1:
                        if len(atm.site) == 0:
                            extents.extend(
                                specify_site_extensions(grp, i, basename, r_site)
                            )
                        else:
                            extents.extend(
                                specify_site_extensions(grp, i, basename, atm.site)
                            )
                else:
                    if len(atm.site) != 1 and len(atm.reg_dim_site[0]) != 1:
                        if len(atm.site) == 0:
                            extents.extend(
                                specify_site_extensions(
                                    grp, i, basename, atm.reg_dim_site[0]
                                )
                            )
                        else:
                            extents.extend(
                                specify_site_extensions(
                                    grp,
                                    i,
                                    basename,
                                    list(set(atm.site) & set(atm.reg_dim_site[0])),
                                )
                            )
            if r_morph:
                if not atm.reg_dim_morphology[0]:
                    if len(atm.morphology) != 1:
                        if len(atm.morphology) == 0:
                            extents.extend(
                                specify_morphology_extensions(grp, i, basename, r_morph)
                            )
                        else:
                            extents.extend(
                                specify_morphology_extensions(
                                    grp, i, basename, atm.morphology
                                )
                            )
                else:
                    if len(atm.morphology) != 1 and len(atm.reg_dim_morphology[0]) != 1:
                        if len(atm.morphology) == 0:
                            extents.extend(
                                specify_morphology_extensions(
                                    grp, i, basename, atm.reg_dim_morphology[0]
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
                                )
                            )
            if not atm.reg_dim_r[0] and "inRing" not in atm.props:
                extents.extend(specify_ring_extensions(grp, i, basename))

            extents.extend(
                specify_external_new_bond_extensions(grp, i, basename, r_bonds)
            )
            for j, atm2 in enumerate(atoms):
                if j < i and not grp.has_bond(atm, atm2):
                    extents.extend(
                        specify_internal_new_bond_extensions(
                            grp, i, j, n_strucs_min, basename, r_bonds
                        )
                    )
                elif j < i:
                    bd = grp.get_bond(atm, atm2)
                    if len(bd.order) > 1 and not bd.reg_dim[0]:
                        extents.extend(
                            specify_bond_extensions(grp, i, j, basename, bd.order)
                        )
                    elif (
                        len(bd.order) > 1
                        and len(bd.reg_dim[0]) > 1
                        and len(bd.reg_dim[0]) > len(bd.reg_dim[1])
                    ):
                        extents.extend(
                            specify_bond_extensions(grp, i, j, basename, bd.reg_dim[0])
                        )

    elif (
        atm_ind is not None and atm_ind2 is not None
    ):  # if both atm_ind and atm_ind2 are defined only look at the bonds between them
        i = atm_ind
        j = atm_ind2
        atm = atoms[i]
        atm2 = atoms[j]
        if j < i and not grp.has_bond(atm, atm2):
            extents.extend(
                specify_internal_new_bond_extensions(
                    grp, i, j, n_strucs_min, basename, r_bonds
                )
            )
        if grp.has_bond(atm, atm2):
            bd = grp.get_bond(atm, atm2)
            if len(bd.order) > 1 and not bd.reg_dim[0]:
                extents.extend(specify_bond_extensions(grp, i, j, basename, bd.order))
            elif (
                len(bd.order) > 1
                and len(bd.reg_dim[0]) > 1
                and len(bd.reg_dim[0]) > len(bd.reg_dim[1])
            ):
                extents.extend(
                    specify_bond_extensions(grp, i, j, basename, bd.reg_dim[0])
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
                    extents.extend(specify_atom_extensions(grp, i, basename, RnH))
                elif typ[0].label == "Rx":
                    extents.extend(specify_atom_extensions(grp, i, basename, r))
                elif typ[0].label == "Rx!H":
                    extents.extend(specify_atom_extensions(grp, i, basename, RxnH))
            else:
                extents.extend(specify_atom_extensions(grp, i, basename, typ))
        else:
            if len(typ) == 1:
                if typ[0].label == "R":
                    extents.extend(
                        specify_atom_extensions(grp, i, basename, atm.reg_dim_atm[0])
                    )  # specify types of atoms
                elif typ[0].label == "R!H":
                    extents.extend(
                        specify_atom_extensions(
                            grp, i, basename, list(set(atm.reg_dim_atm[0]) & set(RnH))
                        )
                    )
                elif typ[0].label == "Rx":
                    extents.extend(
                        specify_atom_extensions(
                            grp, i, basename, list(set(atm.reg_dim_atm[0]) & set(r))
                        )
                    )
                elif typ[0].label == "Rx!H":
                    extents.extend(
                        specify_atom_extensions(
                            grp, i, basename, list(set(atm.reg_dim_atm[0]) & set(RxnH))
                        )
                    )
            else:
                extents.extend(
                    specify_atom_extensions(
                        grp, i, basename, list(set(typ) & set(atm.reg_dim_atm[0]))
                    )
                )
        if not atm.reg_dim_u:
            if len(atm.radical_electrons) != 1:
                if len(atm.radical_electrons) == 0:
                    extents.extend(specify_unpaired_extensions(grp, i, basename, r_un))
                else:
                    extents.extend(
                        specify_unpaired_extensions(
                            grp, i, basename, atm.radical_electrons
                        )
                    )
        else:
            if len(atm.radical_electrons) != 1 and len(atm.reg_dim_u[0]) != 1:
                if len(atm.radical_electrons) == 0:
                    extents.extend(
                        specify_unpaired_extensions(grp, i, basename, atm.reg_dim_u[0])
                    )
                else:
                    extents.extend(
                        specify_unpaired_extensions(
                            grp,
                            i,
                            basename,
                            list(set(atm.radical_electrons) & set(atm.reg_dim_u[0])),
                        )
                    )
        if r_site:
            if not atm.reg_dim_site:
                if len(atm.site) != 1:
                    if len(atm.site) == 0:
                        extents.extend(
                            specify_site_extensions(grp, i, basename, r_site)
                        )
                    else:
                        extents.extend(
                            specify_site_extensions(grp, i, basename, atm.site)
                        )
            else:
                if len(atm.site) != 1 and len(atm.reg_dim_site[0]) != 1:
                    if len(atm.site) == 0:
                        extents.extend(
                            specify_site_extensions(
                                grp, i, basename, atm.reg_dim_site[0]
                            )
                        )
                    else:
                        extents.extend(
                            specify_site_extensions(
                                grp,
                                i,
                                basename,
                                list(set(atm.site) & set(atm.reg_dim_site[0])),
                            )
                        )
        if r_morph:
            if not atm.reg_dim_morphology:
                if len(atm.morphology) != 1:
                    if len(atm.morphology) == 0:
                        extents.extend(
                            specify_morphology_extensions(grp, i, basename, r_morph)
                        )
                    else:
                        extents.extend(
                            specify_morphology_extensions(
                                grp, i, basename, atm.morphology
                            )
                        )
            else:
                if len(atm.morphology) != 1 and len(atm.reg_dim_morphology[0]) != 1:
                    if len(atm.morphology) == 0:
                        extents.extend(
                            specify_morphology_extensions(
                                grp, i, basename, atm.reg_dim_morphology[0]
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
                            )
                        )
        if not atm.reg_dim_r[0] and "inRing" not in atm.props:
            extents.extend(specify_ring_extensions(grp, i, basename))

        extents.extend(specify_external_new_bond_extensions(grp, i, basename, r_bonds))
        for j, atm2 in enumerate(atoms):
            if j < i and not grp.has_bond(atm, atm2):
                extents.extend(
                    specify_internal_new_bond_extensions(
                        grp, i, j, n_strucs_min, basename, r_bonds
                    )
                )
            elif j < i:
                bd = grp.get_bond(atm, atm2)
                if len(bd.order) > 1 and not bd.reg_dim:
                    extents.extend(
                        specify_bond_extensions(grp, i, j, basename, bd.order)
                    )
                elif (
                    len(bd.order) > 1
                    and len(bd.reg_dim[0]) > 1
                    and len(bd.reg_dim[0]) > len(bd.reg_dim[1])
                ):
                    extents.extend(
                        specify_bond_extensions(grp, i, j, basename, bd.reg_dim[0])
                    )

    else:
        raise ValueError("atm_ind must be defined if atm_ind2 is defined")

    for ex in extents:
        ex[0].update_fingerprint()
        if ex[1]:
            ex[1].update_fingerprint()

    return extents


def specify_atom_extensions(grp, i, basename, r):
    """
    generates extensions for specification of the type of atom defined by a given atomtype
    or set of atomtypes
    """
    # cython.declare(grps=list, labelList=list, Rset=set, item=AtomType, grp=Group, grpc=Group, k=AtomType, p=str)

    grps = []
    Rset = set(r)
    for item in r:
        grp = deepcopy(grp)
        grpc = deepcopy(grp)
        old_atom_type = grp.atoms[i].atomtype
        grp.atoms[i].atomtype = [item]
        grpc.atoms[i].atomtype = list(Rset - {item})

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

        grps.append(
            (
                grp,
                grpc,
                basename + "_" + str(i + 1) + old_atom_type_str + "->" + item.label,
                "atomExt",
                (i,),
            )
        )

    return grps


def specify_ring_extensions(grp, i, basename):
    """
    generates extensions for specifying if an atom is in a ring
    """
    # cython.declare(grps=list, label_list=list, grp=Group, grpc=Group, atom_type=list, atom_type_str=str, k=AtomType,
    #                 p=str)

    grps = []
    label_list = []

    grp = deepcopy(grp)
    grpc = deepcopy(grp)
    grp.atoms[i].props["inRing"] = True
    grpc.atoms[i].props["inRing"] = False

    atom_type = grp.atoms[i].atomtype

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

    grps.append(
        (
            grp,
            grpc,
            basename + "_" + str(i + 1) + atom_type_str + "-inRing",
            "ringExt",
            (i,),
        )
    )

    return grps


def specify_unpaired_extensions(grp, i, basename, r_un):
    """
    generates extensions for specification of the number of electrons on a given atom
    """

    grps = []
    label_list = []

    Rset = set(r_un)
    for item in r_un:
        grp = deepcopy(grp)
        grpc = deepcopy(grp)
        grp.atoms[i].radical_electrons = [item]
        grpc.atoms[i].radical_electrons = list(Rset - {item})

        if len(grpc.atoms[i].radical_electrons) == 0:
            grpc = None

        atom_type = grp.atoms[i].atomtype

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

        grps.append(
            (grp, grpc, basename + "_" + str(i + 1) + "-u" + str(item), "elExt", (i,))
        )

    return grps


def specify_site_extensions(grp, i, basename, r_site):
    """
    generates extensions for specification of the number of electrons on a given atom
    """
    x = ATOMTYPES["X"]
    if (
        not any([s.is_specific_case_of(x) for s in grp.atoms[i].atomtype])
        and grp.atoms[i].atomtype[0] != ATOMTYPES["Rx"]
    ):
        return []

    grps = []
    label_list = []

    Rset = set(r_site)
    for item in r_site:
        grp = deepcopy(grp)
        grpc = deepcopy(grp)
        grp.atoms[i].site = [item]
        grpc.atoms[i].site = list(Rset - {item})

        if len(grpc.atoms[i].site) == 0:
            grpc = None

        atom_type = grp.atoms[i].atomtype

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

        grps.append(
            (grp, grpc, basename + "_" + str(i + 1) + "-s" + str(item), "siteExt", (i,))
        )

    return grps


def specify_morphology_extensions(grp, i, basename, r_morph):
    """
    generates extensions for specification of the number of electrons on a given atom
    """
    x = ATOMTYPES["X"]
    if (
        not any([s.is_specific_case_of(x) for s in grp.atoms[i].atomtype])
        and grp.atoms[i].atomtype[0] != ATOMTYPES["Rx"]
    ):
        return []

    grps = []
    label_list = []

    Rset = set(r_morph)
    for item in r_morph:
        grp = deepcopy(grp)
        grpc = deepcopy(grp)
        grp.atoms[i].morphology = [item]
        grpc.atoms[i].morphology = list(Rset - {item})

        if len(grpc.atoms[i].morphology) == 0:
            grpc = None

        atom_type = grp.atoms[i].atomtype

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

        grps.append(
            (
                grp,
                grpc,
                basename + "_" + str(i + 1) + "-m" + str(item),
                "morphExt",
                (i,),
            )
        )

    return grps


def specify_internal_new_bond_extensions(grp, i, j, n_strucs_min, basename, r_bonds):
    """
    generates extensions for creation of a bond (of undefined order)
    between two atoms indexed i,j that already exist in the group and are unbonded
    """
    # cython.declare(newgrp=Group)

    label_list = []

    newgrp = deepcopy(grp)
    newgrp.add_bond(GroupBond(newgrp.atoms[i], newgrp.atoms[j], r_bonds))

    atom_type_i = newgrp.atoms[i].atomtype
    atom_type_j = newgrp.atoms[j].atomtype

    if len(atom_type_i) > 1:
        atom_type_i_str = ""
        for k in atom_type_i:
            label_list.append(k.label)
        for k in sorted(label_list):
            atom_type_i_str += k
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

    if (
        len(newgrp.split()) < n_strucs_min
    ):  # if this formed a bond between two seperate groups in the
        return []
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
                + atom_type_j_str,
                "intNewBondExt",
                (i, j),
            )
        ]


def specify_external_new_bond_extensions(grp, i, basename, r_bonds):
    """
    generates extensions for the creation of a bond (of undefined order) between
    an atom and a new atom that is not H
    """
    # cython.declare(ga=GroupAtom, newgrp=Group, j=int)

    label_list = []

    ga = GroupAtom([ATOMTYPES["Rx!H"]])
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

    return [
        (
            newgrp,
            None,
            basename + "_Ext-" + str(i + 1) + atom_type_str + "-R",
            "extNewBondExt",
            (len(newgrp.atoms) - 1,),
        )
    ]


def specify_bond_extensions(grp, i, j, basename, r_bonds):
    """
    generates extensions for the specification of bond order for a given bond
    """
    # cython.declare(grps=list, label_list=list, Rbset=set, bd=float, grp=Group, grpc=Group)
    grps = []
    label_list = []
    Rbset = set(r_bonds)
    bdict = {1: "-", 2: "=", 3: "#", 1.5: "-=", 4: "$", 0.05: "..", 0: "--"}
    bstrdict = {"S": 1, "D": 2, "T": 3, "B": 1.5, "Q": 4, "R": 0.05, "vdW": 0}
    for bd in r_bonds:
        grp = deepcopy(grp)
        grpc = deepcopy(grp)
        grp.atoms[i].bonds[grp.atoms[j]].order = [bd]
        grp.atoms[j].bonds[grp.atoms[i]].order = [bd]
        grpc.atoms[i].bonds[grpc.atoms[j]].order = list(Rbset - {bd})
        grpc.atoms[j].bonds[grpc.atoms[i]].order = list(Rbset - {bd})

        if len(list(Rbset - {bd})) == 0:
            grpc = None

        atom_type_i = grp.atoms[i].atomtype
        atom_type_j = grp.atoms[j].atomtype

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

        b = None
        for v in bdict.keys():
            if abs(v - bd) < 1e-4:
                b = bdict[v]
        grps.append(
            (
                grp,
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

def generate_extensions_reverse(grp,structs):
    """
    This function is designed to generate extensions by reverse engineering the structures being split rather than extending the original group
    This should be a reliable fallback when traditional extension generation becomes to expensive
    """
    exts = []
    for st in structs:
        new_struct = st.to_group()
        atoms = new_struct.atoms
        aexts = []
        smallest_not_matching_group = None
        for ind in range(len(st.atoms))[::-1]:
            old_struct = new_struct
            new_struct = old_struct.copy(deep=True)
            at = new_struct.atoms[ind]
            if at.label: #don't remove any labeled atoms
                continue
            new_struct.remove_atom(at)
            if not new_struct.is_subgraph_isomorphic(grp,generate_initial_map=True,save_order=True): #removing that atom broke isomorphism with original group so don't delete that atom
                new_struct = old_struct
                continue
            else:
                boos = np.array([item.is_subgraph_isomorphic(new_struct,generate_initial_map=True,save_order=True) for item in structs if item != st])
                if boos.all(): #suddenly matches all groups...don't remove that atom
                    new_struct = old_struct
                    continue
                elif boos.any(): #splits groups
                    aexts.append(new_struct)
                else:
                    if smallest_not_matching_group is None or len(smallest_not_matching_group.atoms) > len(new_struct.atoms):
                        smallest_not_matching_group = new_struct
                    continue
        if len(aexts):
            minlen = min([len(g.atoms) for g in aexts])
            exts.extend([g for g in aexts if len(g.atoms) == minlen])
        else:
            exts.append(smallest_not_matching_group)

    return exts