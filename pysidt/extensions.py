import logging 
import numpy as np


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
    
    for i,datum in enumerate(data):
        if datum.mol.is_subgraph_isomorphic(newgrp, generate_initial_map=True, save_order=True):
            new.append(datum)
        else:
            comp.append(datum)


    return new, comp
    
def get_extension_edge(parent, n_splits, iter_max=np.inf, iter_item_cap=np.inf, r=None, r_bonds=[1,2,3,1.5,4], r_un=[0,1,2,3]):
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
    out_exts = [[]]
    grps = [[parent.group]]
    names = [parent.name]
    first_time = True
    gave_up_split = False

    iter = 0

    while grps[iter] != []:
        grp = grps[iter][-1]

        exts = grp.get_extensions(basename=names[-1], r=r, r_bonds=r_bonds, r_un=r_un, n_splits=n_splits)

        reg_dict = dict()
        ext_inds = []
        for i, (grp2, grpc, name, typ, indc) in enumerate(exts):

            if typ != 'intNewBondExt' and typ != 'extNewBondExt' and (typ, indc) not in reg_dict.keys():
                # first list is all extensions that match at least one reaction
                # second is extensions that match all reactions
                reg_dict[(typ, indc)] = ([], [])

            new,comp = split_mols(parent.items,grp2)
            
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
                out_exts[-1].append(exts[i])  # this extension splits reactions (optimization dim)
                if typ == 'atomExt':
                    reg_dict[(typ, indc)][0].extend(grp2.atoms[indc[0]].atomtype)
                elif typ == 'elExt':
                    reg_dict[(typ, indc)][0].extend(grp2.atoms[indc[0]].radical_electrons)
                elif typ == 'bondExt':
                    reg_dict[(typ, indc)][0].extend(grp2.get_bond(grp2.atoms[indc[0]], grp2.atoms[indc[1]]).order)

            elif boo:  # this extension matches all reactions (regularization dim)
                if typ == 'intNewBondExt' or typ == 'extNewBondExt':
                    # these are bond formation extensions, we want to expand these until we get splits
                    ext_inds.append(i)
                elif typ == 'atomExt':
                    reg_dict[(typ, indc)][0].extend(grp2.atoms[indc[0]].atomtype)
                    reg_dict[(typ, indc)][1].extend(grp2.atoms[indc[0]].atomtype)
                elif typ == 'elExt':
                    reg_dict[(typ, indc)][0].extend(grp2.atoms[indc[0]].radical_electrons)
                    reg_dict[(typ, indc)][1].extend(grp2.atoms[indc[0]].radical_electrons)
                elif typ == 'bondExt':
                    reg_dict[(typ, indc)][0].extend(grp2.get_bond(grp2.atoms[indc[0]], grp2.atoms[indc[1]]).order)
                    reg_dict[(typ, indc)][1].extend(grp2.get_bond(grp2.atoms[indc[0]], grp2.atoms[indc[1]]).order)
                elif typ == 'ringExt':
                    reg_dict[(typ, indc)][1].append(True)
            else:
                # this extension matches no reactions
                if typ == 'ringExt':
                    reg_dict[(typ, indc)][0].append(False)
                    reg_dict[(typ, indc)][1].append(False)

        for typr, indcr in reg_dict.keys():  # have to label the regularization dimensions in all relevant groups
            reg_val = reg_dict[(typr, indcr)]

            if first_time and parent.children == []:
                # parent
                if typr != 'intNewBondExt' and typr != 'extNewBondExt':  # these dimensions should be regularized
                    if typr == 'atomExt':
                        grp.atoms[indcr[0]].reg_dim_atm = list(reg_val)
                    elif typr == 'elExt':
                        grp.atoms[indcr[0]].reg_dim_u = list(reg_val)
                    elif typr == 'ringExt':
                        grp.atoms[indcr[0]].reg_dim_r = list(reg_val)
                    elif typr == 'bondExt':
                        atms = grp.atoms
                        bd = grp.get_bond(atms[indcr[0]], atms[indcr[1]])
                        bd.reg_dim = list(reg_val)

            # extensions being sent out
            if typr != 'intNewBondExt' and typr != 'extNewBondExt':  # these dimensions should be regularized
                for grp2, grpc, name, typ, indc in out_exts[-1]:  # returned groups
                    if typr == 'atomExt':
                        grp2.atoms[indcr[0]].reg_dim_atm = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_atm = list(reg_val)
                    elif typr == 'elExt':
                        grp2.atoms[indcr[0]].reg_dim_u = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_u = list(reg_val)
                    elif typr == 'ringExt':
                        grp2.atoms[indcr[0]].reg_dim_r = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_r = list(reg_val)
                    elif typr == 'bondExt':
                        atms = grp2.atoms
                        bd = grp2.get_bond(atms[indcr[0]], atms[indcr[1]])
                        bd.reg_dim = [list(set(bd.order) & set(reg_val[0])), list(set(bd.order) & set(reg_val[1]))]
                        if grpc:
                            atms = grpc.atoms
                            bd = grpc.get_bond(atms[indcr[0]], atms[indcr[1]])
                            bd.reg_dim = [list(set(bd.order) & set(reg_val[0])),
                                            list(set(bd.order) & set(reg_val[1]))]

        # extensions being expanded
        for typr, indcr in reg_dict.keys():  # have to label the regularization dimensions in all relevant groups
            reg_val = reg_dict[(typr, indcr)]
            if typr != 'intNewBondExt' and typr != 'extNewBondExt':  # these dimensions should be regularized
                for ind2 in ext_inds:  # groups for expansion
                    grp2, grpc, name, typ, indc = exts[ind2]
                    if typr == 'atomExt':
                        grp2.atoms[indcr[0]].reg_dim_atm = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_atm = list(reg_val)
                    elif typr == 'elExt':
                        grp2.atoms[indcr[0]].reg_dim_u = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_u = list(reg_val)
                    elif typr == 'ringExt':
                        grp2.atoms[indcr[0]].reg_dim_r = list(reg_val)
                        if grpc:
                            grpc.atoms[indcr[0]].reg_dim_r = list(reg_val)
                    elif typr == 'bondExt':
                        atms = grp2.atoms
                        bd = grp2.get_bond(atms[indcr[0]], atms[indcr[1]])
                        bd.reg_dim = [list(set(bd.order) & set(reg_val[0])), list(set(bd.order) & set(reg_val[1]))]
                        if grpc:
                            atms = grpc.atoms
                            bd = grpc.get_bond(atms[indcr[0]], atms[indcr[1]])
                            bd.reg_dim = [list(set(bd.order) & set(reg_val[0])),
                                            list(set(bd.order) & set(reg_val[1]))]

        out_exts.append([])
        grps[iter].pop()
        names.pop()

        for ind in ext_inds:  # collect the groups to be expanded
            grpr, grpcr, namer, typr, indcr = exts[ind]
            if len(grps) == iter+1:
                grps.append([])
            grps[iter+1].append(grpr)
            names.append(namer)

        if first_time:
            first_time = False

        if grps[iter] == [] and len(grps) != iter+1 and (not (any([len(x)>0 for x in out_exts]) and iter+1 > iter_max)):
            iter += 1
            if len(grps[iter]) > iter_item_cap:
                logging.error("Recursion item cap hit not splitting {0} reactions at iter {1} with {2} items".format(len(parent.items),iter,len(grps[iter])))
                iter -= 1
                gave_up_split = True

        elif grps[iter] == [] and len(grps) != iter+1 and (any([len(x)>0 for x in out_exts]) and iter+1 > iter_max):
            logging.error("iter_max achieved terminating early")

    out = []
    # compile all of the valid extensions together
    # may be some duplicates here, but I don't think it's currently worth identifying them
    for x in out_exts:
        out.extend(x)

    return out, gave_up_split