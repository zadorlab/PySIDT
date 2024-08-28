from molecule.molecule.atomtype import ATOMTYPES
from pysidt.utils import data_matches_node
import itertools

def simple_regularization(node, Rx, Rbonds, Run, Rsite, Rmorph, test=True):
    """
    Simplest regularization algorithm
    All nodes are made as specific as their descendant reactions
    Training reactions are assumed to not generalize
    For example if an particular atom at a node is Oxygen for all of its
    descendent reactions a reaction where it is Sulfur will never hit that node
    unless it is the top node even if the tree did not split on the identity
    of that atom

    The test option to this function determines whether or not the reactions
    under a node match the extended group before adding an extension.
    If the test fails the extension is skipped.

    In general test=True is needed if the cascade algorithm was used
    to generate the tree and test=False is ok if the cascade algorithm
    wasn't used.
    """
    child_map = []
    for child in node.children:
        #recursively regularize child
        simple_regularization(child, Rx, Rbonds, Run, Rsite, Rmorph)
        
        #generate atom map to children
        if node.group is not None:
            keys = []
            atms = []
            initial_map = dict()
            for atom in child.group.atoms:
                if atom.label and atom.label != '':
                    L = [a for a in node.group.atoms if a.label == atom.label]
                    if L == []:
                        return False
                    elif len(L) == 1:
                        initial_map[atom] = L[0]
                    else:
                        keys.append(atom)
                        atms.append(L)
            if atms:
                for atmlist in itertools.product(*atms):
                    if len(set(atmlist)) != len(atmlist):
                        # skip entries that map multiple graph atoms to the same subgraph atom
                        continue
                    for i, key in enumerate(keys):
                        initial_map[key] = atmlist[i]
                    if child.group.is_mapping_valid(node.group, initial_map, equivalent=False):
                        isos = child.group.find_subgraph_isomorphisms(node.group, initial_map, save_order=True)
                        if len(isos) > 0:
                            def isomorph_sort_key(d):
                                val = 0
                                for k,v in d.items():
                                    if node.group.atoms.index(v) == child.group.atoms.index(k):
                                        val += 1
                                return -val
                            child_map.append({v:k for k,v in sorted(isos,key=isomorph_sort_key)[0].items()})
                            break
                else:
                    raise ValueError(f"Could not find valid mapping between parent {node.name} and child {child.name}")
            else:
                isos = child.group.find_subgraph_isomorphisms(node.group, initial_map, save_order=True)
                def isomorph_sort_key(d):
                    val = 0
                    for k,v in d.items():
                        if node.group.atoms.index(v) == child.group.atoms.index(k):
                            val += 1
                    return -val
                child_map.append({v:k for k,v in sorted(isos,key=isomorph_sort_key)[0].items()})
        
    if node.group is None: #skip None nodes
        return
    
    grp = node.group
    data = node.items
    
    if grp is None:
        return

    R = Rx[:]
    if ATOMTYPES["X"] in R:
        R.remove(ATOMTYPES["X"])
    RnH = R[:]
    if ATOMTYPES["H"] in RnH:
        RnH.remove(ATOMTYPES["H"])
    RxnH = Rx[:]
    if ATOMTYPES["H"] in RxnH:
        RxnH.remove(ATOMTYPES["H"])

    atm_dict = {"R": R, "R!H": RnH, "Rx": Rx, "Rx!H": RxnH}

    indistinguishable = []
    for i, atm1 in enumerate(grp.atoms):
        skip = False
        if (
            node.children == []
        ):  # if the atoms or bonds are graphically indistinguishable don't regularize
            bdpairs = {(atm, tuple(bd.order)) for atm, bd in atm1.bonds.items()}
            for atm2 in grp.atoms:
                if (
                    atm1 is not atm2
                    and atm1.atomtype == atm2.atomtype
                    and len(atm1.bonds) == len(atm2.bonds)
                ):
                    bdpairs2 = {
                        (atm, tuple(bd.order)) for atm, bd in atm2.bonds.items()
                    }
                    if bdpairs == bdpairs2:
                        skip = True
                        indistinguishable.append(i)

        if (
            not skip
            and atm1.reg_dim_atm[1] != []
            and set(atm1.reg_dim_atm[1]) != set(atm1.atomtype)
        ):
            atyp = atm1.atomtype
            if len(atyp) == 1 and atyp[0] in Rx:
                pass
            else:
                if len(atyp) == 1 and atyp[0].label in atm_dict.keys():
                    atyp = atm_dict[atyp[0].label]

                vals = list(set(atyp) & set(atm1.reg_dim_atm[1]))
                assert vals != [], "cannot regularize to empty"
                if all(
                    [
                        set(child_map[q][node.group.atoms[i]].atomtype) <= set(vals)
                        for q,child in enumerate(node.children)
                    ]
                ):
                    if not test:
                        atm1.atomtype = vals
                    else:
                        oldvals = atm1.atomtype
                        atm1.atomtype = vals
                        if not data_matches_node(node, data):
                            atm1.atomtype = oldvals

        if (
            not skip
            and atm1.reg_dim_u[1] != []
            and set(atm1.reg_dim_u[1]) != set(atm1.radical_electrons)
        ):
            if len(atm1.radical_electrons) == 1:
                pass
            else:
                relist = atm1.radical_electrons
                if relist == []:
                    relist = Run
                vals = list(set(relist) & set(atm1.reg_dim_u[1]))
                assert vals != [], "cannot regularize to empty"

                if all(
                    [
                        set(child_map[q][node.group.atoms[i]].radical_electrons) <= set(vals)
                        if child_map[q][node.group.atoms[i]].radical_electrons != []
                        else False
                        for q,child in enumerate(node.children)
                    ]
                ):
                    if not test:
                        atm1.radical_electrons = vals
                    else:
                        oldvals = atm1.radical_electrons
                        atm1.radical_electrons = vals
                        if not data_matches_node(node, data):
                            atm1.radical_electrons = oldvals

        if (
            not skip
            and atm1.reg_dim_site[1] != []
            and set(atm1.reg_dim_site[1]) != set(atm1.site)
        ):
            if len(atm1.site) == 1:
                pass
            else:
                relist = atm1.site
                if relist == []:
                    relist = Rsite
                vals = list(set(relist) & set(atm1.reg_dim_site[1]))
                assert vals != [], "cannot regularize to empty"

                if all(
                    [
                        set(child_map[q][node.group.atoms[i]].site) <= set(vals)
                        if child_map[q][node.group.atoms[i]].site != []
                        else False
                        for q,child in enumerate(node.children)
                    ]
                ):
                    if not test:
                        atm1.site = vals
                    else:
                        oldvals = atm1.site
                        atm1.site = vals
                        if not data_matches_node(node, data):
                            atm1.site = oldvals

        if (
            not skip
            and atm1.reg_dim_morphology[1] != []
            and set(atm1.reg_dim_morphology[1]) != set(atm1.morphology)
        ):
            if len(atm1.morphology) == 1:
                pass
            else:
                relist = atm1.morphology
                if relist == []:
                    relist = Rmorph
                vals = list(set(relist) & set(atm1.reg_dim_morphology[1]))
                assert vals != [], "cannot regularize to empty"

                if all(
                    [
                        set(child_map[q][node.group.atoms[i]].morphology) <= set(vals)
                        if child_map[q][node.group.atoms[i]].morphology != []
                        else False
                        for q,child in enumerate(node.children)
                    ]
                ):
                    if not test:
                        atm1.morphology = vals
                    else:
                        oldvals = atm1.morphology
                        atm1.morphology = vals
                        if not data_matches_node(node, data):
                            atm1.morphology = oldvals

        if (
            not skip
            and atm1.reg_dim_r[1] != []
            and (
                "inRing" not in atm1.props.keys()
                or atm1.reg_dim_r[1][0] != atm1.props["inRing"]
            )
        ):
            if "inRing" not in atm1.props.keys():
                if all(
                    [
                        "inRing" in child_map[q][node.group.atoms[i]].props.keys()
                        for q,child in enumerate(node.children)
                    ]
                ) and all(
                    [
                        child_map[q][node.group.atoms[i]].props["inRing"] == atm1.reg_dim_r[1]
                        for q,child in enumerate(node.children)
                    ]
                ):
                    if not test:
                        atm1.props["inRing"] = atm1.reg_dim_r[1][0]
                    else:
                        if "inRing" in atm1.props.keys():
                            oldvals = atm1.props["inRing"]
                        else:
                            oldvals = None
                        atm1.props["inRing"] = atm1.reg_dim_r[1][0]
                        if not data_matches_node(node, data):
                            if oldvals:
                                atm1.props["inRing"] = oldvals
                            else:
                                del atm1.props["inRing"]
        if not skip:
            for j, atm2 in enumerate(grp.atoms[:i]):
                if j in indistinguishable:  # skip graphically indistinguishable atoms
                    continue
                if grp.has_bond(atm1, atm2):
                    bd = grp.get_bond(atm1, atm2)
                    if len(bd.order) == 1:
                        pass
                    else:
                        vals = list(set(bd.order) & set(bd.reg_dim[1]))
                        if vals != [] and all(
                            [
                                set(
                                    child.group.get_bond(
                                        child_map[q][atm1], child_map[q][atm2]
                                    ).order
                                )
                                <= set(vals)
                                for q,child in enumerate(node.children)
                            ]
                        ):
                            if not test:
                                bd.order = vals
                            else:
                                oldvals = bd.order
                                bd.order = vals
                                if not data_matches_node(node, data):
                                    bd.order = oldvals
