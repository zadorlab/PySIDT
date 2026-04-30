try:
    from molecule.molecule import Group,GroupBond,GroupAtom
    from molecule.molecule.atomtype import ATOMTYPES
except:
    from rmgpy.molecule import Group,GroupBond,GroupAtom
    from rmgpy.molecule.atomtype import ATOMTYPES

def generate_bicyclic_groups(ring_size_max=9):
    out = []
    for ring_size in range(3,ring_size_max+1): #first ring
        tail_atom = None
        grp = Group()
        for i in range(ring_size): #create first ring
            newatm = GroupAtom([ATOMTYPES['R!H']], radical_electrons=0)
            newatm.label = "*"
            if tail_atom is None:
                grp.add_atom(newatm)
                tail_atom = newatm
            else:
                grp.add_atom(newatm)
                bd = GroupBond(tail_atom,newatm,order=["S","D","T","B"])
                grp.add_bond(bd)
                tail_atom = newatm
        else:
            bd = GroupBond(tail_atom,grp.atoms[0],order=["S","D","T","B"])
            grp.add_bond(bd)

        for ring_size2 in range(3,ring_size_max+1): #second ring
            for intersection_size in range(1,min([ring_size,ring_size2])): #intersection between the two rings
                bigrp = grp.copy(deep=True)
                head_atom = bigrp.atoms[0]
                tail_atom = bigrp.atoms[intersection_size-1]
                for i in range(ring_size2-intersection_size): #create second ring
                    newatm = GroupAtom([ATOMTYPES['R!H']], radical_electrons=0)
                    newatm.label = "*"
                    bigrp.add_atom(newatm)
                    bd = GroupBond(tail_atom,newatm,order=["S","D","T","B"])
                    bigrp.add_bond(bd)
                    tail_atom = newatm
                else:
                    bd = GroupBond(tail_atom,head_atom,order=["S","D","T","B"])
                    bigrp.add_bond(bd)

                cycles = bigrp.get_smallest_set_of_smallest_rings()
                cycle1 = cycles[0]
                cycle2 = cycles[1]
                cinter = set(cycle1).intersection(set(cycle2))
                label_bicyclic(bigrp,bigrp,cycle1,cycle2,cinter)
                out.append(bigrp)

    return out

def max_num_fused_cycles(mol):
    cycles = mol.get_deterministic_sssr()
    cycle_sets = [set(cycle) for cycle in cycles]
    fused_cycle_indices = []
    for i,cset in enumerate(cycle_sets):
        for j,cset2 in enumerate(cycle_sets):
            if i == j:
                continue
            if len(cset.intersection(cset2)) > 0:
                for x in fused_cycle_indices:
                    if i in x and j in x:
                        break
                    elif i in x:
                        x.append(j)
                        break
                    elif j in x:
                        x.append(i)
                        break
                else:
                    fused_cycle_indices.append([i,j])
    if fused_cycle_indices:
        return max(len(x) for x in fused_cycle_indices)
    else:
        return 0
    