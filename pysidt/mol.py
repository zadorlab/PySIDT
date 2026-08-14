try:
    from molecule.molecule import Group,GroupBond,GroupAtom
    from molecule.molecule.atomtype import ATOMTYPES
except:
    from rmgpy.molecule import Group,GroupBond,GroupAtom
    from rmgpy.molecule.atomtype import ATOMTYPES

import numpy as np

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
    
def label_bicyclic(mol_ori,m,cycle1,cycle2,cinter):
    #rename cycles check symmetry
    lc1 = len(cycle1)
    lc2 = len(cycle2)
    if lc1 == lc2:
        same_cycle_size = True
        c1 = cycle1
        c2 = cycle2
    elif lc2 > lc1:
        same_cycle_size = False
        c1 = cycle2
        c2 = cycle1
    else:
        same_cycle_size = False
        c1 = cycle1
        c2 = cycle2

    bicycle = set(c1) | set(c2) #all atoms of the bicycle
    
    #find central intersection atoms
    central_atoms = cinter #starting from the cycle intersection
    atoms_to_remove = []
    while len(atoms_to_remove) < len(central_atoms): #each iteration look around each remaining atom
        for a in atoms_to_remove:
            central_atoms.remove(a)
        for a in central_atoms: #remove atoms that are adjacent to atoms outside until you have the intersection atom/atoms farthest from everything else
            if a not in atoms_to_remove and any(b not in central_atoms for b in a.bonds.keys() if b in bicycle):
                atoms_to_remove.append(a)

    collected_atoms = set()
    new_collected_atoms = set(central_atoms)
    
    dist = 0
    while len(collected_atoms) < len(bicycle):
        atoms_to_collect = set()
        collected_atoms |= new_collected_atoms
        for a in new_collected_atoms:
            ind = mol_ori.atoms.index(a)
            if a in cinter:
                label = "*f"
            elif same_cycle_size or a in c1:
                label = "*b"
            else:
                label = "*s"
            label += str(dist)
            matom = m.atoms[ind].label = label
            atoms_to_collect |= {b for b in a.bonds.keys() if b not in collected_atoms and b in bicycle}
            
        dist += 1
        new_collected_atoms = atoms_to_collect

    return

def get_octet_deviation(atom):
    if atom.symbol in ["Li","H"]:
        return 2 - (atom.lone_pairs*2 + atom.radical_electrons + 2*sum(x.order for x in atom.bonds.values() if x.order >= 0.5))
    else:
        return 8 - (atom.lone_pairs*2 + atom.radical_electrons + 2*sum(x.order for x in atom.bonds.values() if x.order >= 0.5))

def get_atomtype_elements(atomtype,element_atomtypes):
    if atomtype in element_atomtypes: #if it is already an element
        return [atomtype]
    else: #is it a specific of an element
        generics = [a for a in atomtype.generic if a in element_atomtypes]
        if len(generics) > 0:
            return generics
        else: #more general than one element
            return element_atomtypes
        
def get_ring_count_in_largest_fused_ring_system(mol):
        """
        Get the number of rings in the largest fused ring system in the molecule.
        Returns 0 if the molecule has no fused rings (only monocycles or no rings).
        """
        polycycles = mol.get_polycycles()
        if not polycycles:
            return 0

        sssr = mol.get_smallest_set_of_smallest_rings()
        if not sssr:
            return 0

        sssr_sets = [set(r) for r in sssr]

        ring_counts = list()
        for polycycle in polycycles:
            poly_set = set(polycycle)
            ring_count = 0
            for ring_set in sssr_sets:
                if ring_set.issubset(poly_set):
                    ring_count += 1
            ring_counts.append(ring_count)

        return max(ring_counts) if ring_counts else 0

def invalidated_by_bredts_rule(mol):
    """Checks if mol has higher order than single bonds at bridgehead carbons
    for rings 8 or smaller this indicates mol is either unstable or 
    very hard to synthesize
    
    Args:
        mol (_type_): _description_

    Returns:
        _type_: _description_
    """
    return any(any(bd.order > 1 for bd in a.bonds.values()) for a in mol.get_all_polycyclic_vertices())
        