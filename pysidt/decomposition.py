from pysidt.mol import *

def atom_decomposition(mol):
    structs = []
    for i in range(len(mol.atoms)):
        m = mol.copy(deep=True)
        m.atoms[i].label = "*"
        structs.append(m)
    return structs


def atom_decomposition_noH(mol):
    structs = []
    for i in range(len(mol.atoms)):
        if mol.atoms[i].is_hydrogen():
            continue
        m = mol.copy(deep=True)
        m.atoms[i].label = "*"
        structs.append(m)
    return structs


def atom_decomposition_noH(mol):
    structs = []
    for i in range(len(mol.atoms)):
        if mol.atoms[i].is_hydrogen():
            continue
        m = mol.copy(deep=True)
        m.atoms[i].label = "*"
        structs.append(m)
    return structs


def bond_decomposition(mol):
    pairs = []
    bonds = mol.get_all_edges()
    for bond in bonds:
        pairs.append((mol.atoms.index(bond.atom1), mol.atoms.index(bond.atom2)))

    structs = []
    for pair in pairs:
        m = mol.copy(deep=True)
        for ind in pair:
            m.atoms[ind].label = "*"
        structs.append(m)

    return structs

def adsorbate_interaction_decomposition(mol):
    surface_bonded_inds = []
    for i,at in enumerate(mol.atoms):
        if at.is_bonded_to_surface() and not at.is_surface_site():
            surface_bonded_inds.append(i)
    
    structs = []
    for i,indi in enumerate(surface_bonded_inds):
        for j,indj in enumerate(surface_bonded_inds):
            if i > j:
                st = mol.copy(deep=True)
                st.atoms[indi].label = "*"
                st.atoms[indj].label = "*"
                structs.append(st)
    
    return structs

def adsorbate_triad_interaction_decomposition(mol):
    surface_bonded_inds = []
    for i,at in enumerate(mol.atoms):
        if at.is_bonded_to_surface() and not at.is_surface_site():
            surface_bonded_inds.append(i)
    
    structs = []
    for i,indi in enumerate(surface_bonded_inds):
        for j,indj in enumerate(surface_bonded_inds):
            for k,indk in enumerate(surface_bonded_inds):
                if i > j and j > k:
                    st = mol.copy(deep=True)
                    st.atoms[indi].label = "*"
                    st.atoms[indj].label = "*"
                    st.atoms[indk].label = "*"
                    structs.append(st)
    
    return structs
    
def adsorbate_site_decomposition(mol):
    surface_bonded_inds = []
    for i,at in enumerate(mol.atoms):
        if at.is_bonded_to_surface() and not at.is_surface_site():
            surface_bonded_inds.append(i)
    
    structs = []
    for i,indi in enumerate(surface_bonded_inds):
        st = mol.copy(deep=True)
        st.atoms[indi].label = "*"
        structs.append(st)
    
    return structs

def ring_decomposition(mol):
    out = []
    cycles = mol.get_deterministic_sssr()
    for cycle in cycles:
        m = mol.copy(deep=True)
        for a in cycle:
            ind = mol.atoms.index(a)
            m.atoms[ind].label = "*"
        out.append(m)

    return out

def get_fused_cycle_sets(mol):
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

    return [[cycles[i] for i in x] for x in fused_cycle_indices],cycles
        
def bicyclic_ring_decomposition(mol):
    out = []
    fused_cycle_lists,cycles = get_fused_cycle_sets(mol)
    for fused_cycles in fused_cycle_lists:
        for i,cycle in enumerate(fused_cycles):
            for j,cycle2 in enumerate(fused_cycles):
                if i <= j:
                    continue
                    
                cinter = set(cycle).intersection(set(cycle2))
                
                if len(cinter) == 0:
                    continue
                    
                m = mol.copy(deep=True)
                label_bicyclic(mol,m,cycle,cycle2,cinter)
                out.append(m)

    return out

def bicyclic_plus_ring_decomposition(mol):
    out = []
    fused_cycle_lists,cycles = get_fused_cycle_sets(mol)
    for fused_cycles in fused_cycle_lists:
        for i,cycle in enumerate(fused_cycles):
            for j,cycle2 in enumerate(fused_cycles):
                if i <= j:
                    continue
                    
                cinter = set(cycle).intersection(set(cycle2))
                
                if len(cinter) == 0:
                    continue
                    
                m = mol.copy(deep=True)
                label_bicyclic(mol,m,cycle,cycle2,cinter)
                out.append(m)

    return ring_decomposition(mol,cycles)+out
