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
