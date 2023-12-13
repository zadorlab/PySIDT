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
        pairs.append((mol.atoms.index(bond.atom1),mol.atoms.index(bond.atom2)))
    
    structs = []
    for pair in pairs:
        m = mol.copy(deep=True)
        for ind in pair:
            m.atoms[ind].label = '*'
        structs.append(m)
    
    return structs

