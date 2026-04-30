import math
try:
    from molecule.molecule import Molecule
except:
    from rmgpy.molecule import Molecule


def data_matches_node(node, data):
    for datum in data:
        if isinstance(datum, Molecule):
            mol = datum
        else:
            mol = datum.mol
        if not mol.is_subgraph_isomorphic(
            node.group, generate_initial_map=True, save_order=True
        ):
            return False
    else:
        return True

def find_shortest_paths(start, end, path=None):
    paths = [[start]]
    outpaths = []
    while paths != []:
        newpaths = []
        for path in paths:
            for node in path[-1].edges.keys():
                if node in path:
                    continue
                elif node is end:
                    outpaths.append(path[:]+[node])
                elif outpaths == []:
                    newpaths.append(path[:]+[node])
        if outpaths:
            return outpaths
        else:
            paths = newpaths
    
    return None

def get_local_atom_configuration_number(valence_map):
    valence_to_available_elements = {q:sum(v for k,v in valence_map.items() if k >= q) for q in valence_map.keys()}
    valence_to_config_number = dict()
    
    for i in valence_map.keys():
        if i == 1:
            valence_to_config_number[i] = sum(v for k,v in valence_map.items())
        elif i == 2:
            valence_to_config_number[i] = math.comb(valence_to_available_elements[1]+2-1,2) + valence_to_available_elements[2]
        elif i == 3:
            valence_to_config_number[i] = 0
            valence_to_config_number[i] += math.comb(valence_to_available_elements[1]+3-1,3)
            valence_to_config_number[i] += valence_to_available_elements[2]*valence_to_available_elements[1] 
            valence_to_config_number[i] += valence_to_available_elements[3]
        elif i == 4:
            valence_to_config_number[i] = 0
            valence_to_config_number[i] += math.comb(valence_to_available_elements[1]+4-1,4)
            valence_to_config_number[i] += math.comb(valence_to_available_elements[1]+2-1,2) * valence_to_available_elements[2]
            valence_to_config_number[i] += math.comb(valence_to_available_elements[2]+2-1,2)
            valence_to_config_number[i] += valence_to_available_elements[3]*valence_to_available_elements[1]
            valence_to_config_number[i] += valence_to_available_elements[4]
        else:
            raise ValueError

    return valence_to_config_number