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