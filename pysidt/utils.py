from molecule.molecule import Molecule


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
