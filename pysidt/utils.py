def data_matches_node(node, data):
    for m in data:
        if not m.is_subgraph_isomorphic(node.group, generate_initial_map=True, save_order=True):
            return False
    else:
        return True
