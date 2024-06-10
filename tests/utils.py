import pytest

from molecule.molecule import Group
from pysidt import write_nodes, read_nodes, SubgraphIsomorphicDecisionTree, Rule, Node

@pytest.fixture
def tree():
    root_rule = Rule(value=2, uncertainty=0.2, num_data=3)
    root_group = Group().from_adjacency_list("""
    1 * R u0
    """)
    root_node = Node(name="root", group=root_group, rule=root_rule, )

    child_rule = Rule(value=1, uncertainty=0.1, num_data=2)
    child_group = Group().from_adjacency_list("""
    1 * R u0
    2 R u0
    """)
    child_node = Node(name="child", group=child_group, rule=child_rule)

    root_node.children = [child_node]
    child_node.parent = root_node

    nodes = {"root": root_node, "child": child_node}
    tree = SubgraphIsomorphicDecisionTree(root_group=root_group, nodes=nodes)
    return tree

def test_write_read_nodes(tree, tmp_path):
    write_path = tmp_path / 'test_nodes.json'
    write_nodes(tree, write_path)
    assert write_path.exists()

    nodes = read_nodes(write_path)
    assert nodes["root"].rule.value == tree.nodes["root"].rule.value
    assert nodes["root"].children[0].rule.value == tree.nodes["root"].children[0].rule.value
    