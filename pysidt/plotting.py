import os

from IPython.display import Image, display
import pydot


def plot_tree(sidt, images=True):
    graph = pydot.Dot("treestruct", graph_type="digraph", overlap="false")
    graph.set_fontname("sans")
    graph.set_fontsize("10")
    if not os.path.exists("./tree"):
        os.makedirs("./tree")
    for name, node in sidt.nodes.items():
        n = pydot.Node(name=name, label=name, fontname="Helvetica", fontsize="16")
        if images:
            img = node.group.draw("png")
            with open("./tree/" + node.name + ".png", "wb") as f:
                f.write(img)
            n.set_image(os.path.abspath("./tree/" + node.name + ".png"))
            n.set_label(" ")
        graph.add_node(n)
    for name, node in sidt.nodes.items():
        for nod in node.children:
            edge = pydot.Edge(node.name, nod.name)
            graph.add_edge(edge)
    graph.write_dot("./tree/tree.dot")
    graph.write_png("./tree/tree.png")
    plt = Image("./tree/tree.png")
    display(plt)
