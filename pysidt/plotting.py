from IPython.display import Image, display
import pydot
import os
import numpy as np

def plot_tree(sidt, images=True, depth=np.inf):
    graph = pydot.Dot("treestruct", graph_type="digraph", overlap="false")
    graph.set_fontname("sans")
    graph.set_fontsize("10")
    if not os.path.exists("./tree"):
        os.makedirs("./tree")
    out_nodes = dict()
    for name, node in sidt.nodes.items():
        if node.depth <= depth:
            n = pydot.Node(name=name, label=name, fontname="Helvetica", fontsize="16")
            if images:
                img = node.group.draw("png")
                with open("./tree/" + node.name + ".png", "wb") as f:
                    f.write(img)
                n.set_image(os.path.abspath("./tree/" + node.name + ".png"))
                n.set_label(" ")
            graph.add_node(n)
            out_nodes[name] = node
    for name, node in out_nodes.items():
        for nod in node.children:
            if nod.name in out_nodes.keys():
                edge = pydot.Edge(node.name, nod.name)
                graph.add_edge(edge)
    graph.write_dot("./tree/tree.dot")
    graph.write_png("./tree/tree.png")
    plt = Image("./tree/tree.png")
    display(plt)
