from molecule.molecule import Group
from pysidt.extensions import split_mols, get_extension_edge
from pysidt.regularization import simple_regularization
from pysidt.decomposition import *
from pysidt.utils import *
import numpy as np
import logging
import json
from sklearn import linear_model
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)


class Node:
    def __init__(
        self,
        group=None,
        items=None,
        rule=None,
        parent=None,
        children=None,
        name=None,
        depth=None,
    ):
        if items is None:
            items = []
        if children is None:
            children = []

        self.group = group
        self.items = items  # list of Datum objects
        self.rule = rule
        self.parent = parent
        self.children = children
        self.name = name
        self.depth = depth

    def __repr__(self) -> str:
        return f"{self.name} rule: {self.rule} depth: {self.depth}"


class Datum:
    """
    Data for training tree
    mol is a possibly labeled Molecule object
    value can be in any format so long as the rule generation process can handle it
    """

    def __init__(self, mol, value):
        self.mol = mol
        self.value = value

    def __repr__(self) -> str:
        return f"{self.mol.smiles} {self.value}"


class SubgraphIsomorphicDecisionTree:
    def __init__(
        self,
        root_group=None,
        nodes=None,
        n_strucs_min=1,
        iter_max=2,
        iter_item_cap=100,
        r=None,
        r_bonds=None,
        r_un=None,
        r_site=None,
        r_morph=None,
    ):
        if nodes is None:
            nodes = {}
        if r_bonds is None:
            r_bonds = [1, 2, 3, 1.5, 4]
        if r_un is None:
            r_un = [0, 1, 2, 3]
        if r_site is None:
            r_site = []
        if r_morph is None:
            r_morph = []

        self.nodes = nodes
        self.n_strucs_min = n_strucs_min
        self.iter_max = iter_max
        self.iter_item_cap = iter_item_cap
        self.r = r
        self.r_bonds = r_bonds
        self.r_un = r_un
        self.r_site = r_site
        self.r_morph = r_morph
        self.skip_nodes = []

        if len(nodes) > 0:
            node = nodes[list(nodes.keys())[0]]
            while node.parent:
                node = node.parent
            self.root = node
        elif root_group:
            self.root = Node(root_group, name="Root")
            self.nodes = {"Root": self.root}

    def load(self, nodes):
        self.nodes = nodes

        if len(nodes) > 0:
            node = nodes[list(nodes.keys())[0]]
            while node.parent:
                node = node.parent
            self.root = node
        else:
            self.root = None

    def select_node(self):
        """
        Picks a node to expand
        """
        for name, node in self.nodes.items():
            if len(node.items) > 1 and not (node.name in self.skip_nodes):
                logging.info("Selected node {}".format(node.name))
                logging.info("Node has {} items".format(len(node.items)))
                return node
        else:
            return None

    def generate_extensions(self, node, recursing=False):
        """
        Generates set of extension groups to a node
        returns list of Groups
        design not to subclass
        """

        out, gave_up_split = get_extension_edge(
            node,
            self.n_strucs_min,
            r=self.r,
            r_bonds=self.r_bonds,
            r_un=self.r_un,
            r_site=self.r_site,
            r_morph=self.r_morph,
            iter_max=self.iter_max,
            iter_item_cap=self.iter_item_cap,
        )

        if not out and not recursing:
            logging.info("recursing")
            logging.info(node.group.to_adjacency_list())
            node.group.clear_reg_dims()
            return self.generate_extensions(node, recursing=True)

        return out  # [(grp2, grpc, name, typ, indc)]

    def choose_extension(self, node, exts):
        """
        select best extension among the set of extensions
        returns a Node object
        almost always subclassed
        """
        minval = np.inf
        minext = None
        for ext in exts:
            new, comp = split_mols(node.items, ext)
            val = np.std([x.value for x in new]) * len(new) + np.std(
                [x.value for x in comp]
            ) * len(comp)
            if val < minval:
                minval = val
                minext = ext

        return minext

    def extend_tree_from_node(self, parent):
        """
        Adds a new node to the tree
        """
        exts = self.generate_extensions(parent)
        extlist = [ext[0] for ext in exts]
        if not extlist:
            self.skip_nodes.append(parent.name)
            return
        ext = self.choose_extension(parent, extlist)
        new, comp = split_mols(parent.items, ext)
        ind = extlist.index(ext)
        grp, grpc, name, typ, indc = exts[ind]
        logging.info("Choose extension {}".format(name))

        node = Node(
            group=grp,
            items=new,
            rule=None,
            parent=parent,
            children=[],
            name=name,
            depth=parent.depth + 1,
        )
        self.nodes[name] = node
        parent.children.append(node)
        if grpc:
            frags = name.split("_")
            frags[-1] = "N-" + frags[-1]
            cextname = ""
            for k in frags:
                cextname += k
                cextname += "_"
            cextname = cextname[:-1]
            nodec = Node(
                group=grpc,
                items=comp,
                rule=None,
                parent=parent,
                children=[],
                name=cextname,
                depth=parent.depth + 1,
            )
            self.nodes[cextname] = nodec
            parent.children.append(nodec)
            parent.items = []
        else:
            for mol in new:
                parent.items.remove(mol)

    def descend_training_from_top(self, only_specific_match=True):
        """
        Moves training data as needed down the tree
        """
        nodes = [self.root]

        while nodes != []:
            new_nodes = []
            for node in nodes:
                self.descend_node(node, only_specific_match=only_specific_match)
                new_nodes += node.children
            nodes = new_nodes

    def descend_node(self, node, only_specific_match=True):
        for child in node.children:
            data_to_add = []
            for datum in node.items:
                if datum.mol.is_subgraph_isomorphic(
                    child.group, generate_initial_map=True, save_order=True
                ):
                    data_to_add.append(datum)

            for datum in data_to_add:
                child.items.append(datum)
                if only_specific_match:
                    node.items.remove(datum)

    def clear_data(self):
        for node in self.nodes.values():
            node.items = []

    def generate_tree(self, data=None, check_data=True):
        """
        generate nodes for the tree based on the supplied data
        """
        self.skip_nodes = []
        if data:
            if check_data:
                for datum in data:
                    if not datum.mol.is_subgraph_isomorphic(
                        self.root.group, generate_initial_map=True, save_order=True
                    ):
                        logging.info("Datum did not match Root node:")
                        logging.info(datum.mol.to_adjacency_list())
                        raise ValueError

            self.clear_data()
            self.root.items = data[:]

        node = self.select_node()

        while node is not None:
            self.extend_tree_from_node(node)
            node = self.select_node()

    def fit_tree(self, data=None):
        """
        fit rule for each node
        """
        if data:
            self.clear_data()
            self.root.items = data[:]
            self.descend_training_from_top(only_specific_match=False)

        for node in self.nodes.values():
            if not node.items:
                logging.info(node.name)
                raise ValueError
            node.rule = sum([d.value for d in node.items]) / len(node.items)

    def evaluate(self, mol):
        """
        Evaluate tree for a given possibly labeled mol
        """
        children = self.root.children
        node = self.root

        while children:
            for child in children:
                if mol.is_subgraph_isomorphic(
                    child.group, generate_initial_map=True, save_order=True
                ):
                    children = child.children
                    node = child
                    break
            else:
                return node.rule

        return node.rule


def write_nodes(tree, file):
    nodesdict = dict()
    for node in tree.nodes.values():
        if node.parent is None:
            p = None
        else:
            p = node.parent.name
        nodesdict[node.name] = {
            "group": node.group.to_adjacency_list(),
            "rule": node.rule,
            "parent": p,
            "children": [x.name for x in node.children],
            "name": node.name,
        }

    with open(file, "w") as f:
        json.dump(nodesdict, f)


def read_nodes(file):
    with open(file, "r") as f:
        nodesdict = json.load(f)
    nodes = dict()
    for n, d in nodesdict.items():
        nodes[n] = Node(
            group=Group().from_adjacency_list(d["group"]),
            rule=d["rule"],
            parent=d["parent"],
            children=d["children"],
            name=d["name"],
            depth=d["depth"],
        )

    for n, node in nodes.items():
        if node.parent:
            node.parent = nodes[node.parent]
        node.children = [nodes[child] for child in node.children]

    return nodes


class MultiEvalSubgraphIsomorphicDecisionTree(SubgraphIsomorphicDecisionTree):
    """
    Makes prediction for a molecule based on multiple evaluations.

    Args:
        `decomposition`: method to decompose a molecule into substructure contributions.
        `root_group`: root group for the tree
        `nodes`: dictionary of nodes for the tree
        `n_strucs_min`: minimum number of disconnected structures that can be in the group. Default is 1.
        `iter_max`: maximum number of times the extension generation algorithm is allowed to expand structures looking for additional splits. Default is 2.
        `iter_item_cap`: maximum number of structures the extension generation algorithm can send for expansion. Default is 100.
        `fract_nodes_expand_per_iter`: fraction of nodes to split at each iteration. If 0, only 1 node will be split at each iteration.
        `r`: atom types to generate extensions. If None, all atom types will be used.
        `r_bonds`: bond types to generate extensions. If None, [1, 2, 3, 1.5, 4] will be used.
        `r_un`: unpaired electrons to generate extensions. If None, [0, 1, 2, 3] will be used.
        `r_site`: surface sites to generate extensions. If None, [] will be used.
        `r_morph`: surface morphology to generate extensions. If None, [] will be used.
    """

    def __init__(
        self,
        decomposition,
        root_group=None,
        nodes=None,
        n_strucs_min=1,
        iter_max=2,
        iter_item_cap=100,
        fract_nodes_expand_per_iter=0,
        r=None,
        r_bonds=None,
        r_un=None,
        r_site=None,
        r_morph=None,
    ):
        if nodes is None:
            nodes = dict()
        if r_bonds is None:
            r_bonds = [1, 2, 3, 1.5, 4]
        if r_un is None:
            r_un = [0, 1, 2, 3]
        if r_site is None:
            r_site = []
        if r_morph is None:
            r_morph = []

        super().__init__(
            root_group=root_group,
            nodes=nodes,
            n_strucs_min=n_strucs_min,
            iter_max=iter_max,
            iter_item_cap=iter_item_cap,
            r=r,
            r_bonds=r_bonds,
            r_un=r_un,
            r_site=r_site,
            r_morph=r_morph,
        )

        self.fract_nodes_expand_per_iter = fract_nodes_expand_per_iter
        self.decomposition = decomposition
        self.mol_submol_node_maps = None
        self.data_delta = None
        self.datums = None
        self.validation_set = None
        self.best_tree_nodes = None
        self.min_val_error = np.inf
        self.assign_depths()

    def select_nodes(self, num=1):
        """
        Picks the nodes with the largest magintude rule values
        """
        if len(self.nodes) > num:
            rulevals = [
                self.node_uncertainties[node.name]
                if len(node.items) > 1
                and not (node.name in self.new_nodes)
                and not (node.name in self.skip_nodes)
                else 0.0
                for node in self.nodes.values()
            ]
            inds = np.argsort(rulevals)
            maxinds = inds[-num:]
            nodes = [
                node
                for i, node in enumerate(self.nodes.values())
                if i in maxinds and len(node.items) > 1
            ]
            return nodes
        else:
            return list(self.nodes.values())

    def extend_tree_from_node(self, parent):
        """
        Adds a new node to the tree
        """
        exts = self.generate_extensions(parent)
        extlist = [ext[0] for ext in exts]
        if not extlist:
            self.skip_nodes.append(parent.name)
            return
        ext = self.choose_extension(parent, extlist)
        if ext is None:
            self.skip_nodes.append(parent.name)
            return
        new, comp = split_mols(parent.items, ext)
        ind = extlist.index(ext)
        grp, grpc, name, typ, indc = exts[ind]

        node = Node(
            group=grp,
            items=new,
            rule=None,
            parent=parent,
            children=[],
            name=name,
            depth=parent.depth + 1,
        )

        assert not (name in self.nodes.keys()), name

        self.nodes[name] = node
        parent.children.append(node)
        self.node_uncertainties[name] = self.node_uncertainties[parent.name]
        self.new_nodes.append(name)

        for k, datum in enumerate(self.datums):
            for i, d in enumerate(self.mol_node_maps[datum]["mols"]):
                if any(d is x for x in new):
                    assert d.is_subgraph_isomorphic(
                        node.group, generate_initial_map=True, save_order=True
                    )
                    self.mol_node_maps[datum]["nodes"][i] = node

        print("adding node {}".format(name))

        if grpc:
            frags = name.split("_")
            frags[-1] = "N-" + frags[-1]
            cextname = ""
            for k in frags:
                cextname += k
                cextname += "_"
            cextname = cextname[:-1]
            nodec = Node(
                group=grpc,
                items=comp,
                rule=None,
                parent=parent,
                children=[],
                name=cextname,
                depth=parent.depth + 1,
            )

            self.nodes[cextname] = nodec
            parent.children.append(nodec)
            self.node_uncertainties[cextname] = self.node_uncertainties[parent.name]
            self.new_nodes.append(cextname)

            for k, datum in enumerate(self.datums):
                for i, d in enumerate(self.mol_node_maps[datum]["mols"]):
                    if any(d is x for x in comp):
                        assert d.is_subgraph_isomorphic(
                            nodec.group, generate_initial_map=True, save_order=True
                        )
                        self.mol_node_maps[datum]["nodes"][i] = nodec

            parent.items = []
        else:
            parent.items = comp

    def choose_extension(self, node, exts):
        """
        select best extension among the set of extensions
        returns a Node object
        almost always subclassed
        """
        maxval = 0.0
        maxext = None
        for ext in exts:
            new, comp = split_mols(node.items, ext)
            newval = 0.0
            compval = 0.0
            for i, datum in enumerate(self.datums):
                dy = self.data_delta[i] / len(self.mol_node_maps[datum]["mols"])
                for j, d in enumerate(self.mol_node_maps[datum]["mols"]):
                    v = self.node_uncertainties[
                        self.mol_node_maps[datum]["nodes"][j].name
                    ]
                    s = sum(
                        self.node_uncertainties[
                            self.mol_node_maps[datum]["nodes"][k].name
                        ]
                        for k in range(len(self.mol_node_maps[datum]["nodes"]))
                    )
                    if any(d is x for x in new):
                        newval += self.data_delta[i] * v / s
                    elif any(d is x for x in comp):
                        compval += self.data_delta[i] * v / s
            val = abs(newval - compval)
            if val > maxval:
                maxval = val
                maxext = ext

        return maxext

    def check_mol_node_maps(self):
        for d, v in self.mol_node_maps.items():
            for i, m in enumerate(v["mols"]):
                nv = self.nodes["Root"]
                assert m.is_subgraph_isomorphic(
                    nv.group, generate_initial_map=True, save_order=True
                )
                children = nv.children
                boo = True
                while boo:
                    for child in children:
                        if m.is_subgraph_isomorphic(
                            child.group, generate_initial_map=True, save_order=True
                        ):
                            children = child.children
                            nv = child
                            break
                    else:
                        boo = False
                assert nv == v["nodes"][i]

    def setup_data(self, data, check_data=False):
        self.datums = data
        self.mol_node_maps = dict()
        for datum in self.datums:
            decomp = self.decomposition(datum.mol)
            self.mol_node_maps[datum] = {
                "mols": decomp,
                "nodes": [self.root for d in decomp],
            }

        if check_data:
            for i, datum in enumerate(self.datums):
                for d in self.mol_node_maps[datum]["mols"]:
                    if not d.is_subgraph_isomorphic(
                        self.root.group, generate_initial_map=True, save_order=True
                    ):
                        logging.info("Datum Submol did not match Root node:")
                        logging.info(d.to_adjacency_list())
                        raise ValueError

        self.clear_data()
        out = []
        for datum in self.datums:
            for d in self.mol_node_maps[datum]["mols"]:
                out.append(d)

        self.root.items = out

    def generate_tree(
        self,
        data=None,
        check_data=True,
        validation_set=None,
        max_nodes=None,
        postpruning_based_on_val=True,
        alpha=0.1,
    ):
        """
        generate nodes for the tree based on the supplied data

        Args:
            `data`: list of Datum objects to train the tree
            `check_data`: if True, check that the data is subgraph isomorphic to the root group
            `validation_set`: list of Datum objects to validate the tree
            `max_nodes`: maximum number of nodes to generate
            `postpruning_based_on_val`: if True, regularize the tree based on the validation set
            `alpha`: regularization parameter for Lasso regression
        """
        self.setup_data(data, check_data=check_data)
        self.val_mae = np.inf
        self.skip_nodes = []
        self.new_nodes = []

        self.validation_set = validation_set

        while True:
            self.fit_tree(alpha=alpha)
            if len(self.nodes) > max_nodes:
                break
            self.new_nodes = []
            num = int(
                max(1, np.round(self.fract_nodes_expand_per_iter * len(self.nodes)))
            )
            nodes = self.select_nodes(num=num)
            if not nodes:
                break
            else:
                for node in nodes:
                    self.extend_tree_from_node(node)

        if self.validation_set and postpruning_based_on_val:
            logging.info("Postpruning based on best validation error")
            nodes_to_remove = []
            for k in list(self.nodes.keys()):
                if k not in self.best_tree_nodes:
                    nodes_to_remove.append(k)

            node_back_mapping = dict()
            for k in nodes_to_remove:
                parent = self.nodes[k]
                while parent.name in nodes_to_remove:
                    parent = parent.parent
                node_back_mapping[self.nodes[k]] = parent
                parent.items.extend(self.nodes[k].items)
                del self.nodes[k]

            for k, datum in enumerate(self.datums):
                for i, n in enumerate(self.mol_node_maps[datum]["nodes"]):
                    if n in node_back_mapping.keys():
                        self.mol_node_maps[datum]["nodes"][i] = node_back_mapping[n]

            for node in self.nodes.values():
                children_to_remove = []
                for child in node.children:
                    if child not in self.nodes.values():
                        children_to_remove.append(child)
                for child in children_to_remove:
                    node.children.remove(child)

            self.fit_tree(data=None, check_data=False, alpha=alpha)

    def fit_tree(self, data=None, check_data=True, alpha=0.1):
        """
        fit rule for each node
        """
        if data:
            self.setup_data(data, check_data=check_data)
            self.descend_training_from_top(only_specific_match=True)

        # generate matrix
        A = sp.csc_matrix((len(self.datums), len(self.nodes)))
        y = np.array([datum.value for datum in self.datums])

        nodes = list(self.nodes.values())
        for i, datum in enumerate(self.datums):
            for node in self.mol_node_maps[datum]["nodes"]:
                while node is not None:
                    try:
                        j = nodes.index(node)
                    except Exception as e:
                        logging.info(node.name)
                        raise e
                    A[i, j] += 1.0
                    node = node.parent

        clf = linear_model.Lasso(
            alpha=alpha,
            fit_intercept=False,
            tol=1e-4,
            max_iter=1000000000,
            selection="random",
        )

        pred = clf.fit(A, y)
        self.data_delta = A * clf.coef_ - y

        if A.shape[1] != 1:
            node_uncertainties = (
                np.diag(np.linalg.pinv((A.T @ A).toarray()))
                * (self.data_delta**2).sum()
                / (len(self.datums) - len(self.nodes))
            )
            self.node_uncertainties = {
                node.name: node_uncertainties[i] for i, node in enumerate(nodes)
            }
        else:
            self.node_uncertainties = {node.name: 1.0 for i, node in enumerate(nodes)}

        for i, val in enumerate(clf.coef_):
            nodes[i].rule = val

        train_error = [self.evaluate(d.mol) - d.value for d in self.datums]

        logging.info("training MAE: {}".format(np.mean(np.abs(np.array(train_error)))))

        if self.validation_set:
            train_mae = np.mean(np.abs(np.array(train_error)))
            val_error = [self.evaluate(d.mol) - d.value for d in self.validation_set]
            val_mae = np.mean(np.abs(np.array(val_error)))
            max_mae = max(val_mae, train_mae)
            if max_mae < self.min_val_error:
                self.min_val_error = max_mae
                self.best_tree_nodes = list(self.nodes.keys())
                self.check_mol_node_maps()
                self.bestA = A
                self.best_nodes = {k: v for k, v in self.nodes.items()}
                self.best_mol_node_maps = {
                    k: {"mols": v["mols"][:], "nodes": v["nodes"][:]}
                    for k, v in self.mol_node_maps.items()
                }
            self.val_mae = val_mae
            logging.info("validation MAE: {}".format(self.val_mae))

        logging.info("# nodes: {}".format(len(self.nodes)))

    def evaluate(self, mol):
        """
        Evaluate tree for a given possibly labeled mol
        """
        out = 0.0
        decomp = self.decomposition(mol)
        for d in decomp:
            children = self.root.children
            node = self.root
            out += node.rule
            boo = True
            while boo:
                for child in children:
                    if d.is_subgraph_isomorphic(
                        child.group, generate_initial_map=True, save_order=True
                    ):
                        children = child.children
                        node = child
                        out += node.rule
                        break
                else:
                    boo = False

        return out

    def descend_node(self, node, only_specific_match=True):
        data_to_add = {child: [] for child in node.children}
        for m in node.items:
            for child in node.children:
                if m.is_subgraph_isomorphic(
                    child.group, generate_initial_map=True, save_order=True
                ):
                    data_to_add[child].append(m)
                    break

        for k, datum in enumerate(self.datums):
            for i, d in enumerate(self.mol_node_maps[datum]["mols"]):
                for child in node.children:
                    if any(d is x for x in data_to_add[child]):
                        self.mol_node_maps[datum]["nodes"][i] = child

        for child in node.children:
            for m in data_to_add[child]:
                child.items.append(m)
                if only_specific_match:
                    node.items.remove(m)

    def regularize(self, data=None, check_data=True):
        if data:
            self.setup_data(data, check_data=check_data)
            self.descend_training_from_top(only_specific_match=False)

        simple_regularization(
            self.nodes["Root"],
            self.r,
            self.r_bonds,
            self.r_un,
            self.r_site,
            self.r_morph,
        )
