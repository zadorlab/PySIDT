from molecule.molecule import Group
from molecule.quantity import ScalarQuantity
from molecule.kinetics.uncertainties import RateUncertainty
from pysidt.extensions import split_mols, get_extension_edge
from pysidt.regularization import simple_regularization
from pysidt.decomposition import *
from pysidt.utils import *
import numpy as np
import logging
import json
from sklearn import linear_model
import scipy.sparse as sp
import scipy

logging.basicConfig(level=logging.INFO)


class Rule:
    def __init__(self, value=None, uncertainty=None, num_data=None):
        self.value = value
        self.uncertainty = uncertainty 
        self.num_data = num_data

    def __repr__(self) -> str:
        return f"{self.value} +|- {np.sqrt(self.uncertainty)} (N={self.num_data})"


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
        return f"Node(name={self.name}, rule={self.rule}, depth={self.depth})"


class Datum:
    """
    Data for training tree
    mol is a possibly labeled Molecule object
    value can be in any format so long as the rule generation process can handle it
    """

    def __init__(self, mol, value, weight=1.0, uncertainty=0.0) -> None:
        self.mol = mol
        self.value = value
        self.weight = weight
        self.uncertainty = uncertainty

    def __repr__(self) -> str:
        return f"{self.mol.smiles} {self.value}"


class SubgraphIsomorphicDecisionTree:
    def __init__(
        self,
        root_group=None,
        nodes=None,
        initial_root_splits=None,
        n_strucs_min=1,
        iter_max=2,
        iter_item_cap=100,
        r=None,
        r_bonds=None,
        r_un=None,
        r_site=None,
        r_morph=None,
        uncertainty_prepruning=False,
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
        self.uncertainty_prepruning = uncertainty_prepruning

        if len(nodes) > 0:
            node = nodes[list(nodes.keys())[0]]
            while node.parent:
                node = node.parent
            self.root = node
        elif root_group:
            self.root = Node(root_group, name="Root", depth=0)
            self.nodes = {"Root": self.root}
            if initial_root_splits:
                for i,grp in enumerate(initial_root_splits):
                    name = "Root_"+str(i)
                    n = Node(grp,name=name,depth=1,parent=self.root)
                    self.root.children.append(n)
                    self.nodes[name] = n

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
            if len(node.items) <= 1 or node.name in self.skip_nodes:
                continue

            if self.uncertainty_prepruning and is_prepruned_by_uncertainty(node):
                continue

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
            if len(self.nodes) > 1:
                self.descend_training_from_top()

        node = self.select_node()

        while node is not None:
            self.extend_tree_from_node(node)
            node = self.select_node()

    def fit_tree(self, data=None, confidence_level=0.95):
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
            

            node_data = [d.value for d in node.items]
            n = len(node_data)
            wsum = sum(d.weight for d in node.items)
            wsq_sum = sum(d.weight**2 for d in node.items)
            data_mean = sum(d.value * d.weight for d in node.items) / wsum
            data_var = sum(d.weight*(d.value - data_mean)**2 for d in node.items)/(wsum - wsq_sum/wsum)
            
            if n == 1:
                node.rule = Rule(value=data_mean, uncertainty=None, num_data=n)
            else:    
                node.rule = Rule(value=data_mean, uncertainty=data_var, num_data=n)
        
        for node in self.nodes.values():
            if node.rule.uncertainty is None:
                node.rule.uncertainty = node.parent.rule.uncertainty

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


def to_dict(obj):
    out_dict = dict()
    out_dict["class"] = obj.__class__.__name__
    attrs = [attr for attr in dir(obj) if not attr.startswith("_")]
    for attr in attrs:
        val = getattr(obj, attr)

        if callable(val) or val == getattr(obj.__class__(), attr):
            continue

        try:
            json.dumps(val)
            out_dict[attr] = val
        except:
            if isinstance(val, ScalarQuantity):
                out_dict[attr] = {
                    "class": val.__class__.__name__,
                    "value": val.value,
                    "units": val.units,
                    "uncertainty": val.uncertainty,
                    "uncertainty_type": val.uncertainty_type,
                }

            elif isinstance(val, RateUncertainty):
                out_dict[attr] = {
                    "class": val.__class__.__name__,
                    "Tref": val.Tref,
                    "correlation": val.correlation,
                    "mu": val.mu,
                    "var": val.var,
                    "N": val.N,
                    "data_mean": val.data_mean,
                }

            else:
                out_dict[attr] = to_dict(val)

    return out_dict


def from_dict(d, class_dict=None):
    """construct objects from dictionary
    
    Args:
        d (dict): dictionary describing object, particularly containing a value
                associated with "class" identifying a string of the class of the object
        class_dict (dict, optional): dictionary mapping class strings to the class objects/constructors

    Returns:
        object associated with dictionary
    """
    if class_dict is None:
        class_dict = globals()

    construct_d = dict()
    for k, v in d.items():
        if k == "class":
            continue
        if isinstance(v, dict) and "class" in v.keys():
            construct_d[k] = from_dict(v, class_dict=class_dict)
        else:
            construct_d[k] = v

    return class_dict[d["class"]](**construct_d)


def write_nodes(tree, file):
    nodesdict = dict()
    for node in tree.nodes.values():
        if node.parent is None:
            p = None
        else:
            p = node.parent.name

        try:
            json.dumps(node.rule)
            rule = node.rule
        except (TypeError, OverflowError):
            rule = to_dict(
                node.rule
            )  # will work on all rmgmolecule objects, new objects need this method implemented
            try:
                json.dumps(rule)
            except:
                raise ValueError(
                    f"Could not serialize object {node.rule.__class__.__name__}"
                )

        nodesdict[node.name] = {
            "group": node.group.to_adjacency_list(),
            "rule": rule,
            "parent": p,
            "children": [x.name for x in node.children],
            "name": node.name,
            "depth": node.depth,
        }

    with open(file, "w") as f:
        json.dump(nodesdict, f)


def read_nodes(file, class_dict=None):
    """_summary_

    Args:
        file (string): string of JSON file to laod
        class_dict (dict): maps class names to classes for any non-JSON
                    serializable types that need constructed

    Returns:
        nodes (list): list of nodes for tree
    """
    if class_dict is None:
        class_dict = globals()

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
        if isinstance(node.rule, dict) and "class" in node.rule.keys():
            node.rule = from_dict(node.rule, class_dict=class_dict)
        else:
            node.rule = from_dict({"class": "Rule", "value": node.rule})

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
        initial_root_splits=None,
        n_strucs_min=1,
        iter_max=2,
        iter_item_cap=100,
        fract_nodes_expand_per_iter=0,
        r=None,
        r_bonds=None,
        r_un=None,
        r_site=None,
        r_morph=None,
        uncertainty_prepruning=False,
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
            initial_root_splits=initial_root_splits,
            n_strucs_min=n_strucs_min,
            iter_max=iter_max,
            iter_item_cap=iter_item_cap,
            r=r,
            r_bonds=r_bonds,
            r_un=r_un,
            r_site=r_site,
            r_morph=r_morph,
            uncertainty_prepruning=uncertainty_prepruning,
        )

        self.fract_nodes_expand_per_iter = fract_nodes_expand_per_iter
        self.decomposition = decomposition
        self.mol_submol_node_maps = None
        self.data_delta = None
        self.datums = None
        self.validation_set = None
        self.best_tree_nodes = None
        self.best_rule_map = None
        self.min_val_error = np.inf
        self.assign_depths()
        self.W = None # weight matrix for weighted least squares
        self.weights = None #weight list for weighted least squares

    def select_nodes(self, num=1):
        """
        Picks the nodes with the largest magintude rule values
        """
        if self.uncertainty_prepruning:
            selectable_nodes = [
                node for node in self.nodes.values() if not is_prepruned_by_uncertainty(node)
            ]
        else:
            selectable_nodes = list(self.nodes.values())

        if len(selectable_nodes) > num:
            rulevals = [
                self.node_uncertainties[node.name]
                if len(node.items) > 1
                and not (node.name in self.new_nodes)
                and not (node.name in self.skip_nodes)
                else 0.0
                for node in selectable_nodes
            ]
            inds = np.argsort(rulevals)
            maxinds = inds[-num:]
            nodes = [
                node
                for i, node in enumerate(selectable_nodes)
                if i in maxinds and len(node.items) > 1
            ]
            return nodes
        else:
            return selectable_nodes

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

        weights = np.array([datum.weight for datum in self.datums])
        if all(w == 1 for w in weights):
            self.W = None
        else:
            weights /= weights.sum()
            self.weights = weights
            W = sp.csc_matrix(np.diag(weights))
            self.W = W

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
        if len(self.nodes) > 1:
            self.descend_training_from_top(only_specific_match=True)
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
            for n,node in self.nodes.items():
                node.rule = self.best_rule_map[n]

    def fit_tree(self, data=None, check_data=True, alpha=0.1, confidence_level=0.95):
        """
        fit rule for each node
        """
        if data:
            self.setup_data(data, check_data=check_data)
            self.descend_training_from_top(only_specific_match=True)

        self.fit_rule(alpha=alpha)

        self.estimate_uncertainty(confidence_level=confidence_level)

    def fit_rule(self, alpha=0.1):
        max_depth = max([node.depth for node in self.nodes.values()])
        y = np.array([datum.value for datum in self.datums])
        preds = np.zeros(len(self.datums))
        self.node_uncertainties = dict()
        weights = self.weights
        W = self.W

        W = self.W

        for depth in range(max_depth + 1):
            nodes = [node for node in self.nodes.values() if node.depth == depth]

            # generate matrix
            A = sp.lil_matrix((len(self.datums), len(nodes)))
            y -= preds

            for i, datum in enumerate(self.datums):
                for node in self.mol_node_maps[datum]["nodes"]:
                    while node is not None:
                        if node in nodes:
                            j = nodes.index(node)
                            A[i, j] += 1.0
                        node = node.parent

            clf = linear_model.Lasso(
                alpha=alpha,
                fit_intercept=False,
                tol=1e-4,
                max_iter=1000000000,
                selection="random",
            )
            if weights is not None:
                lasso = clf.fit(A, y, sample_weight=weights)
            else:
                lasso = clf.fit(A, y)
            
            preds = A * clf.coef_
            self.data_delta = preds - y

            for i, val in enumerate(clf.coef_):
                nodes[i].rule = Rule(value=val, num_data=np.sum(A[:, i]))

        train_error = [self.evaluate(d.mol, estimate_uncertainty=False) - d.value for d in self.datums]

        logging.info("training MAE: {}".format(np.mean(np.abs(np.array(train_error)))))

        if self.validation_set:
            val_error = [self.evaluate(d.mol, estimate_uncertainty=False) - d.value for d in self.validation_set]
            val_mae = np.mean(np.abs(np.array(val_error)))
            if val_mae < self.min_val_error:
                self.min_val_error = val_mae
                self.best_tree_nodes = list(self.nodes.keys())
                self.bestA = A
                self.best_nodes = {k: v for k, v in self.nodes.items()}
                self.best_mol_node_maps = {
                    k: {"mols": v["mols"][:], "nodes": v["nodes"][:]}
                    for k, v in self.mol_node_maps.items()
                }
                self.best_rule_map = {name:self.nodes[name].rule for name in self.best_tree_nodes}
            self.val_mae = val_mae
            logging.info("validation MAE: {}".format(self.val_mae))

        logging.info("# nodes: {}".format(len(self.nodes)))

    def estimate_uncertainty(self, confidence_level=0.95):
        nodes = [node for node in self.nodes.values()]

        weights = self.weights
        W = self.W

        # generate matrix
        A = sp.csc_matrix((len(self.datums), len(nodes)))
        y = np.array([datum.value for datum in self.datums])
        preds = np.zeros(len(self.datums))

        for i, datum in enumerate(self.datums):
            for node in self.mol_node_maps[datum]["nodes"]:
                while node is not None:
                    if node in nodes:
                        j = nodes.index(node)
                        A[i, j] += 1.0
                        preds[i] += node.rule.value
                    node = node.parent

        self.data_delta = preds - y

        if A.shape[1] != 1 and W is not None:
            node_uncertainties = (
                np.diag(np.linalg.pinv((A.T @ W @ A).toarray()))
                * (self.data_delta**2).sum()
                / ((len(self.datums) - len(nodes)))
            )
            self.node_uncertainties.update(
                {node.name: node_uncertainties[i] for i, node in enumerate(nodes)}
            )
        elif A.shape[1] != 1:
            node_uncertainties = (
                np.diag(np.linalg.pinv((A.T @ A).toarray()))
                * (self.data_delta**2).sum()
                / ((len(self.datums) - len(nodes)))
            )
            self.node_uncertainties.update(
                {node.name: node_uncertainties[i] for i, node in enumerate(nodes)}
            )
        else:
            self.node_uncertainties.update(
                {node.name: 1.0 for i, node in enumerate(nodes)}
            )
        
        for node in self.nodes.values():
            node.rule.uncertainty = self.node_uncertainties[node.name]
            
        for node in self.nodes.values():
            if node.rule.uncertainty is None:
                node.rule.uncertainty = node.parent.rule.uncertainty


    def assign_depths(self):
        root = self.root
        _assign_depths(root)

    def evaluate(self, mol, estimate_uncertainty=False):
        """
        Evaluate tree for a given possibly labeled mol
        """
        pred = 0.0
        unc = 0.0
        decomp = self.decomposition(mol)
        for d in decomp:
            children = self.root.children
            node = self.root
            pred += node.rule.value
            if estimate_uncertainty:
                unc += node.rule.uncertainty
            boo = True
            while boo:
                for child in children:
                    if d.is_subgraph_isomorphic(
                        child.group, generate_initial_map=True, save_order=True
                    ):
                        children = child.children
                        node = child
                        pred += node.rule.value
                        if estimate_uncertainty:
                            unc += node.rule.uncertainty
                        break
                else:
                    boo = False

        if estimate_uncertainty:
            return pred, np.sqrt(unc)
        else:
            return pred

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


def _assign_depths(node, depth=0):
    node.depth = depth
    for child in node.children:
        _assign_depths(child, depth=depth + 1)


def is_prepruned_by_uncertainty(node):
    return node.rule.uncertainty <= min(item.uncertainty for item in node.items)