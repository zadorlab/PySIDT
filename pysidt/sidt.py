from molecule.molecule import Group
from molecule.quantity import ScalarQuantity
from molecule.kinetics.uncertainties import RateUncertainty
from molecule.molecule.atomtype import ATOMTYPES
from molecule.molecule.element import bde_elements
from pysidt.extensions import split_mols, get_extension_edge, generate_extensions_reverse
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
    def __init__(self, value=None, uncertainty=None, num_data=None, comment=""):
        self.value = value
        self.uncertainty = uncertainty 
        self.num_data = num_data
        self.comment = comment 
        
    def __repr__(self) -> str:
        return f"{self.value} +|- {np.sqrt(self.uncertainty)} (N={self.num_data}, comment={self.comment})"


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
    """
    Makes prediction for a molecule based on multiple evaluations.

    Args:
        `root_group`: root group for the tree
        `nodes`: dictionary of nodes for the tree
        `n_strucs_min`: minimum number of disconnected structures that can be in the group. Default is 1.
        `iter_max`: maximum number of times the extension generation algorithm is allowed to expand structures looking for additional splits. Default is 2.
        `iter_item_cap`: maximum number of structures the extension generation algorithm can send for expansion. Default is 100.
        `max_structures_to_generate_extensions`: maximum number of structures used in extension generation (a seeded random sample is drawn if larger than this number)
        `max_structures_to_choose_extension`: maximum number of structures used in choosing an extension (a seeded random sample is drawn if larger than this number)
        `r`: atom types to generate extensions. If None, all atom types will be used.
        `r_bonds`: bond types to generate extensions. If None, [1, 2, 3, 1.5, 4] will be used.
        `r_un`: unpaired electrons to generate extensions. If None, [0, 1, 2, 3] will be used.
        `r_site`: surface sites to generate extensions. If None, [] will be used.
        `r_morph`: surface morphology to generate extensions. If None, [] will be used.
    """
    def __init__(
        self,
        root_group=None,
        nodes=None,
        initial_root_splits=None,
        n_strucs_min=1,
        iter_max=2,
        iter_item_cap=100,
        max_structures_to_generate_extensions=400,
        max_structures_to_choose_extension=np.inf,
        max_batch_size=np.inf,
        new_fraction_threshold_to_reopt_node=0.25,
        r=None,
        r_bonds=None,
        r_un=None,
        r_site=None,
        r_morph=None,
        uncertainty_prepruning=False,
        max_nodes=np.inf,
    ):
        if nodes is None:
            nodes = {}
        if r is None:
            r = bde_elements  # set of possible r elements/atoms
            r = [ATOMTYPES[x] for x in r]
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
        self.max_batch_size = max_batch_size
        self.new_fraction_threshold_to_reopt_node = new_fraction_threshold_to_reopt_node
        self.max_nodes = max_nodes
        self.max_structures_to_generate_extensions = max_structures_to_generate_extensions
        self.max_structures_to_choose_extension = max_structures_to_choose_extension
        
        if len(nodes) > 0:
            node = nodes[list(nodes.keys())[0]]
            while node.parent:
                node = node.parent
            self.root = node
        elif root_group:
            if isinstance(root_group,list):
                self.root = Node(group=None, name="Root", depth=0)
                roots = []
                for i,g in enumerate(root_group):
                    roots.append(Node(group=g, name="Root_"+str(i), parent=self.root, depth=1))
                self.nodes = {n.name:n for n in roots}
                self.nodes["Root"] = self.root
                if initial_root_splits:
                    raise ValueError("initial_root_splits not compatible with multiple root groups...construct the initial nodes manually")
            else:
                self.root = Node(group=root_group, name="Root", depth=0)
                self.nodes = {"Root": self.root}
                if initial_root_splits:
                    for i,grp in enumerate(initial_root_splits):
                        name = "Root_"+str(i)
                        n = Node(grp,name=name,depth=1,parent=self.root)
                        self.root.children.append(n)
                        self.nodes[name] = n

    def get_batches(self, data, first_batch_include=[]):
        """
        Break data up into batches

        Args:
            data (_type_): _description_
            first_batch_include (list, optional): _description_. Defaults to [].

        Returns:
            _type_: _description_
        """
        Ndata = len(data)
        shdata = [d for d in data if d not in first_batch_include]
        np.random.shuffle(shdata)
        
        batch1 = first_batch_include[:]
        assert len(batch1) < self.max_batch_size
        batch1.extend(shdata[:self.max_batch_size-len(batch1)])
        if len(batch1) < self.max_batch_size:
            return [batch1]
    
        shdata = shdata[self.max_batch_size-len(batch1):]
        batches = [batch1] 
        N = 0
        while N+self.max_batch_size < len(shdata):
            batches.append(shdata[N:N+self.max_batch_size])
            N += self.max_batch_size
        batchend = shdata[N:]
        if len(batchend) != 0:
            batches.append(batchend)
        
        logging.info("Divided {0} Data points into batches: {1}".format(Ndata,[len(x) for x in batches]))
        return batches
            
    def prune(self,newdata):
        """
        prunes tree
        also clears tree as side effect (but desirable anyway when pruning)
        Args:
            newdata (_type_): _description_
        """
        Nolds = {n:0 for n in self.nodes.keys()}
        for node in self.nodes.values():
            Nitems = len(node.items)
            n = node
            while n.parent:
                Nolds[n.name] += Nitems 
                n = n.parent 
        self.clear_data()
        self.root.items = newdata 
        self.descend_training_from_top(only_specific_match=False)
        Nnews = {n.name:len(n.items) for n in self.nodes.values}
        
        for name,node in self.nodes.items():
            if node.parent and Nnews[name]/Nolds[name] > self.new_fraction_threshold_to_reopt_node:
                node.parent.children.remove(node)
                del self.nodes[name]
            else:
                node.group.clear_reg_dims()
        
    
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
        if len(node.items) <= self.max_structures_to_generate_extensions:
            structs = node.items
            clear_reg_dims = False
        else:
            logging.info(f"Sampling {self.max_structures_to_generate_extensions} structures from {len(node.items)} structures at node {node.name}")
            structs = np.random.choice(node.items,self.max_structures_to_generate_extensions,replace=False)
            clear_reg_dims = True
            
        out, gave_up_split = get_extension_edge(
            group=node.group,
            items=structs,
            node_children=node.children,
            basename=node.name,
            n_strucs_min=self.n_strucs_min,
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

        if not out:
            logging.info("forward extension generation failed, using reverse extension generation")
            grps = generate_extensions_reverse(node.group,structs)
            name = node.name+"_Revgen"
            i = 0
            while name+str(i) in self.nodes.keys():
                i += 1
            
            return [(g,None,node.name+"_Revgen"+str(i),"Revgen",None) for g in grps if g is not None],False

        if not out:
            logging.warning(f"Failed to extend Node {node.name} with {len(node.items)} items")
            logging.warning("node")
            logging.warning(node.group.to_adjacency_list())
            logging.warning("Items:")
            for item in node.items:
                if isinstance(item, Datum):
                    logging.warning(item.value)
                    logging.warning(item.mol.to_adjacency_list())
                else:
                    logging.warning(item.to_adjacency_list())
            return [],clear_reg_dims
        
        if clear_reg_dims:
            node.group.clear_reg_dims()
            
        return out,clear_reg_dims  # [(grp2, grpc, name, typ, indc)]

    def choose_extension(self, node, exts):
        """
        select best extension among the set of extensions
        returns a Node object
        almost always subclassed
        """
        logging.info(f"Choosing from {len(exts)} extensions")
        if len(node.items) <= self.max_structures_to_choose_extension:
            structs = node.items
        else:
            logging.info(f"Sampling {self.max_structures_to_choose_extension} structures from {len(node.items)} structures at node {node.name}")
            structs = np.random.choice(node.items,self.max_structures_to_choose_extension,replace=False)
            
        minval = np.inf
        minext = None
        for ext in exts:
            new, comp = split_mols(structs, ext)
            Lnew = len(new)
            Lcomp = len(comp)
            if Lnew  > 1 and Lcomp > 1:
                val = np.std([x.value for x in new]) * Lnew  + np.std(
                [x.value for x in comp]
            ) * Lcomp
            elif Lnew  == 1 and Lcomp == 1:
                val = 0.0
            elif Lnew  == 1:
                val = np.std([x.value for x in comp]) * Lcomp
            elif Lcomp == 1:
                val = np.std([x.value for x in new]) * Lnew 
            else: #did not split?
                logging.error("group:")
                logging.error(ext.to_adjacency_list())
                logging.error("data:")
                for item in node.items:
                    logging.error(item.mol.to_adjacency_list())
                raise ValueError("Generated extension did not split items")

            if val < minval:
                minval = val
                minext = ext

        return minext

    def extend_tree_from_node(self, parent):
        """
        Adds a new node to the tree
        """
        exts,clear_reg_dims = self.generate_extensions(parent)
        extlist = [ext[0] for ext in exts]
        if not extlist:
            logging.info(f"Skipping node {parent.name}")
            self.skip_nodes.append(parent.name)
            return
        ext = self.choose_extension(parent, extlist)
        new, comp = split_mols(parent.items, ext)
        ind = extlist.index(ext)
        grp, grpc, name, typ, indc = exts[ind]
        if clear_reg_dims:
            grp.clear_reg_dims()
            grpc.clear_reg_dims()
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

    def generate_tree(self, data, check_data=True, first_batch_include=[]):
        """
        generate nodes for the tree based on the supplied data
        """
        np.random.seed(0)
        self.check_subgraph_isomorphic()

        self.skip_nodes = []

        if check_data:
            for datum in data:
                if not datum.mol.is_subgraph_isomorphic(
                    self.root.group, generate_initial_map=True, save_order=True
                ):
                    logging.info("Datum did not match Root node:")
                    logging.info(datum.mol.to_adjacency_list())
                    raise ValueError

        if self.max_batch_size > len(data):
            batches = [data]
        else:
            logging.info("using cascade algorithm")
            batches = self.get_batches(data,first_batch_include=first_batch_include)
        
        data_temp = []
        for i,batch in enumerate(batches):
            data_temp += batch
            if len(batches) > 1:
                logging.info("Starting batch {0} with {1} data points".format(i+1,len(data)))
            if i != 0:
                logging.info("pruning tree with {} nodes".format(len(self.nodes)))
                self.prune(data)
                logging.info("pruned tree down to {} nodes".format(len(self.nodes)))
            self.clear_data()
            self.root.items = data_temp[:]
            if len(self.nodes) > 1:
                self.descend_training_from_top()
            
            node = self.select_node()
            while True:
                if len(self.nodes) > self.max_nodes:
                    break
                if not node:
                    logging.info("Did not find any nodes to expand")
                    break
                else:
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
                logging.warning(f"Node: {node.name} was empty")
                node.rule = None 
                continue
            

            node_data = [d.value for d in node.items]
            n = len(node_data)
            wsum = sum(d.weight for d in node.items)
            wsq_sum = sum(d.weight**2 for d in node.items)
            if (wsum - wsq_sum/wsum) > 1e-3: 
                data_mean = sum(d.value * d.weight for d in node.items) / wsum
                data_var = sum(d.weight*(d.value - data_mean)**2 for d in node.items)/(wsum - wsq_sum/wsum)
            else: #primarily if weights are all 1.0
                data_mean = np.mean(node_data)
                data_var = np.var(node_data)
            
            if n == 1:
                node.rule = Rule(value=data_mean, uncertainty=None, num_data=n)
            else:    
                node.rule = Rule(value=data_mean, uncertainty=data_var, num_data=n)
        
        for node in self.nodes.values():
            n = node
            while n.rule is None:
                n = n.parent
            node.rule = n.rule
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

    def check_subgraph_isomorphic(self):
        for node in self.nodes.values():
            if (node.group is not None) and (node.parent is not None) and (node.parent.group is not None) and not node.group.is_subgraph_isomorphic(node.parent.group, generate_initial_map=True, save_order=True):
                raise ValueError(f"Tree is not subgraph isomorphic: {node.name} is not subgraph isomorphic to parent {node.parent.name}")

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
    else:
        class_dict.update(globals())
    
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
        if node.group is not None:
            nodesdict[node.name] = {
                "group": node.group.to_adjacency_list(),
                "rule": rule,
                "parent": p,
                "children": [x.name for x in node.children],
                "name": node.name,
                "depth": node.depth,
            }
        else:
            nodesdict[node.name] = {
                "group": None,
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
    Base class for Multi-Evaluation SIDTs
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
        max_structures_to_generate_extensions=400,
        max_structures_to_choose_extension=np.inf,
        fract_nodes_expand_per_iter=0,
        max_batch_size=np.inf,
        new_fraction_threshold_to_reopt_node=0.25,
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
            max_structures_to_generate_extensions=max_structures_to_generate_extensions,
            max_structures_to_choose_extension=max_structures_to_choose_extension,
            max_batch_size=max_batch_size,
            new_fraction_threshold_to_reopt_node=new_fraction_threshold_to_reopt_node,
            r=r,
            r_bonds=r_bonds,
            r_un=r_un,
            r_site=r_site,
            r_morph=r_morph,
            uncertainty_prepruning=uncertainty_prepruning,
        )

        if root_group and (isinstance(decomposition,list) or isinstance(root_group,list)):
            assert len(decomposition) == len(root_group)
        
        self.fract_nodes_expand_per_iter = fract_nodes_expand_per_iter
        self.decomposition = decomposition
        self.mol_submol_node_maps = None
        self.data_delta = None
        self.datums = None
        self.validation_set = None
        self.best_tree_nodes = None
        self.best_rule_map = None
        self.min_val_error = np.inf
        self.uncertainties_valid = True
        self.assign_depths()
        self.W = None # weight matrix for weighted least squares
        self.weights = None #weight list for weighted least squares
        self.validation_set = None
        self.test_set = None

    def decompose(self,struct):
        if isinstance(self.decomposition,list):
            ds = []
            for d in self.decomposition:
                ds += d(struct)
            return ds
        else:
            return self.decomposition(struct)
    
    def select_nodes(self, num=1):
        raise NotImplementedError
    
    def extend_tree_from_node(self, parent):
        raise NotImplementedError

    def choose_extension(self, node, exts):
        raise NotImplementedError
    
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
            decomp = self.decompose(datum.mol)
            self.mol_node_maps[datum] = {
                "mols": decomp,
                "nodes": [self.root for d in decomp],
            }

        if check_data:
            for i, datum in enumerate(self.datums):
                for d in self.mol_node_maps[datum]["mols"]:
                    if self.root.group:
                        if not d.is_subgraph_isomorphic(
                            self.root.group, generate_initial_map=True, save_order=True
                        ):
                            logging.info("Datum Submol did not match Root node:")
                            logging.info(d.to_adjacency_list())
                            raise ValueError
                    else:
                        for root_child in self.root.children:
                            if d.is_subgraph_isomorphic(
                                root_child.group, generate_initial_map=True, save_order=True
                            ):
                                break
                        else:
                            logging.info("Datum Submol did not match Root nodes:")
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
        data,
        check_data=True,
        validation_set=None,
        test_set=None,
        max_nodes=None,
        postpruning_based_on_val=True,
        alpha=0.1,
        first_batch_include=[],
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
        np.random.seed(0)
        self.check_subgraph_isomorphic()
        
        if self.max_batch_size > len(data):
            batches = [data]
        else:
            logging.info("using cascade algorithm")
            batches = self.get_batches(data,first_batch_include=first_batch_include)
        data = []
        for i,batch in enumerate(batches):
            data += batch
            if len(batches) > 1:
                logging.info("Starting batch {0} with {1} data points".format(i+1,len(data)))
            if i != 0:
                logging.info("pruning tree with {} nodes".format(len(self.nodes)))
                self.prune(data)
                logging.info("pruned tree down to {} nodes".format(len(self.nodes)))
                
            self.setup_data(data, check_data=check_data)
            
            if len(self.nodes) > 1:
                self.descend_training_from_top(only_specific_match=True)
            self.val_mae = np.inf
            self.skip_nodes = []
            self.new_nodes = []

            self.validation_set = validation_set
            self.test_set = test_set 
            
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
                    logging.info("Did not find any nodes to expand")
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

    def fit_tree(self, data=None, check_data=True, alpha=0.1):
        """
        fit rule for each node
        """
        if data:
            self.setup_data(data, check_data=check_data)
            self.descend_training_from_top(only_specific_match=True)

        self.fit_rule(alpha=alpha)

        self.estimate_uncertainty()

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

        if self.test_set:
            test_error = [self.evaluate(d.mol) - d.value for d in self.test_set]
            test_mae = np.mean(np.abs(np.array(test_error)))
            logging.info("test MAE: {}".format(test_mae))
            
        logging.info("# nodes: {}".format(len(self.nodes)))

    def estimate_uncertainty(self,rel_node_dof_tolerance=1e-5):
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
        rules = [n.rule.value for n in self.nodes.values()]
        rule_mean = abs(np.mean(rules))
        atol = rule_mean*rel_node_dof_tolerance
        self.abs_node_dof_tolerance = atol
        extra_dofs = len([n for n in rules if abs(n)<atol]) #count node rule values driven to zero by lasso
        
        if A.shape[1] != 1 and W is not None and len(self.datums) - len(nodes) + extra_dofs > 0:
            node_uncertainties = (
                np.diag(np.linalg.pinv((A.T @ W @ A).toarray()))
                * (self.data_delta**2).sum()
                / ((len(self.datums) - len(nodes) + extra_dofs))
            )
            self.node_uncertainties.update(
                {node.name: node_uncertainties[i] for i, node in enumerate(nodes)}
            )
        elif A.shape[1] != 1 and len(self.datums) - len(nodes) + extra_dofs > 0:
            node_uncertainties = (
                np.diag(np.linalg.pinv((A.T @ A).toarray()))
                * (self.data_delta**2).sum()
                / ((len(self.datums) - len(nodes) + extra_dofs))
            )
            self.node_uncertainties.update(
                {node.name: node_uncertainties[i] for i, node in enumerate(nodes)}
            )
            self.uncertainties_valid = True
        elif A.shape[1] != 1 and W is not None:
            logging.warning("too few degrees of freedom cannot compute valid uncertainties")
            node_uncertainties = (
                np.diag(np.linalg.pinv((A.T @ W @ A).toarray()))
                * (self.data_delta**2).sum()
            )
            self.node_uncertainties.update(
                {node.name: node_uncertainties[i] for i, node in enumerate(nodes)}
            )
            self.uncertainties_valid = False
        elif A.shape[1] != 1:
            logging.warning("too few degrees of freedom cannot compute valid uncertainties")
            if W is not None:
                node_uncertainties = (
                    np.diag(np.linalg.pinv((A.T @ W @ A).toarray()))
                    * (self.data_delta**2).sum()
                )
            else:
                node_uncertainties = (
                    np.diag(np.linalg.pinv((A.T @ A).toarray()))
                    * (self.data_delta**2).sum()
                )
            self.node_uncertainties.update(
                {node.name: node_uncertainties[i] for i, node in enumerate(nodes)}
            )
            self.uncertainties_valid = False
        else:
            self.node_uncertainties.update(
                {node.name: 1.0 for i, node in enumerate(nodes)}
            )
        
        for node in self.nodes.values():
            node.rule.uncertainty = self.node_uncertainties[node.name]
            
        for node in self.nodes.values():
            if node.rule.uncertainty is None:
                node.rule.uncertainty = node.parent.rule.uncertainty
            elif node.rule.num_data == 0:
                node.rule.uncertainty = 0.0 #if n=0 the LASSO should drive node.rule.value to zero so there should be approximately no variance contribution 

    def assign_depths(self):
        root = self.root
        _assign_depths(root)

    def evaluate(self, mol, trace=False, estimate_uncertainty=False):
        raise NotImplementedError
    
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


class MultiEvalSubgraphIsomorphicDecisionTreeRegressor(MultiEvalSubgraphIsomorphicDecisionTree):
    """
    Makes prediction for a molecule based on multiple evaluations.

    Args:
        `decomposition`: method to decompose a molecule into substructure contributions.
        `root_group`: root group for the tree
        `nodes`: dictionary of nodes for the tree
        `n_strucs_min`: minimum number of disconnected structures that can be in the group. Default is 1.
        `iter_max`: maximum number of times the extension generation algorithm is allowed to expand structures looking for additional splits. Default is 2.
        `iter_item_cap`: maximum number of structures the extension generation algorithm can send for expansion. Default is 100.
        `max_structures_to_generate_extensions`: maximum number of structures used in extension generation (a seeded random sample is drawn if larger than this number)
        `max_structures_to_choose_extension`: maximum number of structures used in choosing an extension (a seeded random sample is drawn if larger than this number)
        `fract_nodes_expand_per_iter`: fraction of nodes to split at each iteration. If 0, only 1 node will be split at each iteration.
        `r`: atom types to generate extensions. If None, all atom types will be used.
        `r_bonds`: bond types to generate extensions. If None, [1, 2, 3, 1.5, 4] will be used.
        `r_un`: unpaired electrons to generate extensions. If None, [0, 1, 2, 3] will be used.
        `r_site`: surface sites to generate extensions. If None, [] will be used.
        `r_morph`: surface morphology to generate extensions. If None, [] will be used.
    """
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
                if i in maxinds and len(node.items) > 1 and not np.isnan(rulevals[i])
            ]
            return nodes
        else:
            return selectable_nodes

    def extend_tree_from_node(self, parent):
        """
        Adds a new node to the tree
        """
        logging.info("Generating extensions")
        exts,clear_reg_dims = self.generate_extensions(parent)
        extlist = [ext[0] for ext in exts]
        if not extlist:
            self.skip_nodes.append(parent.name)
            return
        logging.info("choosing extensions")
        ext = self.choose_extension(parent, extlist)
        if ext is None:
            self.skip_nodes.append(parent.name)
            return
        logging.info("adding extension")
        new, comp = split_mols(parent.items, ext)
        ind = extlist.index(ext)
        grp, grpc, name, typ, indc = exts[ind]
        if clear_reg_dims:
            grp.clear_reg_dims()
            grpc.clear_reg_dims()
            
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

        logging.info("adding node {}".format(name))

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
                        if d.is_subgraph_isomorphic(
                            nodec.group, generate_initial_map=True, save_order=True
                        ):
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
        if len(node.items) <= self.max_structures_to_choose_extension:
            structs = node.items
        else:
            logging.info(f"Sampling {self.max_structures_to_choose_extension} structures from {len(node.items)} structures at node {node.name}")
            structs = np.random.choice(node.items,self.max_structures_to_choose_extension,replace=False)
            
        maxval = 0.0
        maxext = None
        for ext in exts:
            new, comp = split_mols(structs, ext)
            newval = 0.0
            compval = 0.0
            for i, datum in enumerate(self.datums):
                for j, d in enumerate(self.mol_node_maps[datum]["mols"]):
                    if any(d is x for x in new):
                        v = self.node_uncertainties[
                            self.mol_node_maps[datum]["nodes"][j].name
                        ]
                        s = sum(
                            self.node_uncertainties[
                                self.mol_node_maps[datum]["nodes"][k].name
                            ]
                            for k in range(len(self.mol_node_maps[datum]["nodes"]))
                        )
                        newval += self.data_delta[i] * v / s
                    elif any(d is x for x in comp):
                        v = self.node_uncertainties[
                            self.mol_node_maps[datum]["nodes"][j].name
                        ]
                        s = sum(
                            self.node_uncertainties[
                                self.mol_node_maps[datum]["nodes"][k].name
                            ]
                            for k in range(len(self.mol_node_maps[datum]["nodes"]))
                        )
                        compval += self.data_delta[i] * v / s
            val = abs(newval - compval)
            if val > maxval:
                maxval = val
                maxext = ext

        return maxext

    def evaluate(self, mol, trace=False, estimate_uncertainty=False):
        """
        Evaluate tree for a given possibly labeled mol
        """
        pred = 0.0
        unc = 0.0
        decomp = self.decompose(mol)
        if trace:
            tr = []
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
            if trace:
                tr.append(node.name)

        if estimate_uncertainty and trace:
            return pred, np.sqrt(unc), tr
        elif estimate_uncertainty:
            return pred, np.sqrt(unc)
        elif trace:
            return pred, tr
        else:
            return pred

class MultiEvalSubgraphIsomorphicDecisionTreeBinaryClassifier(MultiEvalSubgraphIsomorphicDecisionTree):
    """
    This SIDT class is a multi-evaluation "and" classifier 
    Every molecular decomposition is evaluated in the tree to True or False
    If any decomposition is False the input is classified as False, otherwise it is classified as True
    Note that one can use this as an "or" classifier by flipping the query of interest and the training data (False => True and True => False)
    Args:
        `decomposition`: method to decompose a molecule into substructure contributions.
        `root_group`: root group for the tree
        `nodes`: dictionary of nodes for the tree
        `n_strucs_min`: minimum number of disconnected structures that can be in the group. Default is 1.
        `iter_max`: maximum number of times the extension generation algorithm is allowed to expand structures looking for additional splits. Default is 2.
        `iter_item_cap`: maximum number of structures the extension generation algorithm can send for expansion. Default is 100.
        `max_structures_to_generate_extensions`: maximum number of structures used in extension generation (a seeded random sample is drawn if larger than this number)
        `max_structures_to_choose_extension`: maximum number of structures used in choosing an extension (a seeded random sample is drawn if larger than this number)
        `fract_nodes_expand_per_iter`: fraction of nodes to split at each iteration. If 0, only 1 node will be split at each iteration.
        `r`: atom types to generate extensions. If None, all atom types will be used.
        `r_bonds`: bond types to generate extensions. If None, [1, 2, 3, 1.5, 4] will be used.
        `r_un`: unpaired electrons to generate extensions. If None, [0, 1, 2, 3] will be used.
        `r_site`: surface sites to generate extensions. If None, [] will be used.
        `r_morph`: surface morphology to generate extensions. If None, [] will be used.
        `fract_threshold_to_predict_true`: fraction of relevant structures that favor true classification at which True will be predicted, this helps the algorithm avoid either false negatives or false positives when one is significantly preferable over the other  
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
        max_structures_to_generate_extensions=400,
        max_structures_to_choose_extension=np.inf,
        fract_nodes_expand_per_iter=0,
        max_batch_size=np.inf,
        new_fraction_threshold_to_reopt_node=0.25,
        r=None,
        r_bonds=None,
        r_un=None,
        r_site=None,
        r_morph=None,
        fract_threshold_to_predict_true=0.5,
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
            decomposition=decomposition,
            root_group=root_group,
            nodes=nodes,
            initial_root_splits=initial_root_splits,
            n_strucs_min=n_strucs_min,
            iter_max=iter_max,
            iter_item_cap=iter_item_cap,
            max_structures_to_generate_extensions=max_structures_to_generate_extensions,
            max_structures_to_choose_extension=max_structures_to_choose_extension,
            fract_nodes_expand_per_iter=fract_nodes_expand_per_iter,
            max_batch_size=max_batch_size,
            new_fraction_threshold_to_reopt_node=new_fraction_threshold_to_reopt_node,
            r=r,
            r_bonds=r_bonds,
            r_un=r_un,
            r_site=r_site,
            r_morph=r_morph,
            )

        self.fract_threshold_to_predict_true = fract_threshold_to_predict_true
        self.max_accuracy = 0.0

        if initial_root_splits:
            for node in self.nodes.values():
                if node.rule is None:
                    node.rule = True
        
    def select_nodes(self):
        """
        Picks the node with the most decompositions who a change in the decomposition's classification might improve overall classification
        adding multiple nodes would require the datum map variables to need updated before analyzing what data to split the node over as
        those maps may change when a node is added
        """
        self.datum_truth_map = {datum:[getattr(n,"rule") for n in self.mol_node_maps[datum]["nodes"]] for datum in self.datums}
        self.datum_node_map = {datum:[n for n in self.mol_node_maps[datum]["nodes"]] for datum in self.datums}
        if len(self.nodes) > 1:
            node_scores = {node.name:0 for node in self.nodes.values()}
            for k, datum in enumerate(self.datums):
                boos = self.datum_truth_map[datum]
                nodes = self.datum_node_map[datum]
                c = boos.count(False)
                for i in range(len(nodes)):
                    if datum.value and c >= 1:
                        if not boos[i]:
                            node_scores[nodes[i].name] += 1
                    elif not datum.value and c == 0:
                        if boos[i]:
                            node_scores[nodes[i].name] += 1
            rulevals = [
                node_scores[node.name]
                if len(node.items) > 1
                and not (node.name in self.new_nodes)
                and not (node.name in self.skip_nodes)
                else 0.0
                for node in self.nodes.values()
            ]
            inds = np.argsort(rulevals)
            ind = np.argmax(rulevals)
            v = np.max(rulevals)
            node = list(self.nodes.values())[ind]
            logging.info(f"selected {node.name}")
            if v == 0:
                return None
            else:
                return [node]
        else:
            return list(self.nodes.values())

    def choose_extension(self, node, exts):
        """
        select best extension based on the negative cross entropy
        returns a Node object
        almost always subclassed
        """
        if len(node.items) <= self.max_structures_to_choose_extension:
            structs = node.items
        else:
            logging.info(f"Sampling {self.max_structures_to_choose_extension} structures from {len(node.items)} structures at node {node.name}")
            structs = np.random.choice(node.items,self.max_structures_to_choose_extension,replace=False)
            
        maxval = -np.inf
        maxext = None
        new_maxrule = None
        comp_maxrule = None
        
        for i,ext in enumerate(exts):
            new, comp = split_mols(structs, ext)
            Nnew = len(new)
            Ncomp = len(comp)
            new_class_true = 0
            comp_class_true = 0
            for i, datum in enumerate(self.datums):
                for j, d in enumerate(self.mol_node_maps[datum]["mols"]):
                    if d in new:
                        new.remove(d)
                        if datum.value:
                            new_class_true += 1

                    if d in comp:
                        comp.remove(d)
                        if datum.value:
                            comp_class_true += 1

            assert len(new) == 0
            assert len(comp) == 0
            pnew = new_class_true/Nnew
            pcomp = comp_class_true/Ncomp
            assert pnew <= 1, pnew
            assert pcomp <= 1, pcomp
            if pnew == 0 and pcomp == 0:
                val = -np.inf
            elif pnew == 0:
                val = pcomp*np.log2(pcomp)
            elif pcomp == 0:
                val = pnew*np.log2(pnew)
            else:
                val = pcomp*np.log2(pcomp) + pnew*np.log2(pnew) #negative cross entropy
            if val > maxval:
                maxval = val
                maxext = ext
                if pnew >= self.fract_threshold_to_predict_true:
                    new_maxrule = True
                else:
                    new_maxrule = False
                if pcomp >= self.fract_threshold_to_predict_true:
                    comp_maxrule = True
                else:
                    comp_maxrule = False
        
        return maxext,new_maxrule,comp_maxrule
    
    def extend_tree_from_node(self, parent):
        """
        Adds a new node to the tree
        """
        total_items = parent.items #only give the parent the relevant items that may be important for the classification
        relevant_items = [] #note that self.datum_truth_map and self.datum_node_map are up to date because we only add one node at a time
        
        for k, datum in enumerate(self.datums):
            boos = self.datum_truth_map[datum]
            nodes = self.datum_node_map[datum]
            c = boos.count(False)
            for i in range(len(nodes)):
                mol = self.mol_node_maps[datum]["mols"][i]
                if nodes[i] != parent:
                    continue
                elif datum.value:
                    relevant_items.append(mol)
                elif not datum.value and c == 0:
                    relevant_items.append(mol)
                elif not datum.value and c == 1:
                    if not boos[i]:
                        relevant_items.append(mol)

        if not relevant_items:
            logging.info(f"no relevant items found skipping {parent.name}")
            self.skip_nodes.append(parent.name)
            return
        parent.items = relevant_items
        logging.info(f"extending node {parent.name}")
        Nitems = len(relevant_items)
        logging.info(f"considering {Nitems} relevant items")
        exts,clear_reg_dims = self.generate_extensions(parent)
        
        extlist = [ext[0] for ext in exts]
        if not extlist:
            logging.info(f"no extensions generated skipping {parent.name}")
            self.skip_nodes.append(parent.name)
            return

        ext,new_rule,comp_rule = self.choose_extension(parent, extlist)
        
        if clear_reg_dims:
            ext.clear_reg_dims()
            
        assert parent.name != "Root" or ext
        
        parent.items = total_items #fix parent.items now that we've picked an extension
        
        if ext is None:
            logging.info(f"no extension selected skipping node {parent.name}")
            self.skip_nodes.append(parent.name)
            return
        
        new, comp = split_mols(parent.items, ext)
        ind = extlist.index(ext)
        grp, grpc, name, typ, indc = exts[ind]
        
        node = Node(
            group=grp,
            items=new,
            rule=new_rule,
            parent=parent,
            children=[],
            name=name,
            depth=parent.depth + 1,
        )

        if name in self.nodes.keys():
            name_original = name
            k = 0
            while name in self.nodes.keys():
                name = name_original+"_ident_"+str(k)
                k += 1
        
        self.nodes[name] = node
        parent.children.append(node)
        self.new_nodes.append(name)

        for k, datum in enumerate(self.datums):
            for i, d in enumerate(self.mol_node_maps[datum]["mols"]):
                if any(d is x for x in new):
                    assert d.is_subgraph_isomorphic(
                        node.group, generate_initial_map=True, save_order=True
                    )
                    self.mol_node_maps[datum]["nodes"][i] = node

        logging.info("adding node {}".format(name))
        
        if grpc:
            class_true = 0
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
                rule=comp_rule,
                parent=parent,
                children=[],
                name=cextname,
                depth=parent.depth + 1,
            )

            self.nodes[cextname] = nodec
            parent.children.append(nodec)
            self.new_nodes.append(cextname)

            for k, datum in enumerate(self.datums):
                for i, d in enumerate(self.mol_node_maps[datum]["mols"]):
                    if any(d is x for x in comp):
                        assert d.is_subgraph_isomorphic(nodec.group, generate_initial_map=True, save_order=True), (d.to_adjacency_list(),nodec.group.to_adjacency_list())
                        self.mol_node_maps[datum]["nodes"][i] = nodec
            parent.items = []
        else:
            parent.items = comp
            parent.rule = comp_rule

    def evaluate(self, mol, return_decomp=False):
        """
        Evaluate tree for a given possibly labeled mol
        """
        out = 0.0
        decomp = self.decompose(mol)
        outval = True
        vs = []
        ms = []
        for d in decomp:
            children = self.root.children
            node = self.root
            boo = True
            while boo:
                for child in children:
                    if d.is_subgraph_isomorphic(
                        child.group, generate_initial_map=True, save_order=True
                    ):
                        children = child.children
                        node = child
                        break
                else:
                    boo = False

            if not node.rule:
                if not return_decomp:
                    return False
                else:
                    vs.append(False)
                    ms.append(d)
                    outval = False

            elif return_decomp:
                vs.append(True)
                ms.append(d)
        if return_decomp:
            return outval,ms,vs 
        else:
            return outval

    def analyze_error(self):
        """
        compute overall training and validation errors
        """
        sidt_train_values = [self.evaluate(d.mol) for d in self.datums]
        true_train_values = [d.value for d in self.datums]
        if self.validation_set:
            sidt_val_values = [self.evaluate(d.mol) for d in self.validation_set]
            true_val_values = [d.value for d in self.validation_set]

        if self.test_set:
            sidt_test_values = [self.evaluate(d.mol) for d in self.test_set]
            true_test_values = [d.value for d in self.test_set]
            
        P,N,PP,PN,TP,FN,FP,TN = analyze_binary_classification(sidt_train_values,true_train_values)

        train_acc = (TP + TN)/(P + N)

        logging.info(f"Training Accuracy: {train_acc}")

        if self.validation_set:
            P,N,PP,PN,TP,FN,FP,TN = analyze_binary_classification(sidt_val_values,true_val_values)
    
            val_acc = (TP + TN)/(P + N)
    
            logging.info(f"Validation Accuracy: {val_acc}")

            if self.test_set:
                P,N,PP,PN,TP,FN,FP,TN = analyze_binary_classification(sidt_test_values,true_test_values)
    
                test_acc = (TP + TN)/(P + N)
        
                logging.info(f"Test Accuracy: {test_acc}")
            
            acc = min(train_acc,val_acc)
    
            if acc > self.max_accuracy:
                logging.info(f"Identifying new best tree with min(train,val) accuracy {acc}")
                self.max_accuracy = acc
                self.best_tree_nodes = list(self.nodes.keys())
                self.best_nodes = {k: v for k, v in self.nodes.items()}
                self.best_mol_node_maps = {
                        k: {"mols": v["mols"][:], "nodes": v["nodes"][:]}
                        for k, v in self.mol_node_maps.items()
                    }
                self.best_rule_map = {name:self.nodes[name].rule for name in self.best_tree_nodes}

        logging.info("# nodes: {}".format(len(self.nodes)))

        return train_acc
        
    def generate_tree(
        self,
        data,
        check_data=True,
        validation_set=None,
        test_set=None,
        max_nodes=None,
        postpruning_based_on_val=True,
        root_classification=True,
        max_skip_node_clears=1,
        first_batch_include=[],
    ):
        """
        generate nodes for the tree based on the supplied data

        Args:
            `data`: list of Datum objects to train the tree
            `check_data`: if True, check that the data is subgraph isomorphic to the root group
            `validation_set`: list of Datum objects to validate the tree
            `max_nodes`: maximum number of nodes to generate
            `postpruning_based_on_val`: if True, regularize the tree based on the validation set
            `root_classification`: classification to set the root node to
        """
        np.random.seed(0)
        self.check_subgraph_isomorphic()
        
        self.root.rule = root_classification
        
        if self.max_batch_size > len(data):
            batches = [data]
        else:
            logging.info("using cascade algorithm, generating batches")
            batches = self.get_batches(data,first_batch_include=first_batch_include)
        data = []
        for i,batch in enumerate(batches):
            data += batch
            if len(batches) > 1:
                logging.info("Starting batch {0} with {1} data points".format(i+1,len(data)))
            if i != 0:
                logging.info("pruning tree with {} nodes".format(len(self.nodes)))
                self.prune(data)
                logging.info("pruned tree down to {} nodes".format(len(self.nodes)))
            
            self.setup_data(data, check_data=check_data)
            if len(self.nodes) > 1:
                self.descend_training_from_top(only_specific_match=True)
                for node in self.nodes.values():
                    if node.rule is None:
                        node.rule = True
            self.val_mae = np.inf
            self.skip_nodes = []
            self.new_nodes = []

            self.validation_set = validation_set
            self.test_set = test_set
            num_skip_node_clears = 0
            while True:
                self.analyze_error()
                if len(self.nodes) > max_nodes:
                    break
                self.new_nodes = []
                nodes = self.select_nodes()
                if not nodes:
                    if self.skip_nodes and num_skip_node_clears < max_skip_node_clears:
                        logging.info("Clearing skip_nodes")
                        num_skip_node_clears += 1
                        self.skip_nodes = []
                        continue
                    
                    logging.info("no selected nodes, terminating...")
                    break
                else:
                    for node in nodes:
                        self.extend_tree_from_node(node)
        
        if self.validation_set and postpruning_based_on_val:
            logging.info("Postpruning based on best validation accuracy")
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
            
    def trim_tree(self):
        """
        Many of the tree extension sets improve the split, but do not change predictions
        this function 1) merges nodes with their parents if they can be removed without affecting classifications
        2) merges nodes with their parents if they do not result in different predictions
        """

        self.datum_truth_map = {datum:[getattr(n,"rule") for n in self.mol_node_maps[datum]["nodes"]] for datum in self.datums}
        self.datum_node_map = {datum:[n for n in self.mol_node_maps[datum]["nodes"]] for datum in self.datums}

        check_nodes = True
        while check_nodes:
            check_nodes = False
            items_to_delete = []
            items_to_delete_values = []
            for name,node in self.nodes.items():
                if node.rule or node.parent is None:
                    continue
                break_loop = False
                for k, datum in enumerate(self.datums):
                    boos = self.datum_truth_map[datum]
                    nodes = self.datum_node_map[datum]
                    c = boos.count(False)
                    for i in range(len(nodes)):
                        if not datum.value and c == 1: #classification at this node matters for proper classification of this training item
                            break_loop = True
                            break
                    if break_loop:
                        break
                else:
                    items_to_delete.append(node)
                    items_to_delete_values.append(node.items)
                    
            if items_to_delete:
                ind = np.argmin(np.array(items_to_delete_values))
                node = items_to_delete[ind]
                logging.info(f"Deleting node {node.name} because unnecessary for classification")
                node.parent.children.remove(node)
                node.parent.children.extend(node.children)
                node.parent.items += node.items
                for n in items_to_delete.children:
                    n.parent = items_to_delete.parent
                check_nodes = True
        
        #merge nodes with parents that give same predictions 
        self.setup_data(data=self.datums)
        to_delete = []
        boo = True
        while boo:
            temp_to_delete = []
            for name,node in self.nodes.items():
                if name in to_delete:
                    continue
                if node.parent and node.rule == node.parent.rule:
                    seen_node = False
                    for child in node.parent.children:
                        break_loop = False
                        if child is node:
                            seen_node = True
                            continue
                        if seen_node:
                            for k, datum in enumerate(self.datums):
                                for i, d in enumerate(self.mol_node_maps[datum]["mols"]):
                                    if d.is_subgraph_isomorphic(node.group, generate_initial_map=True, save_order=True):
                                        if all(not d.is_subgraph_isomorphic(c.group, generate_initial_map=True, save_order=True) for c in node.children):
                                            if d.is_subgraph_isomorphic(child.group, generate_initial_map=True, save_order=True):
                                                break_loop = True
                                                break
                            if break_loop:
                                break
                    else:
                        logging.info(f"Removing node {node.name}")

                        ind = node.parent.children.index(node)
                        node.parent.children = node.parent.children[:ind] + node.children + node.parent.children[ind+1:]
                        for n in node.children:
                            n.parent = node.parent
                        node.parent.items += node.items
                        self.analyze_error()
                            
                        for k, datum in enumerate(self.datums):
                            for i, n in enumerate(self.mol_node_maps[datum]["nodes"]):
                                if n == node:
                                    assert self.mol_node_maps[datum]["nodes"][i].rule == node.parent.rule
                                    self.mol_node_maps[datum]["nodes"][i] = node.parent
                                    
                        temp_to_delete.append(name)
            to_delete.extend(temp_to_delete)
            boo = len(temp_to_delete) > 0 
            
        for name in to_delete:
            del self.nodes[name]
            
def analyze_binary_classification(preds,true_values):
    P = sum(true_values)
    N = len(preds) - P
    PP = sum(preds)
    PN = len(preds) - PP
    TP = sum([True for i in range(len(preds)) if preds[i] and true_values[i]])
    FN = sum([True for i in range(len(preds)) if not preds[i] and true_values[i]])
    FP = sum([True for i in range(len(preds)) if preds[i] and not true_values[i]])
    TN = sum([True for i in range(len(preds)) if not preds[i] and not true_values[i]])
    return P,N,PP,PN,TP,FN,FP,TN

def _assign_depths(node, depth=0):
    node.depth = depth
    for child in node.children:
        _assign_depths(child, depth=depth + 1)


def is_prepruned_by_uncertainty(node):
    return node.rule.uncertainty <= min(item.uncertainty for item in node.items)