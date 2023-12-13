from molecule.molecule import Molecule, Group
from pysidt.extensions import split_mols,get_extension_edge
import numpy as np
import logging
import json

class Node:
    
    def __init__(self,group=None,items=[],rule=None,parent=None,children=[],name=None):
        self.group = group
        self.items = items #list of Datum objects
        self.rule = rule
        self.parent = parent
        self.children = children
        self.name = name

class Datum:
    """
    Data for training tree
    mol is a possibly labeled Molecule object
    value can be in any format so long as the rule generation process can handle it
    """
    def __init__(self,mol,value):
        self.mol = mol
        self.value = value

class SubgraphIsomorphicDecisionTree:
    
    def __init__(self,root_group=None,nodes=dict(),n_splits=1,iter_max=2,iter_item_cap=100, 
                 r=None, r_bonds=[1,2,3,1.5,4], r_un=[0,1,2,3],r_site=[],r_morph=[]):
        self.nodes = nodes
        self.n_splits = n_splits 
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
            self.root = Node(root_group,name="Root")
            self.nodes = {"Root":self.root}
    
    def load(self,nodes):
        
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
        for name,node in self.nodes.items():
            if len(node.items) > 1 and not (node.name in self.skip_nodes): 
                logging.info("Selected node {}".format(node.name))
                logging.info("Node has {} items".format(len(node.items)))
                return node
        else:
            return None
    
    def generate_extensions(self,node,recursing=False):
        """
        Generates set of extension groups to a node
        returns list of Groups
        design not to subclass
        """

        out, gave_up_split = get_extension_edge(node, self.n_splits, r=self.r, r_bonds=self.r_bonds, r_un=self.r_un,
                                                r_site=self.r_site, r_morph=self.r_morph,
                                                iter_max=self.iter_max, iter_item_cap=self.iter_item_cap)
        
        if len(out) == 0 and not recursing:
            logging.error("recursing")
            logging.error(node.group.to_adjacency_list())
            # for item in node.items:
            #     logging.error(item.to_adjacency_list())
            node.group.clear_reg_dims()
            return self.generate_extensions(node,recursing=True)
            
        #logging.error([x.mol.to_adjacency_list() for x in node.items])
        return out #[(grp2, grpc, name, typ, indc)]
    
    def choose_extension(self,node,exts):
        """
        select best extension among the set of extensions
        returns a Node object
        almost always subclassed
        """
        minval = np.inf
        minext = None
        for ext in exts:
            new,comp = split_mols(node.items, ext)
            val = np.std([x.value for x in new])*len(new) + np.std([x.value for x in comp])*len(comp)
            if val < minval:
                minval = val 
                minext = ext 
        
        return minext
    
    def extend_tree_from_node(self,parent):
        """
        Adds a new node to the tree 
        """
        exts = self.generate_extensions(parent)
        extlist = [ext[0] for ext in exts]
        if len(extlist) == 0:
            self.skip_nodes.append(parent.name)
            return
        ext = self.choose_extension(parent,extlist)
        new,comp = split_mols(parent.items, ext)
        ind = extlist.index(ext)
        grp,grpc,name,typ,indc = exts[ind]
        logging.info("Choose extension {}".format(name))
        
        node = Node(group=grp,items=new,rule=None,parent=parent,children=[],name=name)
        self.nodes[name] = node
        parent.children.append(node)
        if grpc:
            frags = name.split('_')
            frags[-1] = 'N-' + frags[-1]
            cextname = ''
            for k in frags:
                cextname += k
                cextname += '_'
            cextname = cextname[:-1]
            nodec = Node(group=grpc,items=comp,rule=None,parent=parent,children=[],name=cextname)
            self.nodes[cextname] = nodec
            parent.children.append(nodec)
            parent.items = []
        else:
            for mol in new:
                parent.items.remove(mol)
        
    def descend_training_from_top(self,only_specific_match=True):
        """
        Moves training data as needed down the tree
        """
        nodes = [self.root]
        
        while nodes != []:
            new_nodes = []
            for node in nodes:
                self.descend_node(node,only_specific_match=only_specific_match)
                new_nodes += node.children 
            nodes = new_nodes 
    
    def descend_node(self,node,only_specific_match=True):
        for child in node.children:
            data_to_add = []
            for datum in node.items:
                if datum.mol.is_subgraph_isomorphic(child.group, generate_initial_map=True, save_order=True):
                    data_to_add.append(datum)
            
            for datum in data_to_add:
                child.items.append(datum)
                if only_specific_match:
                    node.items.remove(datum)
                        
    def clear_data(self):
        for node in self.nodes.values():
            node.items = []
            
    def generate_tree(self,data=None,check_data=True):
        """
        generate nodes for the tree based on the supplied data
        """
        self.skip_nodes = []
        if data:
            if check_data:
                for datum in data:
                    if not datum.mol.is_subgraph_isomorphic(self.root.group, generate_initial_map=True, save_order=True):
                        logging.error("Datum did not match Root node:")
                        logging.error(datum.mol.to_adjacency_list())
                        raise ValueError
                        
            self.clear_data()
            self.root.items = data[:]
        
        node = self.select_node()
        
        while node is not None:
            self.extend_tree_from_node(node)
            node = self.select_node()
    
    def fit_tree(self,data=None):
        """
        fit rule for each node
        """
        if data:
            self.clear_data()
            self.root.items = data[:]
            self.descend_training_from_top(only_specific_match=False)
        
        for node in self.nodes.values():
            if len(node.items) == 0:
                logging.error(node.name)
                raise ValueError
            node.rule = sum([d.value for d in node.items])/len(node.items)
    
    def evaluate(self,mol):
        """
        Evaluate tree for a given possibly labeled mol
        """
        children = self.root.children 
        node = self.root
        
        while children != []:
            for child in children:
                if mol.is_subgraph_isomorphic(child.group, generate_initial_map=True, save_order=True):
                    children = child.children 
                    node = child
                    break
            else:
                return node.rule
        
        return node.rule

def write_nodes(tree,file):
    nodesdict = dict()
    for node in tree.nodes.values():
        if node.parent is None:
            p = None
        else:
            p = node.parent.name
        nodesdict[node.name] = {"group": node.group.to_adjacency_list(),
                       "rule": node.rule,
                       "parent": p,
                       "children": [x.name for x in node.children],
                       "name": node.name}

    with open(file,'w') as f:
        json.dump(nodesdict,f)

def read_nodes(file):
    with open(file,'r') as f:
        nodesdict = json.load(f)
    nodes = dict()
    for n,d in nodesdict.items():
        nodes[n] = Node(group=Group().from_adjacency_list(d["group"]),
                        rule=d["rule"],
                        parent=d["parent"],
                        children = d["children"],
                        name = d["name"])

    for n,node in nodes.items():
        if node.parent:
            node.parent = nodes[node.parent]
        node.children = [nodes[child] for child in node.children]
    
    return nodes