# <img align="top" style="width:200px;height:80px;" src="https://github.com/zadorlab/PySIDT/blob/main/PySIDT_logo.png"> 
PySIDT is a Python package for training and running inference on subgraph isomorphic decision trees (SIDTs) for molecular property prediction as described in <a href="https://doi.org/10.26434/chemrxiv-2024-vbh8g">Johnson et al. 2025</a>. 

SIDTs are graph-based decision trees made of nodes associated with molecular substructures. Inference occurs by descending target molecular structures down the decision tree to nodes with matching subgraph isomorphic substructures and making predictions based on the final (most specific) node matched. SIDTs can perform significantly better on smaller datasets (<10,000 datapoints) than deep neural network based approaches. Being trees of molecular substructures, SIDTs are inherently readable and easy to visualize, making them easy to analyze. They are also straightforward to extend and retrain, facilitate uncertainty estimation, and enable integration of expert knowledge.


The SIDT technique was originally developed in <a href="https://doi.org/10.1039/D3RE00684K">Johnson and Green 2024</a>. This implementation incorporates uncertainty prepruning, as detailed in  <a href="https://doi.org/10.1021/acs.jpca.4c00569">Pang et al. 2024</a>. 

Documentation for PySIDT is available <a href="https://github.com/zadorlab/PySIDT/wiki">here</a>.

# Installation from source
- Install PySIDT from source
    - `git clone https://github.com/zadorlab/PySIDT.git`
    - `cd PySIDT`
    - `conda env create -f environment.yml`
    - `conda activate pysidt_env`
    - `pip install -e .`

# Install molecule from source to customize atomtypes
- Install [molecule](https://github.com/ReactionMechanismGenerator/molecule) from source
    - `git clone https://github.com/ReactionMechanismGenerator/molecule.git`
    - `cd molecule`
    - `conda activate pysidt_env`
    - `make`
    - `pip install -e .`
