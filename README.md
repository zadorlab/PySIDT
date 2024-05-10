# PySIDT
This repository contains PySIDT, a package containing a set of low-data machine-learning algorithms for prediction of chemical properties based on the subgraph isomorphic tree generation (SIDT) approach originally developed in <a href="https://doi.org/10.1039/D3RE00684K">Johnson and Green 2021</a>. 

# Installation from source
- Install PySIDT from source
    - `git clone https://github.com/zadorlab/PySIDT.git`
    - `cd PySIDT`
    - `conda activate pysidt`
    - `pip install -e .`

# Install molecule from source to customize atomtypes
- Install [molecule](https://github.com/ReactionMechanismGenerator/molecule) from source
    - `git clone https://github.com/ReactionMechanismGenerator/molecule.git`
    - `cd molecule`
    - `conda activate pysidt`
    - `make`
    - `pip install -e .`
