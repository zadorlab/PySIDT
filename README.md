# PySIDT
This repository contains PySIDT a low-data machine learning algorithm for predicting chemical properties. PySIDT is a general implementation of the subgraph isomorphic tree generation (SIDT) machine learning algorithm developed in <a href="https://chemrxiv.org/engage/chemrxiv/article-details/62c5c941c79aca239053967e">Johnson and Green 2021</a>. While the algorithm in that work was specific to rate coefficients this implementation can be applied to prediction of arbitrary properties. 

# Installation from source
- Install [molecule](https://github.com/ReactionMechanismGenerator/molecule/tree/ts_len_improvements) from source
    - `git clone https://github.com/ReactionMechanismGenerator/molecule.git`
    - `cd molecule`
    - `git checkout ts_len_improvements`
    - `conda env create -f environment.yml --name=pysidt`
    - `conda activate pysidt`
    - `make`
    - `pip install -e .`
- Install PySIDT from source
    - `git clone https://github.com/zadorlab/PySIDT.git`
    - `cd PySIDT`
    - `conda activate pysidt`
    - `conda install pydot`
    - `pip install -e .`
