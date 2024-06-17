# PySIDT
This repository contains PySIDT, a package containing a set of low-data machine-learning algorithms for prediction of chemical properties based on the subgraph isomorphic tree generation (SIDT) approach originally developed in <a href="https://doi.org/10.1039/D3RE00684K">Johnson and Green 2024</a>. While the algorithm in that work was specific to rate coefficients, this implementation can be applied to prediction of arbitrary properties. This implementation also incorporates uncertainty prepruning, as detailed in  <a href="https://doi.org/10.1021/acs.jpca.4c00569">Pang et al. 2024</a>.

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
