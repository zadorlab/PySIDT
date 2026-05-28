import numpy as np
from extensions import get_extensions_for_generative_expansion

def take_generative_step(grp,
    target_function,
    tree,
    decomposition,
    r_full,
    r_bonds_full=[1, 2, 3, 1.5, 4],
    r_un_full=[0, 1, 2, 3],
    r_site_full=[],
    r_morph_full=[],
    r_ncoord_full=[],
    r_label=None,
    r_lone_pairs_full=[],
    basename="",
    n_strucs_min=None,
    n_strucs_max=None,
    max_ring_gen_size=None,
    decomposition_associated=None,
    fraction_to_compute_exactly=0.1):
    """Takes an expansion step in the generative process.

    Args:
        grp: Base group to extend.
        target_function: Objective function f(x,var_x) maximized in generative expansion.
        tree: SIDT tree used to evaluate candidate extensions.
        decomposition: Decomposition mapping that preserves atom ordering relative to grp.
        r_full: Allowed atom types for new atoms; defaults to bond dissociation elements if None.
        r_bonds_full (list, optional): Allowed bond orders for generated new bonds. Defaults to [1, 2, 3, 1.5, 4].
        r_un_full (list, optional): Allowed unpaired electron counts for generated atoms. Defaults to [0, 1, 2, 3].
        r_site_full (list, optional): Allowed site labels for generated atoms. Defaults to [].
        r_morph_full (list, optional): Allowed morphological atom type values for generated atoms. Defaults to [].
        r_ncoord_full (list, optional): Allowed coordination numbers for generated atoms. Defaults to [].
        r_label (list, optional): Allowed atom labels for generated atoms. Defaults to None, which is treated as [''].
        r_lone_pairs_full (list, optional): Allowed lone pair counts for generated atoms. Defaults to [].
        basename (str, optional): Prefix for generated extension names. Defaults to "".
        n_strucs_min (int, optional): Minimum number of fragments in generated structures. Defaults to None.
        n_strucs_max (int, optional): Maximum number of fragments in generated structures. Defaults to None.
        max_ring_gen_size (int, optional): Maximum size of generated rings for internal bond extensions. Defaults to None.
        decomposition_associated (callable, optional): Function that selects whether a decomposition is associated with an atom change. Defaults to None.
        fraction_to_compute_exactly (float, optional): Fraction of top candidates to evaluate exactly. Defaults to 0.1.

    Returns:
        tuple: Selected extension tuple from get_extensions_for_generative_expansion.

    """
    init_values,init_uncertainties = tree.evaluate(grp,estimate_uncertainty=True)
    
    init_target = target_function(init_values, init_uncertainties)
    
    extents = get_extensions_for_generative_expansion(
    grp,
    tree,
    decomposition,
    r_full,
    r_bonds_full=r_bonds_full,
    r_un_full=r_un_full,
    r_site_full=r_site_full,
    r_morph_full=r_morph_full,
    r_ncoord_full=r_ncoord_full,
    r_label=r_label,
    r_lone_pairs_full=r_lone_pairs_full,
    basename=basename,
    n_strucs_min=n_strucs_min,
    n_strucs_max=n_strucs_max,
    max_ring_gen_size=max_ring_gen_size,
    decomposition_associated=decomposition_associated)

    if not extents:
        raise ValueError("No candidate extensions generated for the group")
    
    deltas = [x[-2] for x in extents]
    delta_uncertainties = [x[-1] for x in extents]
    
    rough_target_deltas = np.array([target_function(init_values+delta, delta_uncertainties[i]) - init_target for delta in deltas])
    
    inds = np.argsort(rough_target_deltas)[::-1]
    
    Nexact = int(len(extents) * fraction_to_compute_exactly)
    if Nexact == 0 and len(extents) > 0 and fraction_to_compute_exactly > 0:
        Nexact = 1
    
    exact_inds = inds[:Nexact]
    
    target_deltas_exact = []

    for i in exact_inds:
        ext = extents[i]
        grp = ext[0]
        
        new_target_values, new_target_uncertainties = tree.evaluate(grp, estimate_uncertainty=True)
        new_target_delta = target_function(new_target_values,new_target_uncertainties) - init_target
        target_deltas_exact.append(new_target_delta)
        
    target_deltas_exact = np.array(target_deltas_exact)
    
    target_deltas = rough_target_deltas
    target_deltas[exact_inds] = np.array(target_deltas_exact)
    
    target_deltas -= np.min(target_deltas)
    
    index = np.choice(range(len(target_deltas)), p=target_deltas/np.sum(target_deltas))
    
    return extents[index]