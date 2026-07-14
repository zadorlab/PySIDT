import numpy as np
import logging
from pysidt.extensions import get_extensions_for_generative_expansion, get_molecular_extensions_for_generative_expansion
from pysidt.utils import evaluate_single

def sum_min_weighting(target_values):
    return (target_values - np.min(target_values)) / np.sum(target_values - np.min(target_values))

def exp_neg_weighting(target_values,b):
    exp_vals = np.exp(-b * target_values)
    return exp_vals / np.sum(exp_vals)

def take_generative_step(grp,
    target_function,
    tree,
    decomposition,
    weighting_function,
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
    fraction_to_compute_exactly=0.1,
    specification_extensions_only=False,
    skip_specification_zero_delta_extensions=False,
    only_consider_objective_improving_extensions=False,
    extension_weighting={"shrink":0.2, "growth":0.2, "genspec":0.1, "spec":0.5}):
    """Takes an expansion step in the generative process.

    Args:
        grp: Base group to extend.
        target_function: Objective function f(grp,x,var_x) maximized in generative expansion.
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
        specification_extensions_only (bool, optional): Whether to only consider specification extensions. Defaults to False.
    Returns:
        tuple: Selected extension tuple from get_extensions_for_generative_expansion.

    """
    init_values,init_uncertainties = tree.evaluate(grp,estimate_uncertainty=True)
    
    init_target = target_function(grp, init_values, init_uncertainties)
    
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
        decomposition_associated=decomposition_associated,
        specification_extensions_only=specification_extensions_only,
)
    if not extents:
        raise ValueError("No candidate extensions generated for the group")

    deltas = np.array([x[-2] for x in extents])
    delta_vars = np.array([x[-1] for x in extents])
    
    rough_target_deltas = np.array([
        target_function(extents[i][0],
            init_values + delta,
            np.sqrt(np.maximum(0.0, np.sqrt(init_uncertainties**2 + delta_vars[i]))),
        )
        - init_target
        for i, delta in enumerate(deltas)
    ])
    
    inds = np.argsort(rough_target_deltas)[::-1]
    
    Nexact = int(len(extents) * fraction_to_compute_exactly)
    if Nexact == 0 and len(extents) > 0 and fraction_to_compute_exactly > 0:
        Nexact = 1
    
    exact_inds = inds[:Nexact]
    
    target_deltas_exact = []
    target_uncertainty_deltas_exact = []
    for i in exact_inds:
        ext = extents[i]
        grp = ext[0]
        new_target_values, new_target_uncertainties = tree.evaluate(grp, estimate_uncertainty=True)
        new_target_delta = target_function(grp,new_target_values,new_target_uncertainties) - init_target
        target_deltas_exact.append(new_target_delta)
        target_uncertainty_deltas_exact.append(new_target_uncertainties - init_uncertainties)
        
    target_deltas_exact = np.array(target_deltas_exact)
    target_uncertainty_deltas_exact = np.array(target_uncertainty_deltas_exact)
    
    target_deltas = rough_target_deltas
    target_deltas[exact_inds] = np.array(target_deltas_exact)
    
    # print("Target deltas for candidate extensions:", target_deltas)
    probs = weighting_function(target_deltas)

    assert all(probs >= 0), "Weighting function returned negative probabilities"
    
    ext_classes = np.unique([x[-4] for x in extents])
    ext_class_dict = {ext_class:0 for ext_class in ext_classes}
    
    is_nonnegative_target_delta = any(target_deltas > 0)
    
    shrink_inds = np.array([i for i,ext in enumerate(extents) if ext[-4] in ["genAtomRemovalExt", "genRemoveBridgeExt",]])
    growth_inds = np.array([i for i,ext in enumerate(extents) if ext[-4] in ["extNewBondExt",  "intNewBridgeExt"]])
    genspec_inds = np.array([i for i,ext in enumerate(extents) if ext[-4] not in ["atomGen","ringGen","elGen","lonepairGen","siteGen","morphGen","coordGen","bondGen"]])
    spec_inds = np.array([i for i,ext in enumerate(extents) if ext[-4] not in ["atomExt","ringExt","elExt","lonepairExt","siteExt","morphExt","coordExt","bondExt"]])
    
    for i,ext in enumerate(extents):
        if is_nonnegative_target_delta and only_consider_objective_improving_extensions and target_deltas[i] <= 0:
            probs[i] = 0.0
    
    if len(shrink_inds) > 0:
        shrink_sum = np.sum(probs[shrink_inds])
        if shrink_sum > 0:
            probs[shrink_inds] *= extension_weighting["shrink"]/np.sum(probs[shrink_inds])
            assert np.isclose(np.sum(probs[shrink_inds]), extension_weighting["shrink"]), f"Shrink class probability sum {np.sum(probs[shrink_inds])} not close to target {extension_weighting['shrink']}, probs: {probs[shrink_inds]}"
    if len(growth_inds) > 0:
        growth_sum = np.sum(probs[growth_inds])
        if growth_sum > 0:
            probs[growth_inds] *= extension_weighting["growth"]/np.sum(probs[growth_inds])
            assert np.isclose(np.sum(probs[growth_inds]), extension_weighting["growth"]), f"Growth class probability sum {np.sum(probs[growth_inds])} not close to target {extension_weighting['growth']}, probs: {probs[growth_inds]}"
    if len(genspec_inds) > 0:
        genspec_sum = np.sum(probs[genspec_inds])
        if genspec_sum > 0:
            probs[genspec_inds] *= extension_weighting["genspec"]/np.sum(probs[genspec_inds])
            assert np.isclose(np.sum(probs[genspec_inds]), extension_weighting["genspec"]), f"Genspec class probability sum {np.sum(probs[genspec_inds])} not close to target {extension_weighting['genspec']}, probs: {probs[genspec_inds]}"
    if len(spec_inds) > 0:
        spec_sum = np.sum(probs[spec_inds])
        if spec_sum > 0:
            probs[spec_inds] *= extension_weighting["spec"]/np.sum(probs[spec_inds])
            assert np.isclose(np.sum(probs[spec_inds]), extension_weighting["spec"]), f"Spec class probability sum {np.sum(probs[spec_inds])} not close to target {extension_weighting['spec']}, probs: {probs[spec_inds]}"

    for i,ext in enumerate(extents):
        ext_class_dict[ext[-4]] += probs[i]
        
    
    logging.error("Probability distribution over extension classes before renormalization:")
    logging.error(ext_class_dict)
    
    ext_class_dict = {ext_class:0 for ext_class in ext_classes}
    probs = probs / np.sum(probs)
    
    for i,ext in enumerate(extents):
        ext_class_dict[ext[-4]] += probs[i]
        
    
    logging.error("Probability distribution over extension classes:")
    logging.error(ext_class_dict)
    
    
    index = np.random.choice(range(len(target_deltas)), p=probs)
    
    #logging.error(f"Probability: {probs[index]} Probability distribution: {probs}")
    
    if index in exact_inds:
        eind = exact_inds.tolist().index(index)
        extents[index] = extents[index][:-2] + (target_deltas_exact[eind], target_uncertainty_deltas_exact[eind])
        return extents[index] + (new_target_values, new_target_uncertainties)
    else:
        new_target_values, new_target_uncertainties = tree.evaluate(grp, estimate_uncertainty=True)
        new_target_delta = target_function(extents[index][0],new_target_values,new_target_uncertainties) - init_target
        new_uncertainty_delta = new_target_uncertainties - init_uncertainties
        extents[index] = extents[index][:-2] + (new_target_delta, new_uncertainty_delta)
    return extents[index] + (new_target_values, new_target_uncertainties, )

def molecular_take_generative_step(mol,
    target_function,
    tree,
    decomposition,
    weighting_function,
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
    fraction_to_compute_exactly=0.1,
    generate_extensions_from_tree=True,
    generate_local_extensions=True,
    only_consider_objective_improving_extensions=False,
    extension_weighting={"shrink":0.2, "growth":0.2,  "transform":0.6},
    maximum_size=np.inf):
    """Takes an expansion step in the generative process.

    Args:
        mol: Base Molecule to extend.
        target_function: Objective function f(mol,x,var_x) maximized in generative expansion.
        tree: SIDT tree used to evaluate candidate extensions.
        decomposition: Decomposition mapping that preserves atom ordering relative to mol.
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
        specification_extensions_only (bool, optional): Whether to only consider specification extensions. Defaults to False.
    Returns:
        tuple: Selected extension tuple from get_extensions_for_generative_expansion.

    """
    init_values,init_uncertainties = tree.evaluate(mol,estimate_uncertainty=True)
    
    init_target = target_function(mol, init_values, init_uncertainties)
    
    extents = get_molecular_extensions_for_generative_expansion(
        mol,
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
        decomposition_associated=decomposition_associated,
        generate_extensions_from_tree=generate_extensions_from_tree,
        maximum_size=maximum_size,
)
    if not extents:
        raise ValueError("No candidate extensions generated for the group")

    deltas = np.array([x[-2] for x in extents])
    delta_vars = np.array([x[-1] for x in extents])
    
    rough_target_deltas = np.array([
        target_function(extents[i][0],
            init_values + delta,
            np.sqrt(np.maximum(0.0, np.sqrt(init_uncertainties**2 + delta_vars[i]))),
        )
        - init_target
        for i, delta in enumerate(deltas)
    ])
    
    inds = np.argsort(rough_target_deltas)[::-1]
    
    Nexact = int(len(extents) * fraction_to_compute_exactly)
    if Nexact == 0 and len(extents) > 0 and fraction_to_compute_exactly > 0:
        Nexact = 1
    
    exact_inds = inds[:Nexact]
    
    target_deltas_exact = []
    target_uncertainty_deltas_exact = []
    for i in exact_inds:
        ext = extents[i]
        grp = ext[0]
        new_target_values, new_target_uncertainties = tree.evaluate(grp, estimate_uncertainty=True)
        new_target_delta = target_function(grp,new_target_values,new_target_uncertainties) - init_target
        target_deltas_exact.append(new_target_delta)
        target_uncertainty_deltas_exact.append(new_target_uncertainties - init_uncertainties)
        
    target_deltas_exact = np.array(target_deltas_exact)
    target_uncertainty_deltas_exact = np.array(target_uncertainty_deltas_exact)
    
    target_deltas = rough_target_deltas
    target_deltas[exact_inds] = np.array(target_deltas_exact)
    
    # print("Target deltas for candidate extensions:", target_deltas)
    probs = weighting_function(target_deltas)

    assert all(probs >= 0), "Weighting function returned negative probabilities"
    
    ext_classes = np.unique([x[-4] for x in extents])
    ext_class_dict = {ext_class:0 for ext_class in ext_classes}
    
    is_nonnegative_target_delta = any(target_deltas > 0)
    
    shrink_inds = np.array([i for i,ext in enumerate(extents) if len(ext[0].atoms) < len(mol.atoms)])
    growth_inds = np.array([i for i,ext in enumerate(extents) if len(ext[0].atoms) > len(mol.atoms)])
    transform_inds = np.array([i for i,ext in enumerate(extents) if len(ext[0].atoms) == len(mol.atoms)])
    
    for i,ext in enumerate(extents):
        if is_nonnegative_target_delta and only_consider_objective_improving_extensions and target_deltas[i] <= 0:
            probs[i] = 0.0
    
    if len(shrink_inds) > 0:
        shrink_sum = np.sum(probs[shrink_inds])
        if shrink_sum > 0:
            probs[shrink_inds] *= extension_weighting["shrink"]/np.sum(probs[shrink_inds])
            assert np.isclose(np.sum(probs[shrink_inds]), extension_weighting["shrink"]), f"Shrink class probability sum {np.sum(probs[shrink_inds])} not close to target {extension_weighting['shrink']}, probs: {probs[shrink_inds]}"
    if len(growth_inds) > 0:
        growth_sum = np.sum(probs[growth_inds])
        if growth_sum > 0:
            probs[growth_inds] *= extension_weighting["growth"]/np.sum(probs[growth_inds])
            assert np.isclose(np.sum(probs[growth_inds]), extension_weighting["growth"]), f"Growth class probability sum {np.sum(probs[growth_inds])} not close to target {extension_weighting['growth']}, probs: {probs[growth_inds]}"
    if len(transform_inds) > 0:
        transform_sum = np.sum(probs[transform_inds])
        if transform_sum > 0:
            probs[transform_inds] *= extension_weighting["transform"]/np.sum(probs[transform_inds])
            assert np.isclose(np.sum(probs[transform_inds]), extension_weighting["transform"]), f"Transform class probability sum {np.sum(probs[transform_inds])} not close to target {extension_weighting['transform']}, probs: {probs[transform_inds]}"
    
    for i,ext in enumerate(extents):
        ext_class_dict[ext[-4]] += probs[i]
        
    
    logging.error("Probability distribution over extension classes before renormalization:")
    logging.error(ext_class_dict)
    
    ext_class_dict = {ext_class:0 for ext_class in ext_classes}
    probs = probs / np.sum(probs)
    
    for i,ext in enumerate(extents):
        ext_class_dict[ext[-4]] += probs[i]
        
    
    logging.error("Probability distribution over extension classes:")
    logging.error(ext_class_dict)
    
    
    index = np.random.choice(range(len(target_deltas)), p=probs)
    
    #logging.error(f"Probability: {probs[index]} Probability distribution: {probs}")
    
    if index in exact_inds:
        eind = exact_inds.tolist().index(index)
        extents[index] = extents[index][:-2] + (target_deltas_exact[eind], target_uncertainty_deltas_exact[eind])
        return extents[index] + (new_target_values, new_target_uncertainties)
    else:
        new_target_values, new_target_uncertainties = tree.evaluate(grp, estimate_uncertainty=True)
        new_target_delta = target_function(extents[index][0],new_target_values,new_target_uncertainties) - init_target
        new_uncertainty_delta = new_target_uncertainties - init_uncertainties
        extents[index] = extents[index][:-2] + (new_target_delta, new_uncertainty_delta)
    return extents[index] + (new_target_values, new_target_uncertainties, )

def generate_structure(grp,
    target_function,
    tree,
    decomposition,
    weighting_function,
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
    fraction_to_compute_exactly=0.1,
    iters_per_nstruct=20,
    log_groups=False):
    """
    Generates
    Args:
        grp: Base group to extend.
        target_function: Objective function f(x) maximized initially in generative expansion.
        target_function_with_uncertainty: Objective function f(x, var_x) that penalizes uncertainty maximized in generative expansion.
        tree: SIDT tree used to evaluate candidate extensions.
        decomposition: Decomposition mapping that preserves atom ordering relative to grp.
        weighting_function: Function that takes an array of target deltas and returns a probability distribution over them for selection of generative extensions.
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
        iters_per_nstruct (int, optional): Number of iterations to perform per structure in stage 2. Defaults to 20.
    """
    struct = grp.copy(deep=True)
    name = basename
    Nstruct = len(struct.atoms)
    stage = 1
    iter = 1
    iter_stage_2 = 0
    if log_groups:
        logged_groups = [struct]
    while True:
        print(f"Iteration {iter}, Stage {stage}, Number of atoms: {len(struct.atoms)}")
        print(struct.to_adjacency_list())
        if stage == 1: #get up to the size scale of the system
            struct, _, name, typename, tup, delta_v, delta_var, target_values, target_uncertainties = take_generative_step(
                struct,
                target_function,
                tree,
                decomposition,
                weighting_function,
                r_full,
                r_bonds_full=r_bonds_full,
                r_un_full=r_un_full,
                r_site_full=r_site_full,
                r_morph_full=r_morph_full,
                r_ncoord_full=r_ncoord_full,
                r_label=r_label,
                r_lone_pairs_full=r_lone_pairs_full,
                basename=name,
                n_strucs_min=n_strucs_min,
                n_strucs_max=n_strucs_max,
                max_ring_gen_size=max_ring_gen_size,
                decomposition_associated=decomposition_associated,
                fraction_to_compute_exactly=fraction_to_compute_exactly)
            
            if len(struct.atoms) < Nstruct:
                Nstruct_stage_2 = Nstruct
                stage = 2
                logging.info("Moving to stage 2 at structure size %d", Nstruct_stage_2)
            Nstruct = len(struct.atoms)
            
        elif stage == 2: #generative refinement
            struct, _, name, typename, tup, delta_v, delta_var, target_values, target_uncertainties = take_generative_step(
                struct,
                target_function,
                tree,
                decomposition,
                weighting_function,
                r_full,
                r_bonds_full=r_bonds_full,
                r_un_full=r_un_full,
                r_site_full=r_site_full,
                r_morph_full=r_morph_full,
                r_ncoord_full=r_ncoord_full,
                r_label=r_label,
                r_lone_pairs_full=r_lone_pairs_full,
                basename=name,
                n_strucs_min=n_strucs_min,
                n_strucs_max=n_strucs_max,
                max_ring_gen_size=max_ring_gen_size,
                decomposition_associated=decomposition_associated,
                fraction_to_compute_exactly=fraction_to_compute_exactly)
            iter_stage_2 += 1
            if iter_stage_2 > iters_per_nstruct * Nstruct_stage_2:
                stage = 3
        elif stage == 3: #make structure more specific until it is fully realized
            try:
                struct, _, name, typename, tup, delta_v, delta_var, target_values, target_uncertainties = take_generative_step(
                    struct,
                    target_function,
                    tree,
                    decomposition,
                    weighting_function,
                    r_full,
                    r_bonds_full=r_bonds_full,
                    r_un_full=r_un_full,
                    r_site_full=r_site_full,
                    r_morph_full=r_morph_full,
                    r_ncoord_full=r_ncoord_full,
                    r_label=r_label,
                    r_lone_pairs_full=r_lone_pairs_full,
                    basename=name,
                    n_strucs_min=n_strucs_min,
                    n_strucs_max=n_strucs_max,
                    max_ring_gen_size=max_ring_gen_size,
                    decomposition_associated=decomposition_associated,
                    fraction_to_compute_exactly=fraction_to_compute_exactly,
                    specification_extensions_only=True) 
            except ValueError:
                break
        else:
            raise ValueError("Invalid stage value")
        objective = target_function(target_values, target_uncertainties)
        print(f"Selected extension: {typename}, {tup}, objective: {objective}, delta_v: {delta_v}, delta_var: {delta_var}, target_values: {target_values}, target_uncertainties: {target_uncertainties}")
        if log_groups:
            logged_groups.append(struct)
        iter += 1
        
    if log_groups:
        return struct, logged_groups
    else:
        return struct
    
def molecular_generate_structure(mol,
    target_function,
    tree,
    decomposition,
    weighting_function,
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
    fraction_to_compute_exactly=0.1,
    iters_per_nstruct=20,
    log_groups=False,
    generate_extensions_from_tree=True,
    generate_local_extensions=True,
    maximum_size=np.inf):
    """
    Generates
    Args:
        grp: Base group to extend.
        target_function: Objective function f(x) maximized initially in generative expansion.
        target_function_with_uncertainty: Objective function f(x, var_x) that penalizes uncertainty maximized in generative expansion.
        tree: SIDT tree used to evaluate candidate extensions.
        decomposition: Decomposition mapping that preserves atom ordering relative to grp.
        weighting_function: Function that takes an array of target deltas and returns a probability distribution over them for selection of generative extensions.
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
        iters_per_nstruct (int, optional): Number of iterations to perform per structure in stage 2. Defaults to 20.
    """
    struct = mol.copy(deep=True)
    name = basename
    Nstruct = len(struct.atoms)
    stage = 1
    iter = 1
    iter_stage_2 = 0

    structs = [struct]
    v,unc = tree.evaluate(struct,estimate_uncertainty=True)
    objectives = [target_function(struct,v,unc)]
    while True:
        print(f"Iteration {iter}, Stage {stage}, Number of atoms: {len(struct.atoms)}")
        print(struct.to_adjacency_list())
        if stage == 1: #get up to the size scale of the system
            struct, _, name, typename, tup, delta_v, delta_var, target_values, target_uncertainties = molecular_take_generative_step(
                struct,
                target_function,
                tree,
                decomposition,
                weighting_function,
                r_full,
                r_bonds_full=r_bonds_full,
                r_un_full=r_un_full,
                r_site_full=r_site_full,
                r_morph_full=r_morph_full,
                r_ncoord_full=r_ncoord_full,
                r_label=r_label,
                r_lone_pairs_full=r_lone_pairs_full,
                basename=name,
                n_strucs_min=n_strucs_min,
                n_strucs_max=n_strucs_max,
                max_ring_gen_size=max_ring_gen_size,
                decomposition_associated=decomposition_associated,
                fraction_to_compute_exactly=fraction_to_compute_exactly,
                generate_extensions_from_tree=generate_extensions_from_tree,
                generate_local_extensions=generate_local_extensions,
                maximum_size=maximum_size)
            
            if len(struct.atoms) < Nstruct:
                Nstruct_stage_2 = Nstruct
                stage = 2
                logging.info("Moving to stage 2 at structure size %d", Nstruct_stage_2)
            Nstruct = len(struct.atoms)
            
        elif stage == 2: #generative refinement
            struct, _, name, typename, tup, delta_v, delta_var, target_values, target_uncertainties = molecular_take_generative_step(
                struct,
                target_function,
                tree,
                decomposition,
                weighting_function,
                r_full,
                r_bonds_full=r_bonds_full,
                r_un_full=r_un_full,
                r_site_full=r_site_full,
                r_morph_full=r_morph_full,
                r_ncoord_full=r_ncoord_full,
                r_label=r_label,
                r_lone_pairs_full=r_lone_pairs_full,
                basename=name,
                n_strucs_min=n_strucs_min,
                n_strucs_max=n_strucs_max,
                max_ring_gen_size=max_ring_gen_size,
                decomposition_associated=decomposition_associated,
                fraction_to_compute_exactly=fraction_to_compute_exactly,
                generate_extensions_from_tree=generate_extensions_from_tree,
                generate_local_extensions=generate_local_extensions,
                maximum_size=maximum_size)
            iter_stage_2 += 1
            if iter_stage_2 > iters_per_nstruct * Nstruct_stage_2:
                stage = 3

        objective = target_function(struct, target_values, target_uncertainties)
        print(f"Selected extension: {typename}, {tup}, objective: {objective}, delta_v: {delta_v}, delta_var: {delta_var}, target_values: {target_values}, target_uncertainties: {target_uncertainties}")
        structs.append(struct)
        objectives.append(objective)
        iter += 1
        if stage == 3:
            break
    
    maxind = np.argmax(objectives)
    
    return structs[maxind],objectives[maxind]
