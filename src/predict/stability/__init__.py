"""Stage 4: feature stability via ICC perturbation gate.

See ``docs/modules/stability.md`` and decisions D013 to D016.
"""
from predict.stability.icc import (
    IccRecord,
    IccSource,
    build_reliability_matrix,
    gate_features,
    icc_3_1_absolute,
    invariant_by_construction_features,
)
from predict.stability.perturbations import (
    PerturbationSpec,
    add_gaussian_noise,
    apply_perturbation,
    enumerate_perturbations,
    load_preprocessed_ct_and_mask,
    noise_seed,
    rotate,
    translate,
)

__all__ = [
    # icc
    "IccRecord",
    "IccSource",
    "build_reliability_matrix",
    "gate_features",
    "icc_3_1_absolute",
    "invariant_by_construction_features",
    # perturbations
    "PerturbationSpec",
    "add_gaussian_noise",
    "apply_perturbation",
    "enumerate_perturbations",
    "load_preprocessed_ct_and_mask",
    "noise_seed",
    "rotate",
    "translate",
]
