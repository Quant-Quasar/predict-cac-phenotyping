"""Stage 7: phenotype characterisation.

Modules (added incrementally during stage 7 build; see docs/modules/analyse.md
and decisions D023 through D028 for full rationale):
  * profiles      - per-cluster median + IQR + Cliff's delta + Mann-Whitney
                    + FDR-BH; biological sanity + label balance gates (D023)
  * orthogonality - burden orthogonality (Mann-Whitney + Levene + Cliff's
                    delta on agatston_total) + burden-stratified spatial
                    replication (D024)
  * hypotheses    - 6 pre-specified directional one-sided Mann-Whitney
                    with two-tier verdict (D025)
  * monotonicity  - Spearman + Kendall vs agatston_total burden-axis
                    classification (D026)
  * signatures    - top-N distinguishing features from FDR-adjusted
                    Cliff's delta
  * cross_cohort  - 3-rule feature consistency + partition ARI on shared
                    pids (D027)
  * paper_table   - 15-row main table + 5-row robust sensitivity (D028)
"""
from predict.analyse.hypotheses import (
    DIRECTIONAL_HYPOTHESES,
    directional_hypotheses_table,
    overall_verdict,
    primary_pass,
    run_directional_test,
    secondary_pass,
)
from predict.analyse.cross_cohort import (
    consistency_table,
    partition_ari_table,
    robust_discriminator_count_summary,
)
from predict.analyse.derived_features import (
    augment_raw_with_derived,
    derive_high_density_fraction,
    derive_vessel_burden_gini,
)
from predict.analyse.paper_table import (
    build_paper_table,
    build_robust_sensitivity_table,
)
from predict.analyse.monotonicity import (
    classification_summary,
    classify_feature,
    compute_monotonicity,
)
from predict.analyse.signatures import (
    signature_paragraph_for_paper,
    top_n_signatures,
)
from predict.analyse.orthogonality import (
    BurdenOrthogonalityResult,
    assess_burden_orthogonality,
    burden_stratified_pass_verdict,
    burden_stratified_spatial_replication,
    burden_tertile_assignment,
)
from predict.analyse.profiles import (
    apply_fdr_bh,
    assert_biological_sanity,
    assert_label_balance,
    cliffs_delta,
    compute_cluster_profile,
    determine_focal_diffuse_mapping,
    mannwhitney_u_pval,
)

__all__ = [
    # profiles
    "apply_fdr_bh", "assert_biological_sanity", "assert_label_balance",
    "cliffs_delta", "compute_cluster_profile",
    "determine_focal_diffuse_mapping", "mannwhitney_u_pval",
    # orthogonality
    "BurdenOrthogonalityResult", "assess_burden_orthogonality",
    "burden_stratified_pass_verdict",
    "burden_stratified_spatial_replication", "burden_tertile_assignment",
    # hypotheses
    "DIRECTIONAL_HYPOTHESES", "directional_hypotheses_table",
    "overall_verdict", "primary_pass", "run_directional_test",
    "secondary_pass",
    # monotonicity
    "classification_summary", "classify_feature", "compute_monotonicity",
    # signatures
    "signature_paragraph_for_paper", "top_n_signatures",
    # cross_cohort
    "consistency_table", "partition_ari_table",
    "robust_discriminator_count_summary",
    # derived_features (D019 helpers re-derived at raw scale for stage 7)
    "augment_raw_with_derived", "derive_high_density_fraction",
    "derive_vessel_burden_gini",
    # paper_table
    "build_paper_table", "build_robust_sensitivity_table",
]
