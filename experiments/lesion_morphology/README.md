# Exploratory: lesion-level morphology clustering

**Status**: exploratory; outside the production pipeline. Does not produce
any seam file consumed by stages 1-8. Does not modify any decision.

## Question

Stage 7 (analyse) showed the patient-level partition is a burden continuum,
not a discrete topology phenotype. If morphology phenotypes exist at all
(dense nodular vs soft speckled vs sheet calcification, per radiology
intuition), they should be visible at the LESION level, not at the patient
level. Patient-level aggregation could be masking lesion-type structure by
averaging over heterogeneous within-patient mixtures.

This experiment tests two related claims:

1. **Cluster tendency at the lesion level**: are there discrete morphology
   modes in the 3179-lesion population?
2. **Patient-level signatures**: do patients have characteristic
   distributions over the lesion clusters that recover the focal vs
   distributed topology phenotype the patient-level test refuted?

## What this experiment is NOT

* NOT a new pipeline stage.
* NOT a decision (no D029+).
* NOT something CLAUDE.md or the test suite track.
* NOT a re-run or re-design of stages 5, 6, or 7.

If the experiment produces a publication-grade finding, the paper can
reference it as a lesion-level supplementary analysis. The production
pipeline is unchanged either way.

## Inputs

| File | Source | Used for |
|---|---|---|
| `outputs/03_features/lesions.csv` | stage 3 audit file | the 3179 lesions to cluster |
| `outputs/03_features/features.csv` | stage 3 | per-patient agatston_total + kernel for post-hoc stratification |
| `outputs/06_reduce/cluster_labels_spatial_k2.csv` | stage 6 (full cohort) | patient-level focal/diffuse labels for cross-tab |
| `outputs/06_reduce/cohort_metadata.csv` | stage 5 | per-patient kernel + low_burden_flag |

## Outputs

All under `outputs/exploratory/lesion_morphology/`:

| File | Content |
|---|---|
| `lesion_features.csv` | per-lesion engineered morphology features (raw + transformed) |
| `hopkins.json` | clusterability verdict on the morphology subspace |
| `gap_statistic.json` | gap curves and selected k |
| `lesion_cluster_labels.csv` | per-lesion cluster assignment at gap-selected k |
| `cluster_profiles.csv` | per-cluster median + IQR of every morphology feature |
| `cluster_vessel_distribution.csv` | per-cluster lesion count per vessel (anatomical stratification) |
| `hennig_stability.json` | Hennig median Jaccard per cluster |
| `patient_lesion_signatures.csv` | per-patient cluster-fraction vector (over the lesion cluster ids) |
| `patient_phenotype_crosstab.csv` | crosstab of patient-level focal/diffuse vs dominant lesion cluster |
| `report.txt` | human-readable summary printed by the script |
| `run_header.json` | git hash + library versions + CLI args + seam SHAs |

## How to run

```bash
python experiments/lesion_morphology/run.py            # full experiment
python experiments/lesion_morphology/run.py --k 3      # force k=3 (3 morphology types prior)
python experiments/lesion_morphology/run.py --no-hennig  # skip bootstrap stability for a faster smoke
```

Wall-clock: ~2 minutes at default parameters (gap statistic on 3179
lesions + 100 Hennig bootstraps) on the 80-core remote box.

## Methodological consistency with the production pipeline

This experiment reuses the production-tested helpers from
`predict.discover`:

* `assess_clusterability` for Hopkins (k=2-for-both convention,
  pyclustertend rule)
* `gap_statistic` with the Maitra-Ramler 1-SE-from-argmax rule
* `fit_cluster` for k-means / Ward / GMM
* `hennig_clusterboot` for stability

This keeps the lesion-level result methodologically comparable to the
patient-level result from stages 6 and 7. If the lesion-level Hopkins
also shows H ~ 0.7 and the lesion-level gap is also monotonic to k_max,
the conclusion "no discrete phenotypes" extends below the patient
resolution. If the lesion level shows a clear elbow and high Hennig
stability, that is a genuine morphology phenotype claim worth pursuing.
