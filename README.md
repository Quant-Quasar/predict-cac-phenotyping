# PrediCT — Coronary Artery Calcium Radiomics Phenotyping

Reproducible, endpoint-free discovery of coronary artery calcium (CAC) phenotypes from
non-contrast gated cardiac CT. Built for **Google Summer of Code 2026 (ML4SCI)** on the public
**Stanford COCA** cohort (449 scans with expert polygon calcium annotations; 444 after quality
exclusions).

The central question: *do calcified coronary arteries fall into discrete morphological
phenotypes, or a single continuum?* — answered without using any clinical outcome label, by
combining radiomics feature extraction with unsupervised clustering and stability-based validation.

## Key result

Across three clustering algorithms, three feature spaces, and two scanner-stratified subcohorts,
gap-statistic curves rise monotonically with no phenotype elbow: **the cohort is a single
calcium-burden continuum rather than a set of discrete phenotypes.** Even a feature subspace
constructed to be orthogonal to burden re-expresses the burden axis. The finding is reported with
effect sizes and multiple-testing correction throughout, and replicates under a voxel-size
sensitivity probe and leave-k-out cross-validation.

## Pipeline

```
io → preprocess → features → stability → reduce → discover → analyse → validate
```

| Stage | Purpose |
|---|---|
| `io` | DICOM series loading, calcium-XML parsing, patient discovery, multi-series resolution |
| `preprocess` | Z-coordinate ROI→slice matching, 3D mask rasterisation, resampling to 0.5×0.5×3.0 mm, HU handling, XML-stat round-trip correctness gate |
| `features` | Agatston scoring, 3D lesion grouping, per-vessel and spatial features, density tiers, PyRadiomics IBSI texture/shape |
| `stability` | Deterministic CT perturbations and ICC(3,1) reliability gating of features |
| `reduce` | Variance filtering, ComBat kernel harmonisation, Spearman-r² redundancy clustering, PCA |
| `discover` | Cluster-tendency (Hopkins), gap statistic, consensus clustering, bootstrap stability |
| `analyse` | Per-cluster characterisation, burden-orthogonality tests, pre-registered directional hypotheses, cross-cohort consistency |
| `validate` | External scanner holdout, leave-k-out cross-validation, cross-cohort ARI consolidation |

## Repository layout

```
configs/      YAML configuration (paths, seeds, thresholds, PyRadiomics params, perturbations)
src/predict/  Library code, one subpackage per stage
scripts/      Numbered orchestration scripts (one per stage)
tests/        pytest suite mirroring src/predict/
experiments/  Pre-registered exploratory analyses (each with its own plan + findings)
decisions/    Design-decision registry (one markdown per decision)
docs/         Per-stage documentation, dataset report, reproducibility notes
figures/      Publication figures
```

`data/` and `outputs/` are not tracked (the dataset is large and licensed separately).

## Data

This repository contains **code only**. The Stanford COCA (Coronary Calcium) dataset is publicly
available from the Stanford AIMI shared datasets and is not redistributed here. Place the DICOM
series and calcium XML annotations at the paths configured in `configs/default.yaml`
(`data/raw/`, `data/calcium_xml/`) to run the pipeline end to end.

## Reproducibility

- Every stochastic step takes an explicit seed; parallel bootstraps use per-iteration derived
  seeds so results are byte-identical regardless of worker count.
- Stages communicate through versioned CSV/NPY seams and can be rerun independently.
- 716 unit tests (`pytest tests/`), plus `verify_pipeline.py` runs 114 numerical-invariant checks
  over the produced outputs.
- Every design choice is recorded as an individual entry in `decisions/`.

## Getting started

```bash
# Python 3.8, conda recommended
conda install -c radiomics -c conda-forge pyradiomics
pip install -r requirements.txt
pip install -e .

pytest tests/            # run the test suite
./run_pipeline.sh        # end-to-end run (requires the dataset in place)
```

## License

MIT — see [LICENSE](LICENSE).
