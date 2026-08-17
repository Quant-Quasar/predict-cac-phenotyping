# Pipeline Walk-through

For per-stage deep dives (decisions, empirical results, edge cases), see `docs/modules/<stage>.md`:
- [`modules/io.md`](modules/io.md)
- [`modules/preprocess.md`](modules/preprocess.md)
- [`modules/features.md`](modules/features.md)
- (stability / reduce / analyse / validate — added per stage)



Each stage reads from the previous stage's outputs and writes its own. Stages are independent: any single stage can be rerun without recomputing upstream.

## 1. `io` — Load and discover

**Modules**
- `io.patient_discovery` — intersect `data/raw/` and `data/calcium_xml/`, apply exclusions.
- `io.dicom_loader` — load DICOM series, Z-sort, handle multi-series (D012 rule).
- `io.xml_parser` — parse Apple plist XML annotations.
- `io.spacing` — spacing metadata I/O.

**Output**: `outputs/01_manifest/manifest.csv` — one row per included patient with paths, scanner, kernel, slice count.

## 2. `preprocess` — Mask, resample, harmonise

**Modules**
- `preprocess.mask_builder` — XML polygons → 3D binary mask via Z-coordinate matching (primary) with ImageIndex math as fallback.
- `preprocess.resampling` — isotropic resampling of CT + mask.
- `preprocess.hu_handling` — clip, metal artifact detection.
- `preprocess.kernel_harmonise` — ComBat or within-kernel z-score across Qr36d / I30f.

**Output**: `outputs/02_preprocessed/{pid}_ct.npy`, `outputs/02_preprocessed/{pid}_mask.npy`, `outputs/02_preprocessed/spacing.json`.

## 3. `features` — Extract

**Modules**
- `features.radiomics` — PyRadiomics wrapper, all 7 IBSI families.
- `features.agatston` — Agatston scoring from XML.
- `features.lesion_ccl` — 3D lesion identification by BFS grouping.
- `features.spatial` — lesion count, inter-lesion distances, diffusivity, Gini, per-artery aggregates.
- `features.density_tiers` — per-artery ROI counts in Agatston tiers.
- `features.per_artery` — per-territory mask building and aggregates.

**Output**: `outputs/03_features/features.csv`, `outputs/03_features/agatston.csv`.

## 4. `stability` — Perturb and gate

**Modules**
- `stability.perturbations` — 14 perturbation generators (rotations, translations, noise).
- `stability.icc` — ICC(3,1) absolute-agreement computation.

**Output**: `outputs/04_perturbations/<type>/features.csv` (per perturbation), `outputs/04_perturbations/icc.csv`.

## 5. `reduce` — Variance, redundancy, dimensionality

**Modules**
- `reduce.variance_filter` — drop variance < threshold.
- `reduce.r2_clustering` — hierarchical R² clustering, representative selection.
- `reduce.pca` — PCA on cleaned features.
- `reduce.wgcna` — optional module eigengenes (disconnected from primary path).

**Output**: `outputs/05_cleaned/features.csv`, `outputs/06_reduced/pca_scores.csv`.

## 6. `analyse` — Test and characterise

**Modules**
- `analyse.clusterability` — dip test, gap statistic with k=1, density-based clusterability — runs **before** clustering.
- `analyse.residualise` — burden-residualised feature space for sub-burden discovery.
- `analyse.consensus` — Monti 2003 consensus clustering, gated on clusterability evidence.
- `analyse.manifold` — continuum characterisation (only if clustering not warranted).

**Output**: `outputs/07_clusterability/`, `outputs/08_clusters/`.

## 7. `validate` — Endpoint-free framework

**Axes**
1. Agatston cross-tab (ARI, NMI, Fisher per cell).
2. Cluster quality (silhouette, Davies-Bouldin, Calinski-Harabasz).
3. Perturbation ICC reproducibility (already gated upstream, reported here).
4. Clinical archetype alignment (spotty / dense / diffuse).
5. **Cross-kernel replication** (Qr36d ↔ I30f) — built-in natural experiment.

Plus a correctness gate:

- `validate.xml_roundtrip` — rasterise polygon, recompute Mean/Max/Min HU from voxels, compare against XML-stored stats.

**Output**: `outputs/09_validation/validation_report.md`.
