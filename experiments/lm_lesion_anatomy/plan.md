# Pre-registration: left-main lesion anatomy in the isolated-LM low-burden subgroup

**Locked**: 2026-06-26
**Status**: pre-systematic-query. Every threshold and hypothesis direction
below is fixed BEFORE the modules in this folder are executed against real
cohort outputs. The 10 displaced patients and their 13 LM lesions are
already known from `experiments/lm_isolated_low_burden/findings.md`, which
locked 2026-06-15. This experiment uses the same 10 patients as its
treatment group and the LM lesions of all multivessel patients as its
reference group. No further patient identification step is performed here.

If a threshold is later found to disqualify a clinically obvious case or
to admit a clinically irrelevant case, the failure mode is documented in
`findings.md` and the threshold is NOT relaxed to produce a positive
verdict.

---

## Inputs (fixed, frozen)

| Input | Source | Rows used |
|---|---|---|
| The 10 displaced low-burden pids | `outputs/exploratory/lm_isolated_low_burden/displaced_patients.csv` | {198, 105, 200, 170, 290, 21, 43, 311, 427, 426} |
| Per-lesion morphology and geometry | `outputs/03_features/lesions.csv` | all rows with `vessel == "LM"` |
| Lesion-cluster labels | `outputs/exploratory/lesion_morphology/lesion_cluster_labels.csv` | joined on `(pid, roi_key)` |
| Patient-level burden | `outputs/03_features/features.csv` | `lesion_count_*`, `agatston_total`, `n_calcified_arteries` |
| Spacing | `outputs/02_preprocessed/spacing.json` | slice thickness (3.0 mm fixed) |

The full list of LM lesions partitions cleanly into three groups:

| Group | Definition | Approx. N (LM lesions) |
|---|---|---|
| **Isolated-LM** | LM lesions in the 10 displaced patients | 13 |
| **Multivessel-LM** | LM lesions in patients with `n_calcified_arteries > 1` AND patient NOT in displaced-10 | ~265 (from per-vessel prevalence: 153 LM-positive patients × ~1.8 LM lesions/patient minus 13 isolated) |
| **Other-vessel** (used in P4 only) | All non-LM lesions across the cohort | ~3,000 |

The displaced-10 and multivessel-LM groups are disjoint by construction
(every displaced-10 patient has `n_calcified_arteries == 1`, per
`lm_isolated_low_burden/findings.md` Result 1).

## P1. Multi-lesion structure of the displaced-10 (Step 1, run.py)

The 10 displaced patients carry 13 LM lesions total: 8 patients have 1
lesion each, pid 427 has 2, pid 290 has 3.

For each multi-lesion patient (290, 427), report:

* Number of LM lesions
* Z-positions of each lesion (centroid_mm.z, mm)
* Inter-lesion z-gap (mm) between consecutive z-sorted lesions
* Volume-weighted contiguous extent: total z-range covered by all LM
  lesions, mm

### Pre-registered concentration criterion

A patient is **z-concentrated** iff the volume-weighted contiguous extent
of all their LM lesions is ≤ 10 mm. This bound is one slice-thickness
above the clinical short-end of the LM (10-15 mm anatomical length, per
Lin 2022 and Hoori 2024). Two lesions sitting within 10 mm of each other
behave biologically like one segmental burden, not like multifocal
disease.

**P1 verdict (descriptive + PASS criterion)**:

* **PASS**: both multi-lesion patients (290, 427) are z-concentrated.
* **FAIL**: at least one multi-lesion patient has volume-weighted extent
  > 10 mm. Implication: the lesions are anatomically separated and the
  patient is multifocal-LM, not segmentally-LM.

The 8 single-lesion patients have z-concentration of 0 by definition and
contribute nothing to the PASS/FAIL.

### Rationale

If both multi-lesion patients are z-concentrated, the experiment's later
descriptive language can collapse the 13 lesions into 10 "patient-level
LM disease zones," treating the multi-lesion patients as having one
extended lesion rather than independent foci. If either fails, the
descriptive language must distinguish "13 lesions across 10 patients,
including two patients with multifocal LM disease."

## P2. Cohort-relative LM axial localisation (Step 2, run.py)

The 10 displaced patients have no LAD or LCx calcification, so per-patient
anatomical landmarks (bifurcation point, LAD origin) are not available.
The cohort-wide LM voxel distribution is used as the reference instead.

### Reference construction

Across all 153 LM-positive patients (whether isolated or multivessel),
gather the z-positions of every LM lesion centroid (in mm, using
`centroid_mm.z` from `lesions.csv`). Compute the 33rd and 67th percentiles
of this distribution. Define:

| Tercile | Z range |
|---|---|
| **Superior** (ostial-leaning) | `z >= P67` |
| **Mid** | `P33 <= z < P67` |
| **Inferior** (bifurcation-leaning) | `z < P33` |

The clinical interpretation rests on DICOM patient-coordinate convention
(superior = +z, per the lad_phenotype/plan.md note 2026-06-15): the LM
runs from its aortic-root ostium (superior) to its LAD/LCx bifurcation
(inferior). Superior LM voxels are therefore ostial-leaning; inferior LM
voxels are bifurcation-leaning.

### Pre-registered question

How are the 13 isolated-LM lesions distributed across the three terciles?

**Descriptive only, no PASS/FAIL.** The expected distribution under the
null hypothesis "isolated LM disease is uniformly distributed along the
LM axis" is 4-5 per tercile. Substantial departures are reported as
descriptive observations and forwarded to the Discussion section, NOT
used to make biological claims here.

### Rationale for descriptive framing

The clinical literature on isolated LM calcification offers competing
predictions: ostial bias (bicuspid valve association, aortic root
pulsation), bifurcation bias (low-shear flow-divider region), or no bias
(degenerative process). With N = 13 lesions, the experiment is
underpowered to discriminate between them. Reporting the bin counts and
labelling them descriptive is the honest disclosure.

### Caveat written into the methods section

The cohort LM voxel reference is biased toward multivessel patients
(140 of 153 LM-positive patients have multivessel disease), whose LM
calcium distribution may differ systematically from isolated-LM disease.
This caveat is recorded in `findings.md` Methods and the Discussion.

## P3. LM morphology equivalence test (Step 3, run.py)

The load-bearing analytical step. This addresses Tier 2 #6 from the
strengthening discussion: are isolated-LM lesions morphologically the
same biological entity as multivessel-LM lesions, just appearing in
different total-burden contexts?

### Features and groups

* Group A (treatment): 13 isolated-LM lesions
* Group B (reference): all LM lesions in multivessel patients

The 6 lesion-morphology features (matching `experiments/lesion_morphology/`):

* `log(volume_mm3 + 1)`
* `log(total_area_mm2 + 1)`
* `mean_hu_weighted`
* `max_hu`
* `log(n_rois + 1)`
* `log(volume_mm3 / total_area_mm2 + 1)` (the "thickness" proxy)

### Pre-registered equivalence test

Equivalence (not difference) is the hypothesis. Standard significance
testing favours rejecting null on small N (here Group A is N = 13, so
power to detect any real difference is limited and a failure to reject
is uninformative).

For each of the 6 features:

* Compute Cliff's delta (Group A vs Group B)
* Bootstrap 95% CI on Cliff's delta (1000 resamples, seed = 42)

**Equivalence bound** (locked):

* `|Cliff's delta| < 0.20` AND `upper bound of |CI| < 0.30`

The 0.20 threshold matches D023's robust-discriminator gate (any feature
with `|delta| >= 0.20` is considered a discriminator there; symmetric
equivalence threshold here). The 0.30 CI ceiling prevents declaring
equivalence on wide CIs that merely happen to straddle zero.

**P3 verdict**:

* **PASS (equivalence)**: at least 5 of 6 features satisfy both the
  point-estimate bound and the CI bound. Conclusion: isolated-LM lesions
  are morphologically indistinguishable from multivessel-LM lesions on
  the standard lesion-morphology features. The LM-isolated finding is
  reframed as "same lesion type, wildly different total disease context."
* **FAIL (non-equivalent on ≥ 2 features)**: report which features fail
  and their effect direction. Conclusion: isolated-LM lesions are a
  morphologically distinct entity from multivessel-LM lesions, opening
  a separate paper-2 line.
* **AMBIGUOUS (1 feature fails)**: report the failing feature, flag for
  Discussion. No reframing claim made.

### Rationale

This is the strongest available test of the user's Tier 2 #6 hypothesis.
Equivalence framing (with explicit bound) rather than null-acceptance
prevents the "absence of evidence as evidence of absence" pitfall on
small-N comparisons.

## P4. Crude intra-lesion heterogeneity proxy (Step 4, run.py)

Tier 2 #7 from the strengthening discussion. Honest disclosure: lesion-
level PyRadiomics texture is not in the pipeline (texture is computed on
the union mask, not per-lesion). The best proxy available from
`lesions.csv` is the spread between the per-lesion `max_hu` (raw maximum)
and the area-weighted `mean_hu_weighted`.

### Proxy definition

For each lesion: `het_proxy = max_hu - mean_hu_weighted` (HU units).

A high value means the lesion has a high-density spike against a softer
background (suggestive of a dense core in a less-dense matrix). A low
value means the lesion is approximately uniform.

### Distributions reported (descriptive)

Median + IQR of `het_proxy` for:

* Isolated-LM lesions (N = 13)
* Multivessel-LM lesions
* All non-LM lesions (LAD + LCx + RCA combined)

### Pre-registered descriptive characterisation, no PASS/FAIL

The proxy is too crude to support a biological claim. Report:

* Whether the isolated-LM median is above, below, or within the IQR of
  the multivessel-LM median
* Whether isolated-LM lesions cluster in any extreme decile

### Caveat written into findings.md

The proxy is a single-number summary that conflates:

* True intra-lesion density heterogeneity (rim vs core),
* Density-tier composition within the lesion (number of d3/d4 voxels),
* Per-voxel HU calibration noise (5-10 HU per the perturbation set).

A true rim-vs-core analysis requires per-lesion PyRadiomics extraction,
which is new infrastructure not built in this experiment. The proxy is
labelled crude in every place it appears and is NOT used to make
biological claims.

## Termination conditions (no negotiation)

| Condition | Outcome |
|---|---|
| Either multi-lesion patient (290, 427) fails P1 z-concentration (>10 mm spread) | P1 reframes language to "multifocal LM in patient X" rather than "segmental." No reframing of the overall LM-isolated finding. |
| < 5 of 6 morphology features satisfy P3 equivalence bound | P3 FAIL. Reframe LM-isolated cases as morphologically distinct entity. Paper-2 lead. |
| Exactly 1 of 6 morphology features fails P3 equivalence bound | P3 AMBIGUOUS. Report and flag in Discussion. No equivalence claim. |
| All four steps complete with PASS or descriptive verdicts | Compile `findings.md` with the four results sections + the limitations block (stenosis, lesion-length ratio, true rim-vs-core). Forward all three limitations to paper Discussion. |

## Inputs the experiment does NOT use (recorded so the absence is intentional)

| Input | Why excluded |
|---|---|
| Vessel lumen segmentation | Not available in COCA (non-contrast). Stenosis claim not made. |
| Anatomical LM length per patient | Not available; only calcified-LM extent observable. Lesion-length-ratio claim not made. |
| Per-lesion PyRadiomics texture | Not in pipeline; texture is union-mask only. True rim-vs-core claim not made. |
| Clinical outcomes or follow-up | Not in COCA. |
| Demographics (age, sex, race) | Not in COCA. |

## Hash of this document at lock time

```bash
sha256sum experiments/lm_lesion_anatomy/plan.md
```

The hash will be recorded in `run_header.json` at first successful run so
any post-hoc edit is detectable.
