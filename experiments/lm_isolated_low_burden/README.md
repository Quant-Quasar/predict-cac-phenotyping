# Exploratory: isolated LM calcification at low total Agatston burden

**Status**: exploratory; outside the production pipeline.

## What this finding is

10 patients in the cohort (2.4% of the eligible 420) sit in the
bottom Agatston tertile but project into a region of PCA feature
space dominated by left-main-vessel loadings (PC2 > 2.5). All 10
carry isolated single-vessel LM calcification; the pattern
replicates independently in both kernel strata. Their LM lesions
are moderate-to-dense (median max-HU 335, range 301-604), not
soft microspots. They sit in lesion-morphology clusters distinct
from the high-burden LAD/LM compact-dense cluster.

Conventional Agatston-based risk stratification places these
patients in the lowest-risk category despite their disease
occupying the highest-anatomic-risk segment of the coronary tree.

## See also

* `findings.md` - full report with all numbers, statistical tests,
  cross-stratum replication, lesion-cluster overlap analysis,
  per-patient profiles, biological interpretation, limitations,
  and reproducibility queries.
* `../lad_phenotype/findings.md` - companion experiment
  characterising the LAD/LM-region compact-dense phenotype in the
  HIGH-burden tail. The two findings sit at opposite ends of the
  burden spectrum and are biologically distinct entities.
* `../lesion_morphology/findings.md` - source of the lesion-level
  cluster labels used to verify that displaced-patient LM lesions
  are in clusters 0/2/3 rather than the LAD-dominant 10/11.

## Structure

```
experiments/lm_isolated_low_burden/
├── README.md       (this file)
├── plan.md         pre-registration of thresholds and hypotheses (P1-P5)
├── run.py          systematic analysis: displaced identification,
│                   Fisher test, cross-stratum replication,
│                   density profile, cluster overlap, Wilson CIs
├── finalise.py     bundle outputs into a single report.txt
├── findings.md     locked findings narrative with all numbers
└── tests/
    ├── __init__.py
    └── test_run.py 16 unit tests on the analysis helpers
```

## How to run

```bash
# Smoke tests first
pytest experiments/lm_isolated_low_burden/tests/ -v

# Systematic analysis
python experiments/lm_isolated_low_burden/run.py

# Bundle + print report
python experiments/lm_isolated_low_burden/finalise.py
```

Wall-clock ~5 seconds for the whole pipeline.

Outputs land at `outputs/exploratory/lm_isolated_low_burden/`:

| File | Content |
|---|---|
| `displaced_patients.csv` | per-patient table for the 10 displaced patients |
| `full_cohort_displacement.csv` | every cohort patient with their PC scores + displaced flag |
| `fisher_test.json` | P2 contingency table + Fisher result + PASS verdict |
| `cross_stratum.json` | per-stratum LM rates + Wilson CIs + PASS verdicts |
| `density_profile.json` | tier breakdown of displaced patients' LM lesions + framing |
| `cluster_overlap.json` | every LM lesion of every displaced patient + cluster id + LAD-cluster overlap |
| `summary.json` | top-level PASS/FAIL across all four criteria |
| `run_header.json` | git hash + libs + seam SHAs + plan.md SHA |
| `report.txt` | human-readable narrative produced by finalise.py |

## Discipline notes

* The displacement criterion (PC1 > 0 OR PC2 > 2.5) was inspected
  on the PCA scatter prior to lock; the systematic query was run
  only once after the criterion was written to `plan.md`. No
  iteration on threshold after seeing the systematic result.
* All five test criteria (P1 - P5) and their pass thresholds are
  locked in `plan.md` with a SHA recorded in `run_header.json`.
* No modification to the production pipeline, no decision doc,
  no CLAUDE.md tracking, no verify_pipeline.py coverage.
