# lm_lesion_anatomy

Exploratory sub-experiment that strengthens the `lm_isolated_low_burden`
finding by characterising the **anatomy** of the 13 left-main lesions
carried by the 10 displaced low-burden patients, and by testing whether
those lesions are morphologically distinguishable from LM lesions in
multivessel patients.

NOT part of the production pipeline. Does not produce any seam file
consumed by stages 1-8. Does not modify any decision. No
`verify_pipeline.py` coverage.

The four pre-registered analyses (see `plan.md`):

| Step | Question | Verdict mode |
|---|---|---|
| P1 | Multi-lesion structure: how many LM lesions per patient, what is the inter-lesion z-gap when there is more than one? | descriptive + PASS/FAIL on "effective burden concentration" |
| P2 | Cohort-relative LM axial localisation: do the 13 isolated LM lesions sit at the superior, mid, or inferior end of the cohort LM voxel distribution? | descriptive |
| P3 | Morphology equivalence: are isolated LM lesions distinguishable from LM lesions in multivessel patients on the 6 lesion-morphology features? | PASS/FAIL via Cliff's delta equivalence bound |
| P4 | Crude intra-lesion heterogeneity proxy: how does `max_hu - mean_hu_weighted` compare to LM lesions in multivessel patients and to non-LM lesions? | descriptive, labelled as crude |

Anatomical claims explicitly NOT made by this experiment (recorded as
limitations in `findings.md` and forwarded to paper Discussion):

* No stenosis or luminal narrowing measurement. COCA is non-contrast and
  vessel lumen is not segmentable; the closest crude proxy (calcium CSA
  vs literature LM diameter) is biased per-patient and not retained.
* No lesion-length / total-LM-length ratio. Anatomical LM length is not
  available; only calcified-LM extent is. A population-reference LM
  length (10-15 mm) was considered and rejected as a stenosis-style
  unwarranted approximation.

## Run

```
python experiments/lm_lesion_anatomy/run.py \
    --output-dir outputs/exploratory/lm_lesion_anatomy/
```

Tests live in `tests/`.
