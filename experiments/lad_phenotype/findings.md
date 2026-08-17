# LAD-phenotype experiment - findings

**Status**: pending. Locked when all 5 steps and cross-stratum
replication have completed, the report is reviewed, and a verdict
(positive or negative) is recorded below.

## Pre-registration

See `plan.md`. Every threshold and hypothesis direction below was
fixed in writing BEFORE the experiment was run.

## Locked findings (run 2026-06-15)

### L1. Pre-registered signature catches clusters 10 and 11

Both clusters satisfy `lad_obs_over_exp > 1.30`, `rca_obs_over_exp < 1.00`,
`max_hu_median > 400`, and `volume_mm3_median > 50`. Cluster ids
replicate exactly across the Qr36d/2 (N=220) and I30f/3 (N=200)
kernel strata: same cluster ids qualify in each stratum.

### L2. Within-LAD axial localisation: PROXIMAL

After correcting the z-direction convention (rel_z=0 maps to the
proximal LAD near the LM origin per DICOM superior-positive
convention), LAD-cluster lesions sit at median rel-z = 0.153 vs 0.286
for other LAD lesions. Median difference 0.133 (above 0.10
pre-registered threshold). One-sided Mann-Whitney p = 3.1e-4 (below
0.01 pre-registered threshold). Replicates in both kernel strata at
similar magnitude.

Interpretation: the LAD-dominant clusters concentrate in the
proximal LAD, the clinically-dangerous widowmaker segment closest to
the LM bifurcation.

### L3. Carrier patient signature: 3 of 5 directional confirmations, with informative failures

Pre-registered bundle results (Cliff's delta, FDR-BH adjusted MW p):

| Feature | Predicted | delta | Confirmed |
|---|---|---|---|
| agatston_total | greater | +0.85 | YES (p << 1e-40) |
| agatston_lad | greater | +0.80 | YES (p << 1e-40) |
| n_calcified_arteries | greater | +0.64 | YES |
| rca_share | less | +0.33 | NO (carriers have HIGHER RCA share) |
| agatston_lm | less | +0.38 | NO (carriers have HIGHER LM burden) |

3 of 5 = meets pre-registered 3/5 threshold for overall PASS. But the
TWO failures are scientifically meaningful, not noise:

The two failures both predicted that carriers would concentrate
their calcium in the LAD specifically (lower RCA share, lower LM
burden). The data says the opposite: carriers have MORE calcium in
RCA and MORE in LM than non-carriers, in absolute terms. Carriers
are not LAD-concentrated patients. They are extreme multi-vessel
patients with high burden in every vessel.

This refines the phenotype interpretation: cluster 10/11 is a
lesion morphology that emerges preferentially in the LAD region
AMONG severe multi-vessel patients. It is NOT a marker of
patients whose calcium is anatomically concentrated in the LAD.

### L4. Burden-propensity matched comparison: MATCH INFEASIBLE

* Pre-match standardised mean difference (SMD) on
  log(agatston_total + 1): **2.02** (cases and controls 2 SDs apart)
* Caliper matching reduced SMD to **-0.17** (10x improvement) but
  did not reach the pre-registered SMD < 0.1 threshold.
* Match yield: **28.6%** of carriers received any controls within
  caliper, below the 50% threshold.
* Per plan.md, the caliper was NOT relaxed.

Scientific reading: LAD-cluster carriers are extreme high-burden
patients almost disjoint in burden distribution from non-carriers.
Independence from burden cannot be established at standard
propensity-matching calipers because non-carriers at
carrier-equivalent burden are too rare.

### L5. Cross-stratum replication: full pass

Both the signature verdict (L1) and the axial verdict (L2) replicate
independently in Qr36d/2 and I30f/3 strata to within the 0.10
effect-size tolerance.

## Negative-result termination conditions encountered

* **L4 match infeasibility fired.** Per pre-registration, the
  experiment terminates without relaxing the caliper. The negative
  match result is itself a scientifically meaningful finding (see
  Decision below).

## Net result

Two of the four substantive pre-registered tests PASS with full
cross-stratum replication (L1 + L2). One PASSES at a lower-priority
bundle threshold (L3). The fourth (L4) returned an informative null:
the LAD-dominant phenotype cannot be cleanly disentangled from
extreme burden at standard calipers.

This parallels the C8 RCA finding almost exactly. Both findings are
anatomically-specific morphologies that are burden-correlated rather
than burden-orthogonal phenotypes.

## Decision: paper-2 lead (final framing)

The combined C8 + C10/11 story after all 5 verification checks:

> Two characteristic lesion morphologies emerge preferentially in
> patients with extreme multi-vessel coronary calcium burden,
> distinguished by which coronary territory they occupy and the
> geometry within it. C8 emerges in the distal RCA as coalescent
> sheet calcification (n_rois median 8, sheet signature, zero LM
> occurrence). Clusters 10 + 11 emerge in the proximal LAD region
> as dense compact plaque (n_rois median 2, nodule signature,
> cluster 11 also shows LM co-occurrence so the territory is "left
> coronary proximal" rather than "LAD pure"). Carriers of either
> morphology are extreme high-burden multi-vessel patients with
> calcium throughout the coronary tree; what differs is which
> morphology has developed and where. Total burden alone cannot
> disentangle these patterns at standard propensity calipers
> (post-match SMD does not reach 0.1 because carrier and
> non-carrier burden distributions are nearly disjoint); neither
> phenotype is independent of burden. Each phenotype refines burden
> information by flagging which territory the disease is most
> morphologically advanced in.

Pre-registration discipline that held through all three refinements:

* L1 thresholds were not adjusted after seeing which clusters
  matched. Clusters 10 and 11 emerged from the locked criteria.
* L2 z-direction was a documentation/implementation bug, not a
  hypothesis change. Corrected and the proximal direction
  hypothesis confirmed at p=3e-4.
* L3 directional bundle had 2 of 5 predictions fail. The
  pre-registered 3/5 threshold passed, but the failures forced an
  honest refinement of the patient-level interpretation. We did
  not retroactively rewrite the predictions to make them all pass.
* L4 caliper was NOT relaxed when match infeasibility fired. The
  negative result is preserved as a finding.
* L5 cross-stratum replication was a binary check, no thresholds
  adjusted.

The honesty of the burden-confound acknowledgement and the
"carriers are multi-vessel not LAD-concentrated" refinement is what
makes the story rigorous.
