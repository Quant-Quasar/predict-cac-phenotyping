"""Stage 7 helper: re-derive the two D019 derived features on RAW values.

Background:

  Stage 5's ``prepare_matrix.py`` creates two derived features
  (``high_density_fraction`` and ``vessel_burden_gini``) from the canonical
  per-vessel inputs. Those derived columns live in
  ``outputs/06_reduce/prepared_matrix.csv`` AFTER D019 z-scoring; they are
  NOT written to the raw ``outputs/03_features/features.csv``.

  Stage 7 reads RAW values from features.csv (per D023's clinical-
  interpretability rule). If the derived columns are needed at raw scale
  (e.g., for the D025 directional hypothesis on ``vessel_burden_gini``),
  this helper re-derives them in-place. The formulas are byte-identical
  to ``predict.reduce.prepare_matrix.compute_high_density_fraction`` and
  ``predict.reduce.prepare_matrix.compute_vessel_burden_gini``.

Conventions (locked, match the stage-5 originals):

* ``high_density_fraction = (d3 + d4 count) / (d1 + d2 + d3 + d4 count)``,
  summed across all 4 vessels. Patients with zero total tier-bin count
  return 0.0 (no division by zero).

* ``vessel_burden_gini`` = standard Gini coefficient across the 4 per-vessel
  Agatston values, BUT with two convention-driven floors:

      * patient with 0 vessels with calcium > 0  ->  Gini = 0.0
      * patient with 1 vessel  with calcium > 0  ->  Gini = 0.0
      * patient with 2+ vessels                  ->  standard Gini formula

  This matches the stage-5 implementation. Patients with a single
  calcified artery therefore have ``vessel_burden_gini = 0``, NOT 0.75.
  This is the convention chosen at stage 5 and we do not change it here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


VESSELS: tuple[str, ...] = ("lad", "rca", "lcx", "lm")


def derive_high_density_fraction(df: pd.DataFrame) -> pd.Series | None:
    """(d3 + d4) / (d1 + d2 + d3 + d4) summed across all 4 vessels.

    Returns ``None`` if any of the 16 required ``n_rois_dT_VESSEL`` columns
    is missing. Patients with zero total tier-bin count receive 0.0
    (avoids division by zero).
    """
    d12 = [f"n_rois_d{t}_{v}" for t in (1, 2) for v in VESSELS]
    d34 = [f"n_rois_d{t}_{v}" for t in (3, 4) for v in VESSELS]
    needed = d12 + d34
    if not all(c in df.columns for c in needed):
        return None
    d12_sum = df[d12].sum(axis=1)
    d34_sum = df[d34].sum(axis=1)
    total = d12_sum + d34_sum
    out = pd.Series(0.0, index=df.index)
    mask = total > 0
    out.loc[mask] = d34_sum.loc[mask] / total.loc[mask]
    return out


def derive_vessel_burden_gini(df: pd.DataFrame) -> pd.Series | None:
    """Gini coefficient across the 4 per-vessel Agatston values.

    Convention:
      - 0 calcified vessels: Gini = 0.0 (no burden, no inequality)
      - 1 calcified vessel:  Gini = 0.0 (single value, no inequality)
      - 2+ calcified vessels: standard Gini formula

    Returns ``None`` if any of the 4 ``agatston_VESSEL`` columns is missing.
    """
    cols = [f"agatston_{v}" for v in VESSELS]
    if not all(c in df.columns for c in cols):
        return None
    out = pd.Series(0.0, index=df.index)
    arr = df[cols].to_numpy(dtype=float)
    for i in range(arr.shape[0]):
        vals = arr[i]
        vals = vals[vals > 0]
        n = len(vals)
        if n < 2:
            out.iloc[i] = 0.0
            continue
        sorted_v = np.sort(vals)
        total = sorted_v.sum()
        cum = float(np.sum(np.arange(1, n + 1) * sorted_v))
        gini = (2.0 * cum) / (n * total) - (n + 1) / n
        out.iloc[i] = float(gini)
    return out


def augment_raw_with_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``high_density_fraction`` and
    ``vessel_burden_gini`` appended (if their inputs are available).

    Existing columns of those names are PRESERVED (we do not overwrite).
    This is defensive in case a future stage 3 starts writing them.
    """
    out = df.copy()
    if "high_density_fraction" not in out.columns:
        derived = derive_high_density_fraction(out)
        if derived is not None:
            out["high_density_fraction"] = derived
    if "vessel_burden_gini" not in out.columns:
        derived = derive_vessel_burden_gini(out)
        if derived is not None:
            out["vessel_burden_gini"] = derived
    return out
