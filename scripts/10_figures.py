#!/usr/bin/env python
"""Figure builder for the PrediCT v2 paper + supplementary.

Reads all stages 5-7 outputs + the lesion-morphology experiment outputs
and produces 11 figures (6 main + 5 supplementary) at 300 dpi PDF + PNG.

Outputs land in ``outputs/figures/``.

Usage:
    python scripts/10_figures.py
    python scripts/10_figures.py --only F1 F3      # only specific figures
    python scripts/10_figures.py --skip SF5        # skip the radar
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from predict.config import load_config
from predict.figures import style
from predict.figures.figure_1_pca_continuum import build as build_f1
from predict.figures.figure_2_gap_grid import build as build_f2
from predict.figures.figure_3_spatial_burden_split import build as build_f3
from predict.figures.figure_4_maturation_trajectory import build as build_f4
from predict.figures.figure_5_c8_anatomy import build as build_f5
from predict.figures.figure_6_phenotype_table import build as build_f6
from predict.figures.figure_s1_robust_discriminator_heatmap import build as build_s1
from predict.figures.figure_s2_monotonicity_scatter import build as build_s2
from predict.figures.figure_s3_directional_verdicts import build as build_s3
from predict.figures.figure_s4_hennig_stability import build as build_s4
from predict.figures.figure_s5_lesion_radar import build as build_s5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None,
                        help="only build the listed figures (e.g. F1 SF3)")
    parser.add_argument("--skip", nargs="+", default=None,
                        help="skip the listed figures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("figures")

    cfg = load_config(args.config)
    outputs = cfg.paths.outputs
    out_dir = outputs / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    reduce_dir = outputs / "06_reduce"
    analyse_dir = outputs / "07_analyse"
    lesion_morph_dir = outputs / "exploratory" / "lesion_morphology"
    features_csv = outputs / "03_features" / "features.csv"

    # (figure_id, stem, builder, kwargs)
    figures = [
        ("F1",  "figure_1_pca_continuum",            build_f1, {"reduce_dir": reduce_dir}),
        ("F2",  "figure_2_gap_grid",                 build_f2, {"reduce_dir": reduce_dir}),
        ("F3",  "figure_3_spatial_burden_split",     build_f3,
            {"reduce_dir": reduce_dir, "analyse_dir": analyse_dir, "features_csv": features_csv}),
        ("F4",  "figure_4_maturation_trajectory",    build_f4, {"lesion_morph_dir": lesion_morph_dir}),
        ("F5",  "figure_5_c8_anatomy",               build_f5,
            {"lesion_morph_dir": lesion_morph_dir,
             "lesions_csv": outputs / "03_features" / "lesions.csv",
             "cohort_metadata_csv": reduce_dir / "cohort_metadata.csv"}),
        ("F6",  "figure_6_phenotype_table",          build_f6, {"analyse_dir": analyse_dir}),
        ("SF1", "figure_s1_robust_discriminator_heatmap", build_s1, {"analyse_dir": analyse_dir}),
        ("SF2", "figure_s2_monotonicity_scatter",    build_s2, {"analyse_dir": analyse_dir}),
        ("SF3", "figure_s3_directional_verdicts",    build_s3, {"analyse_dir": analyse_dir}),
        ("SF4", "figure_s4_hennig_stability",        build_s4, {"reduce_dir": reduce_dir}),
        ("SF5", "figure_s5_lesion_radar",            build_s5, {"lesion_morph_dir": lesion_morph_dir}),
    ]

    selected = args.only or [fid for fid, *_ in figures]
    skipped = set(args.skip or [])

    n_ok = 0
    n_fail = 0
    for fid, stem, builder, kwargs in figures:
        if fid not in selected or fid in skipped:
            continue
        try:
            log.info("building %s (%s)...", fid, stem)
            fig = builder(**kwargs)
            paths = style.save(fig, out_dir, stem)
            log.info("  saved %s  +  %s",
                     paths["pdf"].name, paths["png"].name)
            n_ok += 1
        except Exception as e:
            log.error("  FAILED %s: %s", fid, e)
            log.debug(traceback.format_exc())
            n_fail += 1

    log.info("done: %d OK, %d FAILED. outputs in %s", n_ok, n_fail, out_dir)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
