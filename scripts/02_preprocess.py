#!/usr/bin/env python
"""Stage 2 — preprocess.

Reads ``outputs/01_manifest/manifest.csv`` and for each patient:

  1. Loads DICOM (Z-ascending, D012 multi-series rule).
  2. Parses XML.
  3. Builds the 3D mask via Z-coordinate matching (D001).
  4. Runs the XML stat round-trip (D002); Max-exact gate.
  5. Resamples CT + mask to the target voxel grid (D005).
  6. Clips HU to ``[clip_min, clip_max]`` (D003); flags metal artefacts.
  7. Saves ``{pid}_ct.npy`` (int16 HU) and ``{pid}_mask.npy`` (uint8).

Outputs (under ``outputs/02_preprocessed/``):
  - ``{pid}_ct.npy``, ``{pid}_mask.npy``
  - ``spacing.json``  (target voxel grid)
  - ``preprocess_report.csv``  (per-patient counts, roundtrip, metal flag)
  - ``xml_roundtrip.csv``  (per-ROI deltas across the whole cohort)

Usage::

    python scripts/02_preprocess.py --n-workers 16
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from predict.config import Config, load_config
from predict.io import (
    load_patient_dicom,
    parse_calcium_xml,
    save_spacing_metadata,
)
from predict.preprocess import (
    build_3d_mask,
    clip_hu,
    flag_metal_artifact,
    mask_to_sitk,
    resample_to_target,
)
from predict.validate.xml_roundtrip import (
    failed_roi_ids,
    pass_rate,
    trips_to_rows,
    xml_roundtrip_check,
)


def _quality_label(rate: float) -> str:
    if rate >= 1.0:
        return "ok"
    if rate >= 0.95:
        return "partial"
    return "poor"


def _process_one(
    pid: str,
    data_raw_root: str,
    xml_dir: str,
    out_dir: str,
    target_spacing: tuple,
    clip_min: int,
    clip_max: int,
    metal_threshold: int,
    tolerance_mm: float,
) -> dict:
    """Worker: preprocess one patient and write its .npy files."""
    try:
        data_root = Path(data_raw_root).parent  # data/raw → data
        loaded = load_patient_dicom(pid, data_root)
        parse_result = parse_calcium_xml(pid, Path(xml_dir))
        ct_native = sitk.GetArrayFromImage(loaded.ct_sitk)

        # Round-trip first — uses CT + parsed XML directly, no mask required.
        trips = xml_roundtrip_check(
            parse_result, ct_native, loaded.slice_positions,
            tolerance_mm=tolerance_mm,
        )
        rt_rate = pass_rate(trips)
        rt_fail_count = sum(1 for t in trips if not t.passes and t.matched_via != "dirty")
        excluded = failed_roi_ids(trips)

        # Build mask EXCLUDING ROIs that failed the hard gate (D002).
        mask_arr, build_report = build_3d_mask(
            parse_result, loaded,
            tolerance_mm=tolerance_mm,
            excluded_roi_ids=excluded,
        )

        # Integrity check: a calcium-positive cohort should have non-empty masks.
        if mask_arr.sum() == 0:
            return {
                "status": "error",
                "pid": pid,
                "error": "empty_mask_after_build",
            }

        mask_sitk = mask_to_sitk(mask_arr, loaded.ct_sitk)
        ct_resampled, mask_resampled = resample_to_target(
            loaded.ct_sitk, mask_sitk, target_spacing,
            ct_default_value=float(clip_min),
        )

        ct_arr = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
        ct_arr = clip_hu(ct_arr, clip_min, clip_max).astype(np.int16)
        mask_resampled_arr = sitk.GetArrayFromImage(mask_resampled).astype(np.uint8)

        if mask_resampled_arr.sum() == 0:
            return {
                "status": "error",
                "pid": pid,
                "error": "empty_mask_after_resample",
            }

        metal_flag = flag_metal_artifact(ct_arr, mask_resampled_arr, metal_threshold)

        out = Path(out_dir)
        np.save(out / f"{pid}_ct.npy", ct_arr)
        np.save(out / f"{pid}_mask.npy", mask_resampled_arr)

        return {
            "status": "ok",
            "pid": pid,
            "kernel": loaded.kernel,
            "scanner_model": loaded.scanner_model,
            "n_slices_native": loaded.n_slices,
            "n_slices_resampled": int(ct_arr.shape[0]),
            "mask_voxels": int(mask_resampled_arr.sum()),
            "roundtrip_pass_rate": float(rt_rate),
            "roundtrip_fail_count": int(rt_fail_count),
            "roundtrip_quality": _quality_label(rt_rate),
            "n_rois_active": parse_result.n_active_rois,
            "n_dirty_names": len(parse_result.dirty_vessel_names),
            "n_rasterised": build_report.n_rasterised,
            "n_excluded_by_roundtrip": build_report.n_excluded_by_roundtrip,
            "n_skipped_dirty": build_report.n_skipped_dirty,
            "n_skipped_no_match": build_report.n_skipped_no_match,
            "n_matched_by_fallback": build_report.n_matched_by_fallback,
            "metal_artifact_flag": bool(metal_flag),
            "trips": trips_to_rows(trips),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "pid": pid, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n-workers", type=int, default=min(16, mp.cpu_count()))
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N patients (for smoke runs)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("preprocess")

    cfg: Config = load_config(args.config)
    manifest_path = cfg.paths.outputs / "01_manifest" / "manifest.csv"
    if not manifest_path.exists():
        log.error("Manifest missing: %s. Run scripts/01_discover.py first.", manifest_path)
        return 2

    manifest = pd.read_csv(manifest_path, dtype={"pid": str})
    pids = manifest["pid"].astype(str).tolist()
    if args.limit:
        pids = pids[: args.limit]

    out_dir = cfg.paths.outputs / "02_preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_spacing_metadata(out_dir / "spacing.json", cfg.resample.target_spacing)

    log.info(
        "Preprocessing %d patients | workers=%d | target=%s",
        len(pids), args.n_workers, cfg.resample.target_spacing,
    )

    records: list[dict] = []
    trip_rows: list[dict] = []
    errors: list[dict] = []

    if args.n_workers == 1:
        results_iter = (
            _process_one(
                pid,
                str(cfg.paths.data_raw),
                str(cfg.paths.data_xml),
                str(out_dir),
                cfg.resample.target_spacing,
                cfg.hu.clip_min,
                cfg.hu.clip_max,
                cfg.hu.metal_artifact_threshold,
                tolerance_mm=cfg.resample.slice_mm / 2.0,
            )
            for pid in pids
        )
        for result in tqdm(results_iter, total=len(pids), desc="preprocess"):
            _consume(result, records, trip_rows, errors)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one,
                    pid,
                    str(cfg.paths.data_raw),
                    str(cfg.paths.data_xml),
                    str(out_dir),
                    cfg.resample.target_spacing,
                    cfg.hu.clip_min,
                    cfg.hu.clip_max,
                    cfg.hu.metal_artifact_threshold,
                    cfg.resample.slice_mm / 2.0,
                ): pid
                for pid in pids
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="preprocess"):
                _consume(fut.result(), records, trip_rows, errors)

    # Per-patient summary (without the trips field).
    report_rows = [
        {k: v for k, v in r.items() if k != "trips"} for r in records
    ]
    pd.DataFrame(report_rows).to_csv(out_dir / "preprocess_report.csv", index=False)
    pd.DataFrame(trip_rows).to_csv(out_dir / "xml_roundtrip.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "errors.csv", index=False)
        log.warning("%d patients failed; see errors.csv", len(errors))

    n_ok = len(records)
    n_perfect = sum(1 for r in records if r["roundtrip_pass_rate"] >= 1.0)
    n_partial = sum(1 for r in records if 0.95 <= r["roundtrip_pass_rate"] < 1.0)
    n_poor = sum(1 for r in records if r["roundtrip_pass_rate"] < 0.95)
    n_metal = sum(1 for r in records if r["metal_artifact_flag"])
    n_cleaned = sum(r.get("n_excluded_by_roundtrip", 0) for r in records)
    log.info(
        "Done. ok=%d | perfect=%d partial=%d poor=%d | "
        "rois_cleaned=%d | metal_flagged=%d | errors=%d",
        n_ok, n_perfect, n_partial, n_poor, n_cleaned, n_metal, len(errors),
    )
    return 0


def _consume(result: dict, records: list, trip_rows: list, errors: list) -> None:
    if result["status"] == "ok":
        records.append(result)
        trip_rows.extend(result["trips"])
    else:
        errors.append(result)


if __name__ == "__main__":
    sys.exit(main())
