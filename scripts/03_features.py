#!/usr/bin/env python
"""Stage 3 — feature extraction.

Reads ``outputs/01_manifest/manifest.csv`` plus stage-2 outputs
(``{pid}_ct.npy``, ``{pid}_mask.npy``, ``spacing.json``,
``xml_roundtrip.csv``, ``preprocess_report.csv``) and produces one
``features.csv`` row per patient.

Per-patient pipeline (each worker):

  1. Load native DICOM headers via ``load_patient_metadata`` for
     ``slice_positions`` and native pixel spacing.
  2. Parse XML (cheap).
  3. Derive ``excluded_roi_ids`` for this patient from the cohort
     ``xml_roundtrip.csv`` (passes=False, matched_via!=dirty).
  4. Compute Agatston (D011), per-vessel aggregates, density tiers.
  5. Group ROIs into 3D lesions (D007) → spatial features.
  6. Build per-artery masks (D008) — used here only for voxel-count
     reporting; per-vessel PyRadiomics is skipped by default to keep
     stage-3 cost predictable (~107 features per mask × 5 masks per
     patient is too much for a first run). Toggle on with
     ``--per-vessel-radiomics``.
  7. Run PyRadiomics on the resampled whole-mask (D009).
  8. Aggregate into one row + a list of lesion rows for the audit CSV.

Outputs (under ``outputs/03_features/``):

  - ``features.csv``         — main artefact, one row per patient.
  - ``lesions.csv``          — one row per 3D lesion (audit trail).
  - ``feature_extraction_log.csv`` — per-patient runtime + counts.
  - ``errors.csv``           — patients that failed (if any).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from predict.config import Config, load_config
from predict.features.agatston import compute_agatston
from predict.features.density_tiers import compute_density_tiers
from predict.features.feature_schema import zero_features
from predict.features.lesion_ccl import group_rois_into_lesions
from predict.features.per_artery_mask import build_per_artery_masks, per_artery_voxel_counts
from predict.features.per_vessel_aggregates import compute_per_vessel_aggregates
from predict.features.radiomics import create_extractor, extract_pyradiomics
from predict.features.spatial import compute_spatial_features
from predict.io import (
    PatientMetadata,
    load_patient_metadata,
    load_spacing_metadata,
    parse_calcium_xml,
)


LOW_BURDEN_VOXEL_THRESHOLD: int = 100


# ───────────────────── reproducibility breadcrumbs ─────────────────────


def _git_hash(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _file_sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def build_run_header(repo_root: Path, params_yaml: Path) -> dict:
    """Return a dict of reproducibility breadcrumbs for the run."""
    import sys as _sys
    info = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_hash(repo_root),
        "params_yaml_sha": _file_sha(params_yaml),
        "python_version": _sys.version.split()[0],
    }
    # Library versions (lazy imports so this still runs without them).
    for mod in ("numpy", "pandas", "SimpleITK", "pydicom", "radiomics", "cv2"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            info[f"{mod}_version"] = "n/a"
    return info


# ───────────────────── per-patient worker ─────────────────────


def _process_one(
    pid: str,
    data_root_str: str,
    preproc_dir_str: str,
    excluded_set: set,
    target_spacing: tuple,
    quality_label: str,
    include_per_vessel_radiomics: bool,
    extractor_holder: dict,
) -> dict:
    """Compute every feature for one patient."""
    t0 = time.perf_counter()
    try:
        data_root = Path(data_root_str)
        preproc_dir = Path(preproc_dir_str)

        meta: PatientMetadata = load_patient_metadata(pid, data_root)
        parse_result = parse_calcium_xml(pid, data_root / "calcium_xml")

        # Stage-2 artefacts.
        ct = np.load(preproc_dir / f"{pid}_ct.npy")
        mask = np.load(preproc_dir / f"{pid}_mask.npy")

        # ── Feature dict starts from schema zeros ─────────────────────
        row = zero_features()
        row["pid"] = pid
        row["kernel"] = meta.kernel
        row["scanner_model"] = meta.scanner_model
        row["mask_voxels"] = int(mask.sum())
        row["low_burden_flag"] = int(mask.sum()) < LOW_BURDEN_VOXEL_THRESHOLD
        row["roundtrip_quality"] = quality_label

        # ── Agatston (XML, D011) ──────────────────────────────────────
        ag = compute_agatston(
            parse_result,
            slice_thickness_mm=meta.slice_thickness,
            excluded_roi_ids=excluded_set,
        )
        row.update(ag.to_feature_dict())
        row["category"] = ag.category

        # ── Per-vessel aggregates ─────────────────────────────────────
        row.update(compute_per_vessel_aggregates(
            parse_result,
            slice_thickness_mm=meta.slice_thickness,
            excluded_roi_ids=excluded_set,
        ))

        # ── Density tiers ─────────────────────────────────────────────
        row.update(compute_density_tiers(parse_result, excluded_roi_ids=excluded_set))

        # ── Lesion CCL + spatial features ─────────────────────────────
        lesions_per_vessel = group_rois_into_lesions(
            parse_result,
            slice_positions=meta.slice_positions,
            pixel_spacing_xy=(meta.pixel_spacing[0], meta.pixel_spacing[1]),
            slice_thickness_mm=meta.slice_thickness,
            excluded_roi_ids=excluded_set,
        )
        row.update(compute_spatial_features(
            lesions_per_vessel,
            slice_positions=meta.slice_positions,
            slice_thickness_mm=meta.slice_thickness,
        ))

        # ── PyRadiomics on whole-mask ─────────────────────────────────
        # Graceful degrade per D010: if PyRadiomics refuses (mask below
        # minimumROISize, typically very-low-burden patients already flagged
        # by low_burden_flag), keep the row with XML-derived features and
        # leave PyRadiomics columns missing (NaN in the CSV).
        extractor = extractor_holder.get("e")
        if extractor is None:
            extractor = create_extractor()
            extractor_holder["e"] = extractor

        radiomics_status = "ok"
        radiomics_reason = "ok"
        try:
            radiomics = extract_pyradiomics(ct, mask, target_spacing, extractor, pid=pid)
            row.update(radiomics)
        except ValueError as exc:
            # Almost always "Size of the ROI is too small" for tiny masks.
            radiomics_status = "skipped"
            radiomics_reason = str(exc)
            radiomics = {}

        row["radiomics_status"] = radiomics_status
        row["radiomics_reason"] = radiomics_reason

        # ── Optional per-vessel PyRadiomics ───────────────────────────
        per_vessel_voxels: dict[str, int] = {}
        if include_per_vessel_radiomics:
            masks_pv = build_per_artery_masks(
                parse_result, _LoadedShim(meta, ct.shape),
                excluded_roi_ids=excluded_set,
            )
            per_vessel_voxels = per_artery_voxel_counts(masks_pv)
            for vessel, vmask in masks_pv.items():
                if vmask.sum() == 0:
                    continue
                pv_feats = extract_pyradiomics(
                    ct, vmask, target_spacing, extractor, pid=f"{pid}/{vessel}",
                )
                for k, v in pv_feats.items():
                    row[f"{vessel}_{k}"] = v

        # ── Lesion audit rows ─────────────────────────────────────────
        lesion_rows = [
            {
                "pid": pid,
                "vessel": vessel,
                "lesion_idx": i,
                "n_rois": l.n_rois,
                "slice_indices": ";".join(str(s) for s in l.slice_indices),
                "centroid_x_mm": l.centroid_mm[0],
                "centroid_y_mm": l.centroid_mm[1],
                "centroid_z_mm": l.centroid_mm[2],
                "total_area_mm2": l.total_area_mm2,
                "volume_mm3": l.volume_mm3,
                "mean_hu_weighted": l.mean_hu_weighted,
                "max_hu": l.max_hu,
            }
            for vessel, lesions in lesions_per_vessel.items()
            for i, l in enumerate(lesions)
        ]

        runtime = time.perf_counter() - t0
        log_row = {
            "pid": pid,
            "runtime_sec": round(runtime, 2),
            "n_radiomics_features": len(radiomics),
            "radiomics_status": radiomics_status,
            "radiomics_reason": radiomics_reason,
            "n_lesions": sum(len(v) for v in lesions_per_vessel.values()),
            "mask_voxels": int(mask.sum()),
            "low_burden_flag": row["low_burden_flag"],
            "per_vessel_voxels": ";".join(
                f"{k}={v}" for k, v in per_vessel_voxels.items()
            ) if per_vessel_voxels else "",
        }

        return {"status": "ok", "row": row, "lesions": lesion_rows, "log": log_row}

    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "pid": pid,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_sec": round(time.perf_counter() - t0, 2),
        }


class _LoadedShim:
    """Minimal LoadedPatient-like adapter for ``build_per_artery_masks``.

    The mask_builder reads ``loaded.ct_sitk`` only to discover the volume
    shape. We can fabricate a zero SimpleITK image of the right shape and
    spacing from ``meta`` + the resampled CT shape without paying for a
    full image load.
    """

    def __init__(self, meta: PatientMetadata, ct_shape: tuple[int, int, int]):
        import SimpleITK as sitk
        z, y, x = ct_shape
        arr = np.zeros((z, y, x), dtype=np.int16)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(meta.pixel_spacing)
        img.SetOrigin((0.0, 0.0, 0.0))
        self.ct_sitk = img
        self.pid = meta.pid
        self.slice_positions = meta.slice_positions
        self.pixel_spacing = meta.pixel_spacing
        self.slice_thickness = meta.slice_thickness
        self.n_slices = meta.n_slices
        self.scanner_model = meta.scanner_model
        self.kernel = meta.kernel
        self.manufacturer = meta.manufacturer
        self.series_uid = meta.series_uid


# Worker-process scope: cache the extractor across patients to avoid
# reinitialising it for every call.
_worker_extractor_holder: dict = {}


def _worker(args) -> dict:
    return _process_one(*args, _worker_extractor_holder)


# ───────────────────── main ─────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n-workers", type=int, default=min(16, mp.cpu_count()))
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N patients.")
    parser.add_argument("--per-vessel-radiomics", action="store_true",
                        help="Also run PyRadiomics on each per-vessel mask (slow).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("features")

    cfg: Config = load_config(args.config)
    out_dir = cfg.paths.outputs / "03_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    preproc_dir = cfg.paths.outputs / "02_preprocessed"

    # Write a run header for reproducibility (D-feature provenance).
    header = build_run_header(
        repo_root=cfg.paths.outputs.parent,
        params_yaml=cfg.paths.outputs.parent / "params.yaml",
    )
    (out_dir / "run_header.json").write_text(json.dumps(header, indent=2) + "\n",
                                              encoding="utf-8")
    log.info("Run header: %s", header)

    manifest_path = cfg.paths.outputs / "01_manifest" / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"pid": str})
    pids = manifest["pid"].astype(str).tolist()
    if args.limit:
        pids = pids[: args.limit]

    # Load preprocess report for quality labels.
    preproc_report = pd.read_csv(
        preproc_dir / "preprocess_report.csv", dtype={"pid": str},
    )
    quality_by_pid = dict(zip(preproc_report["pid"].astype(str),
                              preproc_report["roundtrip_quality"]))

    # Load roundtrip CSV and group failures per patient.
    trips = pd.read_csv(preproc_dir / "xml_roundtrip.csv", dtype={"pid": str})
    fail_mask = (~trips["passes"].astype(bool)) & (trips["matched_via"] != "dirty")
    fails = trips[fail_mask]
    excluded_by_pid: dict[str, set] = {}
    for _, r in fails.iterrows():
        excluded_by_pid.setdefault(str(r["pid"]), set()).add(
            (int(r["image_index"]), int(r["roi_idx_in_slice"]))
        )

    target_spacing = load_spacing_metadata(preproc_dir / "spacing.json")

    log.info(
        "Features: %d patients | workers=%d | per-vessel-radiomics=%s",
        len(pids), args.n_workers, args.per_vessel_radiomics,
    )

    rows: list[dict] = []
    lesion_rows: list[dict] = []
    log_rows: list[dict] = []
    errors: list[dict] = []

    tasks = [
        (
            pid,
            str(cfg.paths.data_raw.parent),
            str(preproc_dir),
            excluded_by_pid.get(pid, set()),
            target_spacing,
            quality_by_pid.get(pid, "ok"),
            args.per_vessel_radiomics,
        )
        for pid in pids
    ]

    if args.n_workers == 1:
        for t in tqdm(tasks, desc="features"):
            _consume(_worker(t), rows, lesion_rows, log_rows, errors)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {pool.submit(_worker, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="features"):
                _consume(fut.result(), rows, lesion_rows, log_rows, errors)

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "features.csv", index=False)
    if lesion_rows:
        pd.DataFrame(lesion_rows).to_csv(out_dir / "lesions.csv", index=False)
    if log_rows:
        pd.DataFrame(log_rows).to_csv(out_dir / "feature_extraction_log.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "errors.csv", index=False)

    log.info(
        "Done. ok=%d | low_burden=%d | errors=%d | features.csv columns=%d",
        len(rows),
        sum(1 for r in rows if r.get("low_burden_flag")),
        len(errors),
        len(rows[0]) if rows else 0,
    )
    return 0


def _consume(result, rows, lesion_rows, log_rows, errors) -> None:
    if result["status"] == "ok":
        rows.append(result["row"])
        lesion_rows.extend(result["lesions"])
        log_rows.append(result["log"])
    else:
        errors.append(result)


if __name__ == "__main__":
    sys.exit(main())
