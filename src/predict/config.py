"""Shared configuration loader.

Reads ``configs/default.yaml`` and exposes typed constants used across stages.
This is the single source of truth for paths, seeds, thresholds, and cohort
exclusions.

Per-stage modules import from this module. Do not introduce a stage-local
``config.py`` — extend this file or the YAML instead.

Decisions referencing this module:
    D004 — Cohort exclusions at discovery
    D005 — Target voxel grid
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repo root: this file lives at src/predict/config.py
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    data_raw: Path
    data_xml: Path
    outputs: Path


@dataclass(frozen=True)
class Cohort:
    exclude_pids: tuple[str, ...]
    exclude_ge_scanners: bool


@dataclass(frozen=True)
class Resample:
    in_plane_mm: float
    slice_mm: float

    @property
    def target_spacing(self) -> tuple[float, float, float]:
        # SimpleITK convention: (x, y, z) in mm.
        return (self.in_plane_mm, self.in_plane_mm, self.slice_mm)


@dataclass(frozen=True)
class HU:
    agatston_threshold: int
    metal_artifact_threshold: int
    clip_min: int
    clip_max: int


@dataclass(frozen=True)
class LesionGrouping:
    max_inplane_dist_mm: float
    max_slice_gap: int


@dataclass(frozen=True)
class Stability:
    rotation_degrees: tuple[float, ...]
    translation_mm: tuple[float, ...]
    noise_sigma_hu: tuple[float, ...]
    background_fill_hu: float
    noise_seed_multiplier: int
    icc_threshold: float


@dataclass(frozen=True)
class Reduce:
    variance_threshold: float
    icc_threshold: float
    r2_linkage: str
    r2_elbow_min_gap: float
    r2_fallback_distance: float
    pca_cumvar: float


@dataclass(frozen=True)
class Config:
    paths: Paths
    seed: int
    cohort: Cohort
    resample: Resample
    hu: HU
    lesion_grouping: LesionGrouping
    stability: Stability
    reduce: Reduce
    raw: dict[str, Any]  # full parsed YAML, for stage-specific lookup


def _to_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(path: Path | None = None) -> Config:
    """Load config from YAML. Defaults to ``configs/default.yaml``."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        paths=Paths(
            data_raw=_to_path(raw["paths"]["data_raw"]),
            data_xml=_to_path(raw["paths"]["data_xml"]),
            outputs=_to_path(raw["paths"]["outputs"]),
        ),
        seed=int(raw["seed"]),
        cohort=Cohort(
            exclude_pids=tuple(str(p) for p in raw["cohort"]["exclude_pids"]),
            exclude_ge_scanners=bool(raw["cohort"]["exclude_ge_scanners"]),
        ),
        resample=Resample(
            in_plane_mm=float(raw["resample"]["in_plane_mm"]),
            slice_mm=float(raw["resample"]["slice_mm"]),
        ),
        hu=HU(
            agatston_threshold=int(raw["hu"]["agatston_threshold"]),
            metal_artifact_threshold=int(raw["hu"]["metal_artifact_threshold"]),
            clip_min=int(raw["hu"]["clip_min"]),
            clip_max=int(raw["hu"]["clip_max"]),
        ),
        lesion_grouping=LesionGrouping(
            max_inplane_dist_mm=float(raw["lesion_grouping"]["max_inplane_dist_mm"]),
            max_slice_gap=int(raw["lesion_grouping"]["max_slice_gap"]),
        ),
        stability=Stability(
            rotation_degrees=tuple(float(x) for x in raw["stability"]["rotation_degrees"]),
            translation_mm=tuple(float(x) for x in raw["stability"]["translation_mm"]),
            noise_sigma_hu=tuple(float(x) for x in raw["stability"]["noise_sigma_hu"]),
            background_fill_hu=float(raw["stability"]["background_fill_hu"]),
            noise_seed_multiplier=int(raw["stability"]["noise_seed_multiplier"]),
            icc_threshold=float(raw["stability"]["icc_threshold"]),
        ),
        reduce=Reduce(
            variance_threshold=float(raw["reduce"]["variance_threshold"]),
            icc_threshold=float(raw["reduce"]["icc_threshold"]),
            r2_linkage=str(raw["reduce"]["r2_linkage"]),
            r2_elbow_min_gap=float(raw["reduce"]["r2_elbow_min_gap"]),
            r2_fallback_distance=float(raw["reduce"]["r2_fallback_distance"]),
            pca_cumvar=float(raw["reduce"]["pca_cumvar"]),
        ),
        raw=raw,
    )


@lru_cache(maxsize=1)
def default_config() -> Config:
    """Cached accessor for the default config. Tests can ignore this."""
    return load_config()


# Canonical vessel names. The four full-name strings COCA emits map to these.
# Dirty / unknown names are returned as ``None`` from the parser.
VESSEL_NAME_MAP: dict[str, str] = {
    "Left Anterior Descending Artery": "LAD",
    "Right Coronary Artery": "RCA",
    "Left Circumflex Artery": "LCx",
    "Left Coronary Artery": "LM",
}
VESSEL_NAMES: tuple[str, ...] = ("LAD", "RCA", "LCx", "LM")
