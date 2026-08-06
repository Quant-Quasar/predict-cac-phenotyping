"""Shared matplotlib styling for publication-grade figures.

Single source of truth for fonts, colours, sizes, and the save helper.
Importing this module sets the global matplotlib rcParams; call
``apply()`` again if any other code overrides them.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Cohort + phenotype + class palettes (consistent across all figures).
PALETTE = {
    # Burden tertiles
    "low":   "#4878D0",
    "mid":   "#EE854A",
    "high":  "#D65F5F",
    # Spatial k=2 phenotypes
    "focal":   "#82C341",
    "diffuse": "#956CB4",
    # Lesion-experiment broad classes
    "soft_microspots": "#82C9F4",
    "moderate_nodules": "#FFC107",
    "dense_plaques":   "#D43F3A",
    # Vessels
    "LAD": "#8C564B", "RCA": "#E377C2", "LCx": "#7F7F7F", "LM": "#BCBD22",
    # Generic categorical (for cohorts, algorithms, etc.)
    "neutral_blue":   "#3274A1",
    "neutral_orange": "#E1812C",
    "neutral_green":  "#3A923A",
    # Cell colours for confirmed/refuted matrices
    "confirmed": "#3A923A",
    "refuted":   "#C03028",
    "ns":        "#BBBBBB",
}


def apply() -> None:
    """Set the global rcParams. Idempotent."""
    mpl.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linewidth":    0.4,
        "legend.frameon":    False,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "figure.dpi":        100,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "pdf.fonttype":      42,   # embed TrueType (editable in Illustrator)
        "ps.fonttype":       42,
    })


def save(fig: "mpl.figure.Figure", out_dir: Path, stem: str) -> dict[str, Path]:
    """Save a figure to PDF + PNG at 300 dpi. Returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": out_dir / f"{stem}.pdf",
        "png": out_dir / f"{stem}.png",
    }
    fig.savefig(paths["pdf"])
    fig.savefig(paths["png"])
    plt.close(fig)
    return paths


# Apply once on import so every figure script gets the styling for free.
apply()
