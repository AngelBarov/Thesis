"""Shared style for the result-figure mock-ups.

Figures are always written to the repository ``Visualization`` folder,
regardless of the working directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "Visualization"

FIGW = 6.3

C = {
    "yield": "#2c6b9e",
    "continue": "#c05a2c",
    "brake": "#2c6b9e",
    "maintain": "#c05a2c",
    "safety": "#3c6e9e",
    "efficiency": "#e0902f",
    "comfort": "#5d9b4f",
    "interaction": "#8a76b8",
    "L": "#2c6b9e",
    "R": "#c05a2c",
    "A": "#4a4a4a",
    "grid": "0.88",
    "p1": "#4e79a7",
    "p2": "#8fb03e",
    "p3": "#c4562f",
}

PHASE_COLORS = {
    "Approach": "#d9e3ec",
    "Anticipation": "#f5dfa8",
    "Yield": "#a8c6de",
    "Continue": "#eab08a",
    "Conflict": "#d98b8b",
    "Resolution": "#cfd9c8",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "0.35",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "grid.color": C["grid"],
            "grid.linewidth": 0.6,
        }
    )


def save(fig, name: str, note: str = "Illustrative synthetic data - figure layout mock-up") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.text(
        tight.x0 / fig.get_figwidth(),
        tight.y0 / fig.get_figheight() - 0.012,
        note,
        fontsize=5.6,
        color="0.5",
        ha="left",
        va="top",
    )
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def smoothstep(n: int):
    t = np.linspace(0.0, 1.0, n)
    return t * t * (3 - 2 * t)
