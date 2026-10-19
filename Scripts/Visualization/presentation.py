"""Slide-style pictures of how a decision is made.

These are not result graphs. They are the frames to put on a slide when
explaining the model. Output goes to Visualization/presentation/.

    python "Scripts/Visualization/presentation.py"
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(__file__).resolve().parents[2] / "Visualization" / "presentation"

A, L, R = "#4a4a4a", "#2c6b9e", "#c05a2c"
SAFETY, EFF, COM = "#3c6e9e", "#e0902f", "#5d9b4f"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 13,
        "axes.linewidth": 0,
    })


def _car(ax, x, y, label, color, w=8.2, h=2.15):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.45",
        facecolor="white", edgecolor=color, linewidth=2.2, zorder=4,
    ))
    ax.text(x, y, label, ha="center", va="center", fontsize=15, color=color,
            fontweight="bold", zorder=5)


def _road(ax):
    ax.add_patch(Rectangle((0, 4.2), 100, 4.6, facecolor="#f3f4f6", edgecolor="none", zorder=0))
    ax.add_patch(Polygon([(0, 0.2), (78, 0.2), (90, 4.2), (0, 4.2)], facecolor="#e7e8ea", edgecolor="none", zorder=0))
    ax.plot([0, 100], [8.8, 8.8], color="#4a4a4a", lw=2)
    ax.plot([0, 100], [4.2, 4.2], color="#6a6a6a", lw=1.2, ls=(0, (8, 8)))
    ax.plot([0, 78], [0.2, 0.2], color="#4a4a4a", lw=2)
    ax.plot([78, 90], [0.2, 4.2], color="#4a4a4a", lw=2)
    ax.text(2.2, 8.15, "target lane", fontsize=11, color="#6a6a6a")
    ax.text(2.2, 0.7, "ending lane", fontsize=11, color="#6a6a6a")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10.2)
    ax.axis("off")


def _caption(fig, text):
    fig.text(0.06, 0.06, text, fontsize=15, color="#222", va="bottom", ha="left", wrap=True)


def _frame(title):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=110, facecolor="white")
    fig.text(0.06, 0.93, title, fontsize=20, fontweight="bold", color="#1a1a1a", va="top")
    return fig


def _grab(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _costs(ax, highlight=None):
    """Two horizontal stacks: Yield cheaper than Continue."""
    actions = ["Yield", "Continue"]
    parts = {
        "safety": ([0.35, 1.15], SAFETY, "safety"),
        "efficiency": ([0.70, 0.18], EFF, "efficiency"),
        "comfort": ([0.40, 0.10], COM, "comfort"),
    }
    y = [1.15, 0.15]
    left = np.zeros(2)
    for values, color, label in parts.values():
        values = np.array(values)
        ax.barh(y, values, left=left, height=0.55, color=color, label=label)
        left = left + values
    ax.set_yticks(y, actions, fontsize=14)
    ax.set_xlim(0, 2.3)
    ax.set_xlabel("cost  (lower is better)", fontsize=12)
    ax.tick_params(length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    if highlight == "yield":
        ax.add_patch(Rectangle((-0.08, 0.82), 2.15, 0.66, fill=False, edgecolor=L, lw=2.0))
        ax.text(1.55, 1.15, "cheaper", va="center", color=L, fontsize=13, fontweight="bold")


def frame_scene(step, title, caption, cars, extra=None):
    fig = _frame(f"{step}   {title}")
    ax = fig.add_axes([0.05, 0.18, 0.90, 0.68])
    _road(ax)
    for x, y, label, color in cars:
        _car(ax, x, y, label, color)
    if extra:
        extra(ax)
    _caption(fig, caption)
    return _grab(fig)


def frame_score(step, title, caption, highlight=None):
    fig = _frame(f"{step}   {title}")
    ax_road = fig.add_axes([0.04, 0.20, 0.46, 0.64])
    _road(ax_road)
    _car(ax_road, 34, 7.0, "L", L)
    _car(ax_road, 52, 2.4, "A", A)
    _car(ax_road, 30, 2.4, "R", R)
    ax_road.annotate(
        "", xy=(58, 5.0), xytext=(54, 3.3),
        arrowprops=dict(arrowstyle="-|>", color=A, lw=1.6),
    )
    ax = fig.add_axes([0.56, 0.28, 0.38, 0.50])
    _costs(ax, highlight=highlight)
    _caption(fig, caption)
    return _grab(fig)


def unfolding_gif():
    frames = []
    frames.append(frame_scene(
        "1", "No agreement beforehand",
        "L, R and the AV cannot talk first. There is no shared plan. Each driver will act on their own cost.",
        [(40, 7.0, "L", L), (50, 2.4, "A", A), (28, 2.4, "R", R)],
    ))

    def signal(ax):
        ax.plot(54.5, 3.5, marker=">", ms=14, color="#e0a100")
        ax.text(58, 3.5, "A signals", color="#8a6a00", fontsize=13, va="center")

    frames.append(frame_scene(
        "2", "A moves first",
        "The AV signals, slows, or starts to move across. L and R see that move. They still do not know each other's costs.",
        [(40, 7.0, "L", L), (50, 2.4, "A", A), (28, 2.4, "R", R)],
        extra=signal,
    ))
    frames.append(frame_score(
        "3", "L scores both options",
        "For Yield and for Continue, L adds safety, efficiency and comfort. A lower total is more attractive.",
    ))
    frames.append(frame_score(
        "4", "Cheaper is more likely, not certain",
        "Yield costs less here, so L is more likely to yield. The model still leaves some probability on Continue.",
        highlight="yield",
    ))

    def gap(ax):
        ax.annotate("", xy=(46, 7.0), xytext=(36, 7.0), arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
        ax.text(41, 7.7, "gap opens", ha="center", fontsize=12, color="#333")
        ax.annotate("", xy=(48, 5.2), xytext=(52, 3.4), arrowprops=dict(arrowstyle="-|>", color=A, lw=1.6))

    frames.append(frame_scene(
        "5", "The action changes the next stage",
        "If L yields, the gap in front of L grows. R's safety cost is no longer the one from the previous stage.",
        [(30, 7.0, "L", L), (52, 3.6, "A", A), (26, 2.4, "R", R)],
        extra=gap,
    ))
    frames.append(frame_scene(
        "6", "Another stage only if someone still responds",
        "A is in the target lane. The interaction stops when the merge is resolved. It continues only if a driver still changes action.",
        [(30, 7.0, "L", L), (48, 7.0, "A", A), (22, 2.4, "R", R)],
    ))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "how_a_decision_unfolds.gif"
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=1600, loop=0, disposal=2,
    )
    frames[3].save(OUT / "how_a_decision_unfolds.png")
    print("wrote", path)


def stages_picture():
    fig = plt.figure(figsize=(12.8, 7.2), dpi=120, facecolor="white")
    fig.text(0.06, 0.92, "Stages are not fixed in advance", fontsize=22, fontweight="bold", va="top")
    fig.text(
        0.06, 0.84,
        "Each box is one decision. The next box exists only because of the action just taken.",
        fontsize=14, color="#333", va="top",
    )

    boxes = [
        (0.06, "Stage 1", "A signals\nor slows", A, False),
        (0.28, "Stage 2", "L yields\nor continues", L, False),
        (0.50, "Stage 3", "R brakes\nor maintains", R, False),
        (0.72, "Stage 4?", "Only if the gap\nis still contested", "#888", True),
    ]
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for x, title, body, color, dashed in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, 0.38), 0.18, 0.28,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="white", edgecolor=color,
            linewidth=2.0, linestyle=(0, (4, 3)) if dashed else "solid",
        ))
        ax.text(x + 0.09, 0.60, title, ha="center", va="center", fontsize=14, color=color, fontweight="bold")
        ax.text(x + 0.09, 0.48, body, ha="center", va="center", fontsize=13, color="#222")
    for x in (0.24, 0.46, 0.68):
        ax.annotate("", xy=(x + 0.04, 0.52), xytext=(x, 0.52), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.6))
    ax.text(0.50, 0.28, "The chain stops when the merge is resolved.", ha="center", fontsize=14, color="#333")
    ax.text(
        0.50, 0.16,
        "A later trial is not another stage. It is a new round:\nthe first meeting with the AV can change the next one.",
        ha="center", fontsize=14, color="#333",
    )
    path = OUT / "several_stages.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def main() -> None:
    _style()
    unfolding_gif()
    stages_picture()


if __name__ == "__main__":
    main()
