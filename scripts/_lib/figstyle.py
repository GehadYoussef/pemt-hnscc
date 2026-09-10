"""Publication figure style shared by every plotting script.

Usage:
    from _lib.figstyle import apply_style, save_figure, PALETTE, CLASS_COLOURS

Conventions
- Vector SVG (text kept as text so fonts render on the reader's machine) plus a
  600 dpi PNG of the same figure. Both go to manuscript/figures/.
- Arial where available, with metric-compatible fallbacks.
- No titles inside panels; the caption carries the description.
- Axis labels in plain English, no variable names or underscores.
- Colour-blind-safe Okabe-Ito palette.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

PALETTE = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "vermilion": "#D55E00", "purple": "#CC79A7", "black": "#000000",
    "grey": "#7F7F7F", "lightgrey": "#C8C8C8",
}
CLASS_COLOURS = {
    "pEMT_high": PALETTE["vermilion"],
    "epithelial_like": PALETTE["blue"],
    "fibroblast_stromal_like": PALETTE["green"],
}
CLASS_LABELS = {
    "pEMT_high": "pEMT-high",
    "epithelial_like": "Epithelial-like",
    "fibroblast_stromal_like": "Fibroblast / stromal-like",
}
TERTILE_COLOURS = {"Low": PALETTE["blue"], "Mid": PALETTE["grey"], "High": PALETTE["vermilion"]}


def apply_style() -> None:
    matplotlib.use("Agg")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 600, "figure.dpi": 100,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    })


def save_figure(fig, out_dir: Path, stem: str) -> None:
    """Write <stem>.svg and <stem>.png (600 dpi) into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def mm(x: float) -> float:
    """Millimetres to inches for figsize."""
    return x / 25.4
