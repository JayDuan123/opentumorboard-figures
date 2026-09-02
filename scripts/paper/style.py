#!/usr/bin/env python3
"""House plotting style: A4 canvas, Helvetica, 8/10 pt.

WHY THERE IS NO HELVETICA FILE HERE. The usual recipe converts macOS's
`/System/Library/Fonts/Helvetica.dfont` with `fondu`. That file does not exist on
Linux, Helvetica is licensed and not redistributable, so the substitute is chosen per
output format instead:

  PDF  `pdf.use14corefonts` writes `/BaseFont /Helvetica` and lets the viewer supply
       it. Verified: the PDF really does reference Helvetica, and the middle dot and
       en dash this project uses as separators both survive and extract correctly.
  PNG  falls back to Nimbus Sans, URW's metrically exact Helvetica clone, already
       present at /usr/share/fonts/urw-base35. Same metrics, so a PNG and its PDF
       lay out identically even though the glyphs are drawn from different files.

ONE THING TO KNOW ABOUT THE TWO PDF SETTINGS. `pdf.fonttype = 42` and
`pdf.use14corefonts = True` do not combine: with core fonts on, nothing is embedded
and only the base 14 are available, so fonttype is moot and any family that is not
one of the 14 silently becomes Helvetica. Both are set because 42 is what applies if
core fonts are ever switched off; the behaviour above is what actually ships.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# A4, the page these figures are laid into. Width is fixed; height is taken as a
# fraction, so `figsize=(A4_W, 0.42 * A4_H)`.
A4_W, A4_H = 210 / 25.4, 297 / 25.4          # 8.27 x 11.69 in

LARGE_SIZE = 10
MEDIUM_SIZE = 8
SMALLER_SIZE = 8

# Helvetica first; Nimbus Sans is the metric clone that actually renders on Linux.
SANS = ["Helvetica", "Nimbus Sans", "Arial", "Liberation Sans", "DejaVu Sans"]


def resolved_sans() -> str:
    """The family that will actually draw. Raster output uses this, not Helvetica."""
    for name in SANS:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue
    raise SystemExit(
        "style: none of " + ", ".join(SANS) + " is installed. Install a Helvetica-metric "
        "face (urw-base35 provides Nimbus Sans) rather than letting matplotlib fall back "
        "silently - a paper figure must not change typeface without saying so.")


def apply(dark: bool = False, core_fonts: bool = True) -> str:
    """Set the house style. Returns the family that will actually render."""
    if dark:
        plt.style.use("dark_background")
        mpl.rcParams.update({"ytick.color": "w", "xtick.color": "w",
                             "axes.labelcolor": "w", "axes.edgecolor": "w"})
    family = resolved_sans()
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "pdf.use14corefonts": core_fonts,
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "font.size": MEDIUM_SIZE,
        "axes.labelsize": MEDIUM_SIZE,
        "axes.titlesize": MEDIUM_SIZE,
        "xtick.labelsize": SMALLER_SIZE,
        "ytick.labelsize": SMALLER_SIZE,
        "figure.titlesize": MEDIUM_SIZE,
        "legend.fontsize": MEDIUM_SIZE,
    })
    return family
