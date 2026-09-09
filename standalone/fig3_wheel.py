#!/usr/bin/env python3
"""Fig. 3 — OpenTumorBoard four-arm composition wheel.

Standalone: all numbers are hard-coded, so this script has no data
dependencies. Just matplotlib. Run:

    python fig3_wheel.py --output-dir out --stem fig3

Optional: --font-family Arimo --font-dir /path/with/Arimo-*.ttf   for a
Helvetica-metric-compatible look.

Four quadrants (Cancer site / Specialist / Specialist turn type / Modality)
around one wheel, each on its own log-scaled colour family.
"""
from __future__ import annotations
import argparse, math, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Wedge


# ── data ─────────────────────────────────────────────────────────────────────
SITES = {"Thyroid / parathyroid": 126, "Genitourinary": 84, "Head & neck": 68,
         "Thoracic / lung": 59, "Breast": 54, "CNS / brain & spine": 49,
         "Hepatobiliary / pancreas": 33, "Gynecologic": 32, "Hematologic": 24,
         "Colorectal / anal": 20, "Skin / melanoma": 16, "Upper GI": 14,
         "Sarcoma (bone / soft tissue)": 12, "Skull base": 8,
         "Neuroendocrine (site NOS)": 5, "Peritoneum": 4, "Unknown primary": 3}
ROLES = {"surgeon": 5879, "medical oncologist": 5215, "radiation oncologist": 1316,
         "radiologist": 1165, "other": 1006, "molecular pathologist": 831,
         "pathologist": 685, "genetic counselor": 62,
         "clinical trial specialist": 28, "nurse navigator": 28}
QTYPES = {"findings": 4436, "treatment": 3818, "evidence": 2208,
          "next action": 2136, "uncertainty": 1161, "eligibility": 1102,
          "agreement": 574, "clarification": 571, "clinical trial": 209}
MODALITY = [("Text", 611), ("CT", 366), ("Histopathology", 338),
            ("PET/nuclear", 206), ("MRI", 178), ("Molecular/NGS", 163),
            ("Ultrasound", 121), ("Cytology", 64), ("X-ray", 12),
            ("Mammography", 10), ("Endoscopy", 1)]
SITE_RENAME = {"Unknown primary": "Unknown"}


# ── style ────────────────────────────────────────────────────────────────────
FS = 6.5
INK = "#1b1b1b"
FAMILY = {
    "site":       ["#1f4e79", "#2f6aa3", "#3f7ab8", "#528ac7", "#6ea0d3",
                   "#8ab5de", "#a3c8e6", "#b8d5ec", "#c9dff1", "#d6e6f4",
                   "#e1ecf7", "#dfe6f4", "#dae1f0", "#d3dcec", "#ccd7e8",
                   "#c4d1e4", "#bcc9e0"],
    "specialist": ["#1f5e3a", "#2c7047", "#3a8258", "#4e9469", "#65a67d",
                   "#7db892", "#95c9a7", "#addbbb", "#c4e6cb", "#dbf1de"],
    "qtype":      ["#5a2f8a", "#6e449b", "#8259ac", "#966ebd", "#a984cd",
                   "#bd9adc", "#d1b1e8", "#e0c6ef", "#ecdcf5"],
    "video":      ["#a25a1c", "#b56a2a", "#c17a3a", "#cc8b4b", "#d69c5f",
                   "#deac72", "#e5bc87", "#ecc99c", "#f1d6b1", "#f5e0c4",
                   "#f8e9d6"],
}
R_IN_RING, R_OUT_RING = 0.80, 0.92
R_BAR_MIN, R_BAR_MAX = 0.10, 0.76
R_LABEL = R_OUT_RING + 0.04
R_ARM_TEXT = (R_IN_RING + R_OUT_RING) / 2


def _short(k: str) -> str:
    label = k.replace("_", " ")
    label = label[:1].upper() + label[1:]
    label = re.sub(r"\s*\([^)]*\)\s*$", "", label)
    label = re.split(r"\s*/\s*", label)[0].strip()
    words = label.split()
    if len(words) < 2 or len(label) <= 12:
        return label
    best_i, best_diff = 1, 10**9
    for i in range(1, len(words)):
        d = abs(len(" ".join(words[:i])) - len(" ".join(words[i:])))
        if d < best_diff:
            best_i, best_diff = i, d
    return " ".join(words[:best_i]) + "\n" + " ".join(words[best_i:])


def _hex_rgb(h): h = h.lstrip("#"); return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))
def _rgb_hex(rgb): return "#{:02x}{:02x}{:02x}".format(*(int(round(c*255)) for c in rgb))


def draw_gradient_ring(ax, ang_start, ang_end, family, n_steps=200):
    deep, light = _hex_rgb(family[0]), _hex_rgb(family[-1])
    step = (ang_start - ang_end) / n_steps
    for k in range(n_steps):
        t = k / (n_steps - 1)
        rgb = tuple(deep[i]*(1-t) + light[i]*t for i in range(3))
        a1, a0 = ang_start - step*k, ang_start - step*(k+1)
        ax.add_patch(Wedge((0, 0), R_OUT_RING, a0 - 0.05, a1 + 0.05,
                           width=R_OUT_RING - R_IN_RING,
                           facecolor=_rgb_hex(rgb), edgecolor="none", zorder=3))


def draw_quadrant(ax, entries, ang_start, ang_end, family, scale="log"):
    n = len(entries)
    span = (ang_start - ang_end) / n
    values = [v for _, v in entries]
    if scale == "log":
        norm_max = math.log10(max(max(values), 1))
        norm = lambda v: math.log10(max(v, 1)) / norm_max if norm_max else 0
    else:
        norm_max = max(values) if values else 1
        norm = lambda v: v / norm_max if norm_max else 0
    a = ang_start
    for i, (label, v) in enumerate(entries):
        col = family[i % len(family)]
        a1, a0 = a, a - span
        h = norm(v)
        r_top = R_BAR_MIN + (R_BAR_MAX - R_BAR_MIN) * h
        ax.add_patch(Wedge((0, 0), r_top, a0, a1, width=r_top - R_BAR_MIN,
                           facecolor=col, edgecolor="white", lw=0.6, zorder=2))
        mid = (a0 + a1) / 2
        rad = math.radians(mid)
        x, y = R_LABEL * math.cos(rad), R_LABEL * math.sin(rad)
        rot, ha = (mid + 180, "right") if 90 < mid % 360 < 270 else (mid, "left")
        ax.text(x, y, _short(label), rotation=rot, rotation_mode="anchor",
                ha=ha, va="center", fontsize=FS, color=INK, zorder=6)
        a -= span
    return norm_max


def draw_arc_ticks(ax, ang_start, ang_end, majors, norm_max, scale,
                   minors=()):
    def frac(t):
        if scale == "log":
            if t <= 0 or math.log10(t) >= norm_max: return None
            return math.log10(t) / norm_max
        if t >= norm_max: return None
        return t / norm_max

    for t in minors:
        f = frac(t)
        if f is None: continue
        r = R_BAR_MIN + (R_BAR_MAX - R_BAR_MIN) * f
        n_seg = 60
        step = (ang_start - ang_end) / n_seg
        for k in range(0, n_seg, 2):
            a1, a0 = ang_start - step*k, ang_start - step*(k+1)
            ax.add_patch(Wedge((0, 0), r + 0.0015, a0, a1, width=0.003,
                               facecolor="#dde3ec", edgecolor="none", zorder=1.3))

    for t in majors:
        f = frac(t)
        if f is None: continue
        r = R_BAR_MIN + (R_BAR_MAX - R_BAR_MIN) * f
        n_seg = 32
        step = (ang_start - ang_end) / n_seg
        for k in range(0, n_seg, 2):
            a1, a0 = ang_start - step*k, ang_start - step*(k+1)
            ax.add_patch(Wedge((0, 0), r + 0.002, a0, a1, width=0.004,
                               facecolor="#a9b3c1", edgecolor="none", zorder=1.5))
        for a in (ang_start, ang_end):
            rad = math.radians(a)
            ax.plot([r*math.cos(rad), (r+0.020)*math.cos(rad)],
                    [r*math.sin(rad), (r+0.020)*math.sin(rad)],
                    color="#5b6572", lw=0.7, zorder=2)
        mid = (ang_start + ang_end) / 2
        rad = math.radians(mid)
        ax.text(r*math.cos(rad), r*math.sin(rad), f"{t:,}",
                fontsize=FS, color=INK, ha="center", va="center", zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


def curved_text(ax, text, ang_start, ang_end):
    a = (ang_start + ang_end) / 2
    rad = math.radians(a)
    x, y = R_ARM_TEXT * math.cos(rad), R_ARM_TEXT * math.sin(rad)
    flip = 180 < a % 360 < 360
    rot = a + 90 if flip else a - 90
    ax.text(x, y, text, rotation=rot, rotation_mode="anchor",
            ha="center", va="center",
            fontsize=FS, color="white", zorder=7)


def resolve_font(family, font_dir):
    if font_dir:
        for p in sorted(Path(font_dir).glob("*")):
            if p.suffix.lower() in (".ttf", ".otf"):
                fm.fontManager.addfont(str(p))
    plt.rcParams["font.family"] = family


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("."))
    ap.add_argument("--stem", default="fig3_wheel")
    ap.add_argument("--font-family", default="DejaVu Sans")
    ap.add_argument("--font-dir", type=Path, default=None)
    a = ap.parse_args()
    resolve_font(a.font_family, a.font_dir)

    site_entries = [(SITE_RENAME.get(k, k), v) for k, v in SITES.items()]
    role_entries = list(ROLES.items())
    q_entries = list(QTYPES.items())
    d_entries = MODALITY

    fig = plt.figure(figsize=(3.6, 3.6))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(-1.60, 1.60); ax.set_ylim(-1.60, 1.60)
    ax.set_aspect("equal"); ax.axis("off")

    GAP = 4.0
    Q = {"a": (90 - GAP/2,   0 + GAP/2),
         "b": (180 - GAP/2, 90 + GAP/2),
         "c": (270 - GAP/2, 180 + GAP/2),
         "d": (360 - GAP/2, 270 + GAP/2)}
    for k, fam in [("a", FAMILY["site"]), ("b", FAMILY["specialist"]),
                   ("c", FAMILY["qtype"]), ("d", FAMILY["video"])]:
        draw_gradient_ring(ax, *Q[k], fam)

    log_minor = [2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70, 80, 90,
                 200, 300, 400, 500, 600, 700, 800, 900,
                 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    nmax_a = draw_quadrant(ax, site_entries, *Q["a"], FAMILY["site"])
    draw_arc_ticks(ax, *Q["a"], (10, 100), nmax_a, "log", log_minor)
    curved_text(ax, "Cancer site", *Q["a"])
    nmax_b = draw_quadrant(ax, role_entries, *Q["b"], FAMILY["specialist"])
    draw_arc_ticks(ax, *Q["b"], (10, 100, 1000), nmax_b, "log", log_minor)
    curved_text(ax, "Specialist", *Q["b"])
    nmax_c = draw_quadrant(ax, q_entries, *Q["c"], FAMILY["qtype"])
    draw_arc_ticks(ax, *Q["c"], (10, 100, 1000), nmax_c, "log", log_minor)
    curved_text(ax, "Specialist turn", *Q["c"])
    nmax_d = draw_quadrant(ax, d_entries, *Q["d"], FAMILY["video"])
    draw_arc_ticks(ax, *Q["d"], (10, 100), nmax_d, "log", log_minor)
    curved_text(ax, "Modality", *Q["d"])

    for ang in (0, 90, 180, 270):
        rad = math.radians(ang)
        ax.plot([R_BAR_MIN*math.cos(rad), R_OUT_RING*math.cos(rad)],
                [R_BAR_MIN*math.sin(rad), R_OUT_RING*math.sin(rad)],
                color="#e2e6ec", lw=0.6, zorder=1)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    for suf in ("pdf", "png", "svg"):
        fig.savefig(a.output_dir / f"{a.stem}.{suf}", dpi=300,
                    bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"wrote {a.stem}.{{pdf,png,svg}} in {a.output_dir}")


if __name__ == "__main__":
    main()
