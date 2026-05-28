"""
Missing-modality performance plots for all Concat*+* architectures
using CT-FM embeddings (outputs/fold_specific_ctfm/).

Covers every combination of:
  - Architecture : ConcatFC+FC, ConcatFC+ODST, ConcatODST+FC, ConcatODST+ODST
  - Freeze MS    : unfrozen / ms-frozen
  - Modality set : CT+Tab, WSI+CT, WSI+CT+Tab, WSI+Tab

Natural-missingness x-axis corrections (same convention as plot_over_missing.py):
  - WSI : 0% artificially-missing point plotted at x=64 (natural missingness ~36%);
          25% and 50% artificially-missing points removed (too few samples).
  - CT  : 0% artificially-missing point plotted at x=5  (small natural missingness).
  - Tab : no shift (x stays at 0).

Produces:
  1. One "missing %" curve plot per experiment (arch × freeze × modality_set) — 32 files
  2. One subplot grid per modality_set (4 archs × 2 freeze states)             —  4 files
"""

import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = "outputs/fold_specific_ctfm"
TARGET     = "OS_1d_regression"
METRIC     = "c_index"
N_FOLDS    = 5
OUT_DIR    = "tests/ablation_ctfm"
os.makedirs(OUT_DIR, exist_ok=True)

FONT = "Didot, GFS Didot, Bodoni MT, Times New Roman, serif"

# Modality directory → (display label, ordered list of modality names)
MODALITY_CONFIGS = {
    "ct_tab_cox-label_cix": {
        "label"     : "CT+Tab",
        "modalities": ["CT", "Tabular"],
    },
    "wsi_ct_cox-label_cix": {
        "label"     : "WSI+CT",
        "modalities": ["WSI", "CT"],
    },
    "wsi_ct_tab_cox-label_cix": {
        "label"     : "WSI+CT+Tab",
        "modalities": ["WSI", "CT", "Tabular"],
    },
    "wsi_tab_cox-label_cix": {
        "label"     : "WSI+Tab",
        "modalities": ["WSI", "Tabular"],
    },
}

ARCH_ORDER = [
    #("ConcatFC",   "FC"),
    ("ConcatFC",   "ODST")#,
    #("ConcatODST", "FC"),
    #("ConcatODST", "ODST"),
]

# ---------------------------------------------------------------------------
# Color / marker palette — one entry per modality, high contrast
# ---------------------------------------------------------------------------
MOD_STYLE = {
    "WSI"     : dict(color="#2166ac", dash="solid",  symbol="circle",   name="WSI"),
    "CT"      : dict(color="#d6604d", dash="solid",  symbol="square",   name="CT"),
    "Tabular" : dict(color="#1a9641", dash="solid",  symbol="diamond",  name="Tabular"),
}

FREEZE_LABEL = {False: "Unfrozen", True: "Frozen MS"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def adjust_natural_missingness(points, mod_name):
    """
    Apply the same x-axis corrections used in plot_over_missing.py:
      - WSI: remove 25% and 50% artificially-missing points (too sparse);
             shift 0% → x=64 to reflect ~36% natural missingness.
      - CT:  shift 0% → x=5 to reflect small natural missingness.
      - Tab: no change.
    `points` is a list of (pct_missing, mean, se) tuples.
    Returns an adjusted, sorted list.
    """
    if mod_name == "WSI":
        points = [(x, y, s) for x, y, s in points if x not in (25, 50)]
        points = [(64 if x == 0 else x, y, s) for x, y, s in points]
    elif mod_name == "CT":
        points = [(5 if x == 0 else x, y, s) for x, y, s in points]
    return sorted(points, key=lambda t: t[0])


def get_results_subdir(exp_dir):
    results_root = os.path.join(exp_dir, "provided_provided", "results")
    if not os.path.isdir(results_root):
        return None
    names = [n for n in os.listdir(results_root)
             if os.path.isdir(os.path.join(results_root, n))]
    return names[0] if names else None


def find_exp_dir(mod_dir_path, concat_layer, layer2, frozen):
    pattern = rf"^{re.escape(concat_layer)}\+{re.escape(layer2)}_"
    candidates = []
    for name in os.listdir(mod_dir_path):
        if not re.match(pattern, name):
            continue
        if TARGET not in name:
            continue
        is_frozen = name.endswith("_ms-frozen")
        if frozen and is_frozen:
            candidates.append(name)
        elif not frozen and not is_frozen:
            candidates.append(name)
    if not candidates:
        return None
    return os.path.join(mod_dir_path, sorted(candidates)[-1])


def list_missing_subfolders(exp_dir, sub):
    """Return all numeric missing-% subfolders inside fold 0."""
    fold0 = os.path.join(exp_dir, "provided_provided", "results", sub, "0")
    if not os.path.isdir(fold0):
        return []
    return [
        d for d in os.listdir(fold0)
        if re.match(r"^\d+(-\d+)+$", d)
        and os.path.isdir(os.path.join(fold0, d))
    ]


def read_stats(exp_dir, sub, subfolder):
    """Return (mean, se) for METRIC from averages CSV, or (None, None)."""
    csv_path = os.path.join(
        exp_dir, "provided_provided", "results", sub,
        "0", subfolder, "unbalanced", "test",
        "averages_classes_average_performance.csv",
    )
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path, header=[0, 1], index_col=[0, 1])
    try:
        mean = float(df[(METRIC, "mean")].iloc[0])
        std  = float(df[(METRIC, "std")].iloc[0])
        return mean, std / np.sqrt(N_FOLDS)
    except (KeyError, IndexError):
        return None, None


def load_experiment_data(exp_dir, modalities):
    """
    Load all missing-% data for an experiment.
    Returns a dict: modality_name → sorted list of (pct_missing, mean, se).
    """
    sub = get_results_subdir(exp_dir)
    if sub is None:
        return None

    subfolders = list_missing_subfolders(exp_dir, sub)
    if len(subfolders) < 3:
        return None

    n_mods = len(modalities)
    # zero-folder: all zeros
    zero_folder = "-".join(["0"] * n_mods)

    mean0, se0 = read_stats(exp_dir, sub, zero_folder)
    if mean0 is None:
        return None

    # Group subfolders by which modality is being masked (non-zero index)
    data = {m: [(0, mean0, se0)] for m in modalities}

    for sf in subfolders:
        parts = [int(x) for x in sf.split("-")]
        if len(parts) != n_mods:
            continue
        nonzero = [(i, v) for i, v in enumerate(parts) if v > 0]
        if len(nonzero) != 1:
            continue          # skip multi-modality missing combos
        idx, pct = nonzero[0]
        mod_name = modalities[idx]
        mean, se = read_stats(exp_dir, sub, sf)
        if mean is not None:
            data[mod_name].append((pct, mean, se))

    # Sort each modality's list by % missing
    for m in modalities:
        data[m] = sorted(data[m], key=lambda t: t[0])

    return data


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def add_modality_traces(fig, data, modalities, show_legend=True,
                        row=None, col=None, legend_prefix=""):
    """Add line + SE-band traces for each modality."""
    for mod in modalities:
        if mod not in data or len(data[mod]) < 2:
            continue
        style = MOD_STYLE[mod]
        color_line = hex_to_rgba(style["color"], 0.90)
        color_fill = hex_to_rgba(style["color"], 0.15)

        adjusted = adjust_natural_missingness(data[mod], mod)
        xs   = [t[0] for t in adjusted]
        ys   = [t[1] for t in adjusted]
        ses  = [t[2] for t in adjusted]
        y_up = [y + s for y, s in zip(ys, ses)]
        y_dn = [y - s for y, s in zip(ys, ses)]

        kw = dict(row=row, col=col) if row is not None else {}

        # SE shaded band
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=y_up + y_dn[::-1],
            fill="toself",
            fillcolor=color_fill,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ), **kw)

        # Main line
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers+lines",
            name=legend_prefix + style["name"],
            line=dict(width=3, color=color_line, dash=style["dash"]),
            marker=dict(
                size=11,
                symbol=style["symbol"],
                color=color_line,
                line=dict(width=1.5, color="rgba(0,0,0,0.6)"),
            ),
            error_y=dict(
                type="data",
                array=ses,
                visible=True,
                thickness=1.5,
                width=5,
                color=hex_to_rgba(style["color"], 0.55),
            ),
            showlegend=show_legend,
        ), **kw)


def axis_style(title="", tickvals=None, ticktext=None):
    d = dict(
        title=f"<b>{title}</b>" if title else "",
        showgrid=True,
        gridcolor="#ebebeb",
        gridwidth=1,
        zeroline=False,
        linecolor="black",
        linewidth=1.5,
        mirror=True,
        title_font=dict(size=26, family=FONT, color="black"),
        tickfont=dict(size=22, family=FONT, color="black"),
    )
    if tickvals is not None:
        d["tickmode"] = "array"
        d["tickvals"] = tickvals
    if ticktext is not None:
        d["ticktext"] = ticktext
    return d


def compute_x_ticks(modalities):
    """
    Return (tickvals, ticktext) for a shared x-axis that contains curves for
    multiple modalities.  The natural-missingness-adjusted starting positions
    (WSI→64, CT→5, Tabular→0) are each labelled "0%" so that readers can
    locate the baseline for each curve on the axis.
    """
    has_wsi = "WSI"     in modalities
    has_ct  = "CT"      in modalities
    has_tab = "Tabular" in modalities

    # Build sorted (position, label) pairs
    pts = {}
    if has_tab:
        pts[0]  = "0%"    # Tabular natural-missing baseline
    if has_ct:
        pts[5]  = "5%"    # CT natural-missing baseline (~5 %)
    if has_wsi:
        pts[64] = "64%"   # WSI natural-missing baseline (~36 % → 64 % present)
    for v in [25, 50, 75, 100]:
        if v not in pts:
            pts[v] = f"{v}%"

    # When Tab(0) and CT(5) are both present they are very close; keep both
    # but rely on the plot width to separate them visually.
    sorted_vals = sorted(pts)
    return sorted_vals, [pts[v] for v in sorted_vals]


def layout_single(title=""):
    return dict(
        title={},
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(size=20, family=FONT, color="black"),
        width=900,
        height=560,
        margin=dict(l=110, r=30, t=30, b=80),
    )


# ---------------------------------------------------------------------------
# 1. Individual plots: one per (arch, freeze, modality_set)
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Individual missing-modality plots")
print("=" * 60)

# Collect data for later use in combined plots
# all_data[(arch, frozen, mod_dir_key)] = data dict or None
all_data = {}

for mod_dir_key, mod_cfg in MODALITY_CONFIGS.items():
    mod_label  = mod_cfg["label"]
    modalities = mod_cfg["modalities"]
    mod_dir_path = os.path.join(BASE_DIR, mod_dir_key)

    for concat_layer, layer2 in ARCH_ORDER:
        arch_label = f"{concat_layer}+{layer2}"

        for frozen in [False, True]:
            freeze_label = FREEZE_LABEL[frozen]
            key = (concat_layer, layer2, frozen, mod_dir_key)

            exp_dir = find_exp_dir(mod_dir_path, concat_layer, layer2, frozen)
            if exp_dir is None:
                print(f"  [SKIP] {arch_label} {freeze_label} {mod_label}: no exp dir")
                all_data[key] = None
                continue

            data = load_experiment_data(exp_dir, modalities)
            all_data[key] = data

            if data is None:
                print(f"  [SKIP] {arch_label} {freeze_label} {mod_label}: no data")
                continue

            print(f"  [PLOT] {arch_label} {freeze_label} {mod_label}")

            tick_vals, tick_text = compute_x_ticks(modalities)
            fig = go.Figure()
            add_modality_traces(fig, data, modalities)
            fig.update_xaxes(**axis_style("% Missing", tick_vals, tick_text))
            fig.update_yaxes(**axis_style("C-index (Mean ± SE)"))
            fig.update_layout(**layout_single())

            safe = re.sub(r"[^\w\-]", "_", f"{arch_label}_{freeze_label}_{mod_label}")
            fig.write_image(os.path.join(OUT_DIR, f"{safe}.png"), scale=2)

# ---------------------------------------------------------------------------
# 2. Subplot grids: one per modality_set
#    Rows = archs (4), Cols = frozen/unfrozen (2)
#    One curve per masked modality — easy visual comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Subplot grids: archs × freeze, per modality set")
print("=" * 60)

for mod_dir_key, mod_cfg in MODALITY_CONFIGS.items():
    mod_label  = mod_cfg["label"]
    modalities = mod_cfg["modalities"]

    n_archs = len(ARCH_ORDER)
    n_freeze = 2

    fig = make_subplots(
        rows=n_archs, cols=n_freeze,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.07,
        horizontal_spacing=0.04,
    )

    any_plotted = False
    for row_i, (concat_layer, layer2) in enumerate(ARCH_ORDER, start=1):
        for col_i, frozen in enumerate([False, True], start=1):
            key = (concat_layer, layer2, frozen, mod_dir_key)
            data = all_data.get(key)
            if data is None:
                continue

            show_leg = (row_i == 1 and col_i == 1)
            add_modality_traces(
                fig, data, modalities,
                show_legend=show_leg,
                row=row_i, col=col_i,
            )
            any_plotted = True

    if not any_plotted:
        continue

    # Axis labels
    g_tick_vals, g_tick_text = compute_x_ticks(modalities)
    for row_i in range(1, n_archs + 1):
        for col_i in range(1, n_freeze + 1):
            ax_idx = (row_i - 1) * n_freeze + col_i
            x_ax = f"xaxis{ax_idx if ax_idx > 1 else ''}"
            y_ax = f"yaxis{ax_idx if ax_idx > 1 else ''}"

            fig.update_layout(**{
                x_ax: dict(
                    tickmode="array",
                    tickvals=g_tick_vals,
                    ticktext=g_tick_text,
                    showgrid=True, gridcolor="#ebebeb",
                    linecolor="black", linewidth=1.5, mirror=True,
                    tickfont=dict(size=15, family=FONT, color="black"),
                    title=dict(text="<b>% Missing</b>",
                               font=dict(size=16, family=FONT, color="black"))
                    if row_i == n_archs else {},
                ),
                y_ax: dict(
                    showgrid=True, gridcolor="#ebebeb",
                    linecolor="black", linewidth=1.5, mirror=True,
                    tickfont=dict(size=15, family=FONT, color="black"),
                    title=dict(text="<b>C-index</b>",
                               font=dict(size=16, family=FONT, color="black"))
                    if col_i == 1 else {},
                ),
            })

    fig.update_layout(
        title={},
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(size=16, family=FONT, color="black"),
        width=1000,
        height=260 * n_archs + 60,
        margin=dict(l=90, r=30, t=30, b=80),
    )

    safe_mod2 = re.sub(r"[^\w]", "_", mod_label)
    out_name = f"grid_{safe_mod2}.png"
    fig.write_image(os.path.join(OUT_DIR, out_name), scale=2)
    print(f"  [SAVE] {out_name}")

print(f"\nAll plots saved to {OUT_DIR}/")
