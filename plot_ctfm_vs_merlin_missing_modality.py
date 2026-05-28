"""
Missing-modality performance plots: CT-FM vs Merlin CT embeddings.

For every (arch × freeze × modality_set) experiment, one figure is produced
that overlays two sets of curves on the same axes:
  — CT-FM  (solid lines)  : outputs/fold_specific_ctfm/
  — Merlin (dashed lines) : outputs/fold_specific/

Color encodes the masked modality (WSI=blue, CT=red, Tabular=green).
Line style encodes the CT encoder (solid=CT-FM, dashed=Merlin).
SE is shown as a shaded band (CT-FM) or error bars (Merlin) to reduce clutter.

Natural-missingness x-axis corrections (identical to plot_over_missing.py):
  - WSI : 0% → x=64, remove 25% and 50% points.
  - CT  : 0% → x=5.
  - Tab : no shift.

Produces:
  1. One plot per experiment (arch × freeze × modality_set)  — up to 32 files
  2. One subplot grid per modality_set (4 archs × 2 freeze)  —  4 files
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
SOURCES = {
    "CT-FM" : "outputs/fold_specific_ctfm",
    "Merlin": "outputs/fold_specific",
}

TARGET  = "OS_1d_regression"
METRIC  = "c_index"
N_FOLDS = 5
OUT_DIR = "tests/ablation_ctfm_vs_merlin"
os.makedirs(OUT_DIR, exist_ok=True)

FONT = "Didot, GFS Didot, Bodoni MT, Times New Roman, serif"

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
    ("ConcatFC",   "FC"),
    ("ConcatFC",   "ODST"),
    ("ConcatODST", "FC"),
    ("ConcatODST", "ODST"),
]

# Only plot this specific (modality_set → arch) combination
SELECTED_ARCH = {
    "wsi_ct_cox-label_cix"    : ("ConcatFC",   "FC"),
    "wsi_tab_cox-label_cix"   : ("ConcatFC",   "ODST"),
    "ct_tab_cox-label_cix"    : ("ConcatFC",   "ODST"),
    "wsi_ct_tab_cox-label_cix": ("ConcatODST", "ODST"),
}

FREEZE_LABEL = {False: "Unfrozen", True: "Frozen MS"}

# No CT encoder involved → no Merlin comparison, no source label
NO_MERLIN_MODS = {"wsi_tab_cox-label_cix"}

# ---------------------------------------------------------------------------
# Visual encoding
# ---------------------------------------------------------------------------
MOD_STYLE = {
    "WSI"    : dict(color="#2166ac", symbol="circle",  name="WSI"),
    "CT"     : dict(color="#d6604d", symbol="square",  name="CT"),
    "Tabular": dict(color="#1a9641", symbol="diamond", name="Tabular"),
}

SOURCE_STYLE = {
    "CT-FM" : dict(dash="solid",  opacity_line=0.92, opacity_fill=0.18,
                   band=True,  errbar=False, line_width=3.5, marker_size=12),
    "Merlin": dict(dash="dash",   opacity_line=0.80, opacity_fill=0.00,
                   band=False, errbar=True,  line_width=2.5, marker_size=10),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def adjust_natural_missingness(points, mod_name):
    """
    Same x-axis corrections as plot_over_missing.py.
    WSI: remove 25/50 points, shift 0 → 64.
    CT : shift 0 → 5.
    Tab: no change.
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
        # skip "copy" directories
        if "copy" in name.lower():
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
    fold0 = os.path.join(exp_dir, "provided_provided", "results", sub, "0")
    if not os.path.isdir(fold0):
        return []
    return [
        d for d in os.listdir(fold0)
        if re.match(r"^\d+(-\d+)+$", d)
        and os.path.isdir(os.path.join(fold0, d))
    ]


def read_stats(exp_dir, sub, subfolder):
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
    """Returns dict: modality_name → list of raw (pct_missing, mean, se)."""
    sub = get_results_subdir(exp_dir)
    if sub is None:
        return None
    subfolders = list_missing_subfolders(exp_dir, sub)
    if len(subfolders) < 3:
        return None

    n_mods = len(modalities)
    zero_folder = "-".join(["0"] * n_mods)
    mean0, se0 = read_stats(exp_dir, sub, zero_folder)
    if mean0 is None:
        return None

    data = {m: [(0, mean0, se0)] for m in modalities}
    for sf in subfolders:
        parts = [int(x) for x in sf.split("-")]
        if len(parts) != n_mods:
            continue
        nonzero = [(i, v) for i, v in enumerate(parts) if v > 0]
        if len(nonzero) != 1:
            continue
        idx, pct = nonzero[0]
        mod_name = modalities[idx]
        mean, se = read_stats(exp_dir, sub, sf)
        if mean is not None:
            data[mod_name].append((pct, mean, se))

    for m in modalities:
        data[m] = sorted(data[m], key=lambda t: t[0])
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def compute_x_ticks(modalities):
    """
    Return (tickvals, ticktext) for a shared x-axis covering multiple modalities.
    Natural-missingness starting positions (WSI→64, CT→5, Tabular→0) are each
    labelled "0%" so that readers can locate each modality's baseline on the axis.
    """
    has_wsi = "WSI"     in modalities
    has_ct  = "CT"      in modalities
    has_tab = "Tabular" in modalities

    pts = {}
    if has_tab:
        pts[0]  = "0%"
    if has_ct:
        pts[5]  = "5%"
    if has_wsi:
        pts[64] = "64%"
    for v in [25, 50, 75, 100]:
        if v not in pts:
            pts[v] = f"{v}%"

    sorted_vals = sorted(pts)
    return sorted_vals, [pts[v] for v in sorted_vals]


def add_source_traces(fig, data, modalities, source_name,
                      show_legend=True, row=None, col=None,
                      legend_ref="legend"):
    """Add traces for one source (CT-FM or Merlin) across all modalities.

    legend_ref: "legend" for CT-FM row, "legend2" for Merlin row.
    Modalities are always added in WSI → CT → Tabular order so both rows
    align column-by-column in the horizontal legend layout.
    """
    ss  = SOURCE_STYLE[source_name]
    kw  = dict(row=row, col=col) if row is not None else {}

    # Enforce fixed display order regardless of experiment modality order
    MOD_ORDER = ["WSI", "CT", "Tabular"]
    ordered = [m for m in MOD_ORDER if m in modalities]

    for mod in ordered:
        if mod not in data or len(data[mod]) < 2:
            continue

        ms      = MOD_STYLE[mod]
        c_line  = hex_to_rgba(ms["color"], ss["opacity_line"])
        c_fill  = hex_to_rgba(ms["color"], ss["opacity_fill"])

        pts = adjust_natural_missingness(data[mod], mod)
        xs  = [t[0] for t in pts]
        ys  = [t[1] for t in pts]
        ses = [t[2] for t in pts]

        legend_name = ms["name"]   # just the modality name — row header says the source

        # SE shaded band (CT-FM only)
        if ss["band"]:
            y_up = [y + s for y, s in zip(ys, ses)]
            y_dn = [y - s for y, s in zip(ys, ses)]
            fig.add_trace(go.Scatter(
                x=xs + xs[::-1],
                y=y_up + y_dn[::-1],
                fill="toself",
                fillcolor=c_fill,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
                legend=legend_ref,
            ), **kw)

        # Main line
        err_y = dict(
            type="data", array=ses, visible=True,
            thickness=1.5, width=5,
            color=hex_to_rgba(ms["color"], 0.50),
        ) if ss["errbar"] else None

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers+lines",
            name=legend_name,
            line=dict(width=ss["line_width"], color=c_line, dash=ss["dash"]),
            marker=dict(
                size=ss["marker_size"],
                symbol=ms["symbol"],
                color=c_line,
                line=dict(width=1.5, color="rgba(0,0,0,0.55)"),
            ),
            error_y=err_y,
            showlegend=show_legend,
            legend=legend_ref,
        ), **kw)


def base_axis(title="", tickvals=None, ticktext=None, font_size=22, title_size=24):
    d = dict(
        title=f"<b>{title}</b>" if title else "",
        showgrid=True, gridcolor="#ebebeb", gridwidth=1,
        zeroline=False, linecolor="black", linewidth=1.5, mirror=True,
        title_font=dict(size=title_size, family=FONT, color="black"),
        tickfont=dict(size=font_size, family=FONT, color="black"),
    )
    if tickvals is not None:
        d["tickmode"] = "array"
        d["tickvals"] = tickvals
    if ticktext is not None:
        d["ticktext"] = ticktext
    return d


def _legend_row(y, title_text, font_size=17):
    """Return a horizontal legend dict anchored at the given y position."""
    return dict(
        title=dict(
            text=title_text,
            font=dict(size=font_size, family=FONT, color="black"),
            side="left",
        ),
        orientation="h",
        yanchor="top", y=y,
        xanchor="left", x=0.0,
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(0,0,0,0.22)",
        borderwidth=1.5,
        font=dict(size=font_size, family=FONT, color="black"),
        itemsizing="constant",
        itemwidth=95,
    )


def layout_single(title="", n_modalities=2, has_merlin=True):
    base = dict(
        title=dict(text=title, font=dict(size=20, family=FONT, color="black"),
                   x=0.5, xanchor="center") if title else {},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=18, family=FONT, color="black"),
        width=980,
        height=630,
        margin=dict(l=115, r=30, t=55, b=220),
    )
    if not has_merlin:
        # Single legend row, no source label
        base["legend"] = dict(
            orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0.0,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(0,0,0,0.22)",
            borderwidth=1.5,
            font=dict(size=17, family=FONT, color="black"),
            itemsizing="constant",
            itemwidth=95,
        )
    elif n_modalities >= 3:
        # Stacked: CT-FM on top, Merlin below
        base["legend"]  = _legend_row(-0.22, "<b>CT-FM  ——</b>")
        base["legend2"] = _legend_row(-0.35, "<b>Merlin  - -</b>")
    else:
        # Side-by-side on the same line
        leg1 = _legend_row(-0.22, "<b>CT-FM  ——</b>")
        leg2 = _legend_row(-0.22, "<b>Merlin  - -</b>")
        leg2["x"] = 0.52
        base["legend"]  = leg1
        base["legend2"] = leg2
    return base


# ---------------------------------------------------------------------------
# 1. Individual plots
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Individual CT-FM vs Merlin plots")
print("=" * 60)

# all_data[source_name][(concat, layer2, frozen, mod_dir_key)] = data or None
all_data = {src: {} for src in SOURCES}

for mod_dir_key, mod_cfg in MODALITY_CONFIGS.items():
    mod_label  = mod_cfg["label"]
    modalities = mod_cfg["modalities"]
    
    for concat_layer, layer2 in ARCH_ORDER:
        arch_label = f"{concat_layer}+{layer2}"

        for frozen in [False, True]:
            freeze_label = FREEZE_LABEL[frozen]
            key = (concat_layer, layer2, frozen, mod_dir_key)

            # Load data from both sources
            src_data = {}
            for src_name, base_dir in SOURCES.items():
                mod_dir_path = os.path.join(base_dir, mod_dir_key)
                if not os.path.isdir(mod_dir_path):
                    all_data[src_name][key] = None
                    continue
                exp_dir = find_exp_dir(mod_dir_path, concat_layer, layer2, frozen)
                if exp_dir is None:
                    all_data[src_name][key] = None
                    continue
                d = load_experiment_data(exp_dir, modalities)
                all_data[src_name][key] = d
                if d is not None:
                    src_data[src_name] = d

            if not src_data:
                print(f"  [SKIP] {arch_label} {freeze_label} {mod_label}: no data in any source")
                continue

            print(f"  [PLOT] {arch_label} {freeze_label} {mod_label}  "
                  f"(sources: {list(src_data.keys())})")

            has_merlin = mod_dir_key not in NO_MERLIN_MODS

            fig = go.Figure()
            LEGEND_REF = {"CT-FM": "legend", "Merlin": "legend2"}
            for src_name in ["CT-FM", "Merlin"]:   # CT-FM drawn first (behind)
                if src_name not in src_data:
                    continue
                if not has_merlin and src_name == "Merlin":
                    continue
                add_source_traces(fig, src_data[src_name], modalities, src_name,
                                  legend_ref=LEGEND_REF[src_name])

            tick_vals, tick_text = compute_x_ticks(modalities)
            fig.update_xaxes(**base_axis("% Missing", tick_vals, tick_text))
            fig.update_yaxes(**base_axis("C-index (Mean ± SE)"))
            fig.update_layout(
                **layout_single(
                    f"{arch_label}  ·  {mod_label}  ·  {freeze_label}",
                    n_modalities=len(modalities),
                    has_merlin=has_merlin,
                )
            )

            safe = re.sub(r"[^\w\-]", "_",
                          f"{arch_label}_{freeze_label}_{mod_label}")
            fig.write_image(os.path.join(OUT_DIR, f"{safe}.png"), scale=2)

# ---------------------------------------------------------------------------
# 2. Subplot grids: rows=archs, cols=freeze, per modality_set
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Subplot grids: archs × freeze, per modality set")
print("=" * 60)

for mod_dir_key, mod_cfg in MODALITY_CONFIGS.items():
    mod_label  = mod_cfg["label"]
    modalities = mod_cfg["modalities"]

    n_archs  = len(ARCH_ORDER)
    n_freeze = 2

    subplot_titles = []
    for concat_layer, layer2 in ARCH_ORDER:
        for frozen in [False, True]:
            subplot_titles.append(
                f"<b>{concat_layer}+{layer2}</b> · {FREEZE_LABEL[frozen]}"
            )

    fig = make_subplots(
        rows=n_archs, cols=n_freeze,
        shared_xaxes=True, shared_yaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.07, horizontal_spacing=0.04,
    )

    LEGEND_REF  = {"CT-FM": "legend", "Merlin": "legend2"}
    shown_leg   = {"CT-FM": False, "Merlin": False}   # per-source, across all cells
    any_plotted = False

    for row_i, (concat_layer, layer2) in enumerate(ARCH_ORDER, start=1):
        for col_i, frozen in enumerate([False, True], start=1):
            key = (concat_layer, layer2, frozen, mod_dir_key)

            for src_name in ["CT-FM", "Merlin"]:
                d = all_data[src_name].get(key)
                if d is None:
                    continue
                # Each source shows its legend items exactly once (first cell found)
                show_leg = not shown_leg[src_name]
                add_source_traces(
                    fig, d, modalities, src_name,
                    show_legend=show_leg,
                    legend_ref=LEGEND_REF[src_name],
                    row=row_i, col=col_i,
                )
                shown_leg[src_name] = True
                any_plotted = True

    if not any_plotted:
        continue

    # Per-subplot axis formatting
    g_tick_vals, g_tick_text = compute_x_ticks(modalities)
    for row_i in range(1, n_archs + 1):
        for col_i in range(1, n_freeze + 1):
            ax_idx = (row_i - 1) * n_freeze + col_i
            x_ax = f"xaxis{ax_idx if ax_idx > 1 else ''}"
            y_ax = f"yaxis{ax_idx if ax_idx > 1 else ''}"
            fig.update_layout(**{
                x_ax: dict(
                    tickmode="array", tickvals=g_tick_vals, ticktext=g_tick_text,
                    showgrid=True, gridcolor="#ebebeb",
                    linecolor="black", linewidth=1.5, mirror=True,
                    tickfont=dict(size=14, family=FONT, color="black"),
                    **(dict(title=dict(text="<b>% Missing</b>",
                                       font=dict(size=15, family=FONT, color="black")))
                       if row_i == n_archs else {}),
                ),
                y_ax: dict(
                    showgrid=True, gridcolor="#ebebeb",
                    linecolor="black", linewidth=1.5, mirror=True,
                    tickfont=dict(size=14, family=FONT, color="black"),
                    **(dict(title=dict(text="<b>C-index</b>",
                                       font=dict(size=15, family=FONT, color="black")))
                       if col_i == 1 else {}),
                ),
            })

    for ann in fig.layout.annotations:
        ann.font = dict(size=16, family=FONT, color="black")

    fig.update_layout(
        title=dict(
            text=(f"<b>CT-FM (—) vs Merlin (- -) · "
                  f"Missing-modality performance — {mod_label}</b>"),
            font=dict(size=20, family=FONT, color="black"),
            x=0.5, xanchor="center",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend =_legend_row(-0.07, "<b>CT-FM  ——</b>", font_size=15),
        legend2=_legend_row(-0.12, "<b>Merlin  - -</b>", font_size=15),
        font=dict(size=15, family=FONT, color="black"),
        width=1050,
        height=265 * n_archs + 200,
        margin=dict(l=90, r=30, t=75, b=180),
    )

    safe_mod = re.sub(r"[^\w]", "_", mod_label)
    out_name = f"grid_{safe_mod}.png"
    fig.write_image(os.path.join(OUT_DIR, out_name), scale=2)
    print(f"  [SAVE] {out_name}")

print(f"\nAll plots saved to {OUT_DIR}/")
