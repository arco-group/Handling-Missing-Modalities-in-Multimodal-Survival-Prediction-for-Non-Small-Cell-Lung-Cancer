"""
Trimodal missing-modality comparison plot for WSI+CT+Tabular experiments.

Three CT embedding sources, each fixed to its best configuration:
  - Merlin  : ConcatODST+ODST, ms-frozen  → outputs/fold_specific/
  - CT-FM   : ConcatODST+ODST, ms-frozen  → outputs/fold_specific_ctfm/
  - CT-CLIP : ConcatFC (ms-frozen)         → outputs/fold_specific_ctclip/

One figure, three subplots (one column per masked modality: WSI, CT, Tabular).
Natural-missingness axis clipping:
  - WSI : x-axis cut from 64% to 100% (25% and 50% artificially-missing removed, too few samples).
  - CT  : x-axis cut from 5% to 100%.
  - Tab : full range 0–100%.
"""

import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR    = "outputs"
OUTPUT_PATH = "tests/ablation_wsi_ct_tab"
os.makedirs(OUTPUT_PATH, exist_ok=True)

METRIC      = "c_index"
METRIC_NAME = "C-index"
TARGET      = "OS_1d_regression"
MOD_LIST    = [1, 2, 3]   # WSI, CT, Tabular
MODAL_NAME  = {1: "WSI", 2: "CT", 3: "Tabular"}
FONT        = "Didot, GFS Didot, Bodoni MT, Times New Roman, serif"

FOLDERS_3 = [
    "0-0-0",
    "0-0-25", "0-0-50", "0-0-75", "0-0-100",
    "0-25-0", "0-50-0", "0-75-0", "0-100-0",
    "25-0-0", "50-0-0", "75-0-0", "100-0-0",
]

# Hardcoded best experiment per CT embedding source
SOURCES = {
    "Merlin" : {
        "root"   : os.path.join(BASE_DIR, "fold_specific",      "wsi_ct_tab_cox-label_cix"),
        "pattern": r"ConcatFC\+ODST",
        "freeze" : "ms-frozen",
    },
    "CT-FM"  : {
        "root"   : os.path.join(BASE_DIR, "fold_specific_ctfm", "wsi_ct_tab_cox-label_cix"),
        "pattern": r"ConcatFC\+ODST",
        "freeze" : "ms-frozen",
    },
    "CT-CLIP": {
        "root"   : os.path.join(BASE_DIR, "fold_specific_ctclip", "wsi_ct_tab_cox-label_cix"),
        "pattern": r"ConcatFC_",
        "freeze" : "ms-frozen",
    },
}

SOURCE_COLORS = {"Merlin": "#1f77b4", "CT-FM": "#e67e22", "CT-CLIP": "#2ca02c"}
SOURCE_DASH   = {"Merlin": "solid",   "CT-FM": "dot",     "CT-CLIP": "dash"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_exp_dir(root, pattern, freeze):
    if not os.path.isdir(root):
        return None
    candidates = [
        e for e in os.listdir(root)
        if re.search(pattern, e) and TARGET in e and "copy" not in e.lower()
    ]
    if freeze == "ms-frozen":
        candidates = [e for e in candidates if e.endswith("_ms-frozen")]
    else:
        candidates = [e for e in candidates if not e.endswith("_ms-frozen")]
    if not candidates:
        return None
    return os.path.join(root, sorted(candidates)[-1])


def get_results_subname(exp_dir):
    results_dir = os.path.join(exp_dir, "provided_provided", "results")
    if not os.path.isdir(results_dir):
        return None
    names = os.listdir(results_dir)
    return names[0] if names else None


def read_metric_row(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    for _, row in df.iterrows():
        return row
    return None


def load_source_data(src_name, cfg):
    exp_dir = find_exp_dir(cfg["root"], cfg["pattern"], cfg["freeze"])
    if exp_dir is None:
        print(f"  [SKIP – no exp dir] {src_name}")
        return None

    sub_name = get_results_subname(exp_dir)
    if sub_name is None:
        print(f"  [SKIP – no results dir] {src_name}")
        return None

    results = []
    for subfolder in FOLDERS_3:
        csv_path = os.path.join(
            exp_dir, "provided_provided", "results", sub_name,
            "0", subfolder, "unbalanced", "test",
            "averages_classes_average_performance.csv",
        )
        row = read_metric_row(csv_path)
        if row is None:
            continue
        parts = [int(x) for x in subfolder.split("-")]
        entry = {"subfolder": subfolder,
                 "mean": row[(METRIC, "mean")],
                 "std" : row[(METRIC, "std")]}
        for idx, mod_id in enumerate(MOD_LIST):
            entry[f"modality_{MODAL_NAME[mod_id]}_missing%"] = parts[idx]
        results.append(entry)

    if not results:
        print(f"  [SKIP – no result rows] {src_name}")
        return None

    print(f"  [OK] {src_name}  ({os.path.basename(exp_dir)})")
    df_s = pd.DataFrame(results)

    # Assign masked modality
    df_s["missing_modality"] = df_s["subfolder"].apply(
        lambda s: MOD_LIST[np.argmax([int(x) for x in s.split("-")])]
        if s != "0-0-0" else MOD_LIST[0]
    )

    # Duplicate the 0-0-0 baseline row for each modality
    df_zero    = df_s[df_s["subfolder"] == "0-0-0"].copy()
    df_nonzero = df_s[df_s["subfolder"] != "0-0-0"].copy()
    df_p = df_nonzero.copy()
    for mod_id in MOD_LIST:
        z = df_zero.copy()
        z["missing_modality"] = mod_id
        df_p = pd.concat([df_p, z], ignore_index=True)

    # Compute x position (missing percentage)
    mapper = {0: 0, 25: 25, 50: 50, 75: 75, 100: 100}
    df_p["missing_percentage"] = df_p["subfolder"].apply(
        lambda s: mapper[max(int(x) for x in s.split("-"))]
    )

    # Remove WSI 25% and 50% artificially-missing points (too few samples)
    df_p = df_p[~(
        (df_p["missing_modality"] == 1) &
        (df_p["missing_percentage"].isin([25, 50]))
    )]

    # Shift 0%-missing baseline to natural-missingness x positions
    df_p.loc[
        (df_p["missing_modality"] == 1) & (df_p["missing_percentage"] == 0),
        "missing_percentage"
    ] = 64
    df_p.loc[
        (df_p["missing_modality"] == 2) & (df_p["missing_percentage"] == 0),
        "missing_percentage"
    ] = 5

    return df_p.sort_values(["missing_modality", "missing_percentage"])


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading experiments …")
source_data = {}
for src_name, cfg in SOURCES.items():
    d = load_source_data(src_name, cfg)
    if d is not None:
        source_data[src_name] = d

# ---------------------------------------------------------------------------
# Trimodal comparison figure
# ---------------------------------------------------------------------------
subplot_titles = [f"<b>Masked: {MODAL_NAME[m]}</b>" for m in MOD_LIST]
fig = make_subplots(
    rows=1, cols=3,
    shared_yaxes=True,
    subplot_titles=subplot_titles,
    horizontal_spacing=0.04,
)

# Per-modality x-axis tick config
# WSI/CT baselines are shifted to their natural-missingness x position (64%, 5%)
mod_xaxis = {
    1: dict(tickvals=[64, 75, 100],        ticktext=["64%", "75%", "100%"],                range=[60, 102]),
    2: dict(tickvals=[5, 25, 50, 75, 100], ticktext=["5%", "25%", "50%", "75%", "100%"],  range=[1,  102]),
    3: dict(tickvals=[0, 25, 50, 75, 100], ticktext=["0%", "25%", "50%", "75%", "100%"],  range=[-2, 102]),
}

sources_ordered = [s for s in ("Merlin", "CT-FM", "CT-CLIP") if s in source_data]

for col_idx, mod_id in enumerate(MOD_LIST, start=1):
    for src in sources_ordered:
        df_p   = source_data[src]
        subset = df_p[df_p["missing_modality"] == mod_id].copy()
        if subset.empty:
            continue

        hx = SOURCE_COLORS[src].lstrip("#")
        r, g, b    = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        line_color = f"rgba({r},{g},{b},0.88)"
        fill_color = f"rgba({r},{g},{b},0.13)"
        sem        = subset["std"] / np.sqrt(5)

        x_fwd = list(subset["missing_percentage"])
        y_up  = list(subset["mean"] + sem)
        y_dn  = list(subset["mean"] - sem)

        # Shaded SE band
        fig.add_trace(go.Scatter(
            x=x_fwd + x_fwd[::-1],
            y=y_up + y_dn[::-1],
            fill="toself", fillcolor=fill_color,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col_idx)

        # Main line — legend only in first subplot to avoid duplicates
        fig.add_trace(go.Scatter(
            x=subset["missing_percentage"],
            y=subset["mean"],
            mode="markers+lines",
            marker=dict(
                size=11, color=line_color, symbol="circle",
                line=dict(width=1.2, color="rgba(0,0,0,0.5)"),
            ),
            name=src,
            legendgroup=src,
            showlegend=(col_idx == 1),
            line=dict(width=3, color=line_color, dash=SOURCE_DASH[src]),
            error_y=dict(
                type="data", array=list(sem), visible=True,
                thickness=1.5, width=6,
                color=f"rgba({r},{g},{b},0.45)",
            ),
        ), row=1, col=col_idx)

# Axis styling
axis_common = dict(
    showgrid=True, gridcolor="#e8e8e8", zeroline=False,
    linecolor="black", linewidth=1.5, mirror=True,
    tickfont=dict(size=21, family=FONT, color="black"),
)
for col_idx, mod_id in enumerate(MOD_LIST, start=1):
    cfg = mod_xaxis[mod_id]
    fig.update_xaxes(
        title_text="<b>% Missing</b>",
        title_font=dict(size=23, family=FONT, color="black"),
        tickmode="array",
        tickvals=cfg["tickvals"],
        ticktext=cfg["ticktext"],
        range=cfg["range"],
        **axis_common,
        row=1, col=col_idx,
    )

yaxis_common = {**axis_common, "mirror": False, "showline": False}
fig.update_yaxes(
    title_text=f"<b>{METRIC_NAME} (Mean ± SE)</b>",
    title_font=dict(size=23, family=FONT, color="black"),
    **yaxis_common,
    row=1, col=1,
)
# Remove y-axis title from cols 2 and 3 (shared)
for col_idx in [2, 3]:
    fig.update_yaxes(title_text="", row=1, col=col_idx)

# Subplot title font
for ann in fig.layout.annotations:
    ann.font = dict(size=23, family=FONT, color="black")

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(
        title=dict(
            text="<b>CT Embeddings</b>",
            font=dict(size=23, family=FONT, color="black"),
        ),
        orientation="h",
        yanchor="top", y=-0.22,
        xanchor="center", x=0.5,
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="rgba(0,0,0,0.20)",
        borderwidth=1.5,
        font=dict(size=23, family=FONT, color="black"),
        itemsizing="constant",
        itemwidth=70,
    ),
    font=dict(size=21, family=FONT, color="black"),
    width=1380, height=560,
    margin=dict(l=100, r=30, t=65, b=155),
)

out_png = os.path.join(OUTPUT_PATH, f"{TARGET}_trimodal_comparison.png")
fig.write_image(out_png, scale=2)
print(f"\nSaved → {out_png}")
