"""
Ablation study and missing-modality performance plots for WSI+CT+Tabular experiments.
Covers:
  - fold_specific/wsi_ct_tab_cox-label_cix/  (ConcatFC+FC, ConcatFC+ODST, ConcatODST+FC, ConcatODST+ODST, frozen/unfrozen)
  - fold_specific_ctfm/wsi_ct_tab_cox-label_cix/  (ConcatFC, frozen/unfrozen — CTFM embeddings)
Experiments with fewer than 3 missing-% result subfolders are skipped.
"""

import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR       = "outputs"
RESULTS_SPLIT  = "fold_specific"
RESULTS_CTFM   = "fold_specific_ctfm"
OUTPUT_PATH    = "tests/ablation_wsi_ct_tab"
os.makedirs(OUTPUT_PATH, exist_ok=True)

METRIC      = "c_index"
METRIC_NAME = "C-index"
TARGET      = "OS_1d_regression"
MOD_LIST    = [1, 2, 3]   # WSI, CT, Tabular
MODAL_NAME  = {1: "WSI", 2: "CT", 3: "Tabular"}
FOLDERS_3   = [
    "0-0-0",
    "0-0-25", "0-0-50", "0-0-75", "0-0-100",
    "0-25-0", "0-50-0", "0-75-0", "0-100-0",
    "25-0-0", "50-0-0", "75-0-0", "100-0-0",
]
COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}   # WSI=blue, CT=green, Tab=red
FONT   = "Didot, GFS Didot, Bodoni MT, Times New Roman, serif"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_exp_dir(root_dir, model_pattern, target, freeze):
    """Return the last matching experiment directory path or None."""
    if not os.path.isdir(root_dir):
        return None
    all_exp = os.listdir(root_dir)
    exps = [e for e in all_exp if re.search(model_pattern, e) and target in e]
    if freeze == "ms-frozen":
        exps = [e for e in exps if "ms-frozen" in e]
    else:
        exps = [e for e in exps if "ms-frozen" not in e and "copy" not in e]
    if not exps:
        return None
    return os.path.join(root_dir, sorted(exps)[-1])


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
        return row   # first (only) row
    return None


def count_missing_subfolders(exp_dir, sub_name):
    folder_0 = os.path.join(exp_dir, "provided_provided", "results", sub_name, "0")
    if not os.path.isdir(folder_0):
        return 0
    return sum(
        1 for f in os.listdir(folder_0)
        if os.path.isdir(os.path.join(folder_0, f)) and re.match(r"[\d]+-[\d]+-[\d]+", f)
    )

# ---------------------------------------------------------------------------
# Experiment catalogue
# ---------------------------------------------------------------------------
#  Each entry: (label, root_dir, model_pattern, freeze, ct_source)
#  ct_source: "clip" = standard embeddings, "ctfm" = CTFM embeddings
EXPERIMENTS = []

_root_merlin = os.path.join(BASE_DIR, RESULTS_SPLIT,       "wsi_ct_tab_cox-label_cix")
_root_ctclip = os.path.join(BASE_DIR, "fold_specific_ctclip", "wsi_ct_tab_cox-label_cix")
_root_ctfm   = os.path.join(BASE_DIR, RESULTS_CTFM,           "wsi_ct_tab_cox-label_cix")

# Merlin CT embeddings — multiple architectures
for arch in ["ConcatFC\\+FC", "ConcatFC\\+ODST", "ConcatODST\\+FC", "ConcatODST\\+ODST"]:
    arch_clean = arch.replace("\\", "")
    for freeze in ["unfrozen", "ms-frozen"]:
        EXPERIMENTS.append({
            "label"  : f"{arch_clean} ({freeze})",
            "arch"   : arch_clean,
            "freeze" : freeze,
            "root"   : _root_merlin,
            "pattern": arch,
            "ct_src" : "Merlin",
        })

# CT-CLIP embeddings
for freeze in ["unfrozen", "ms-frozen"]:
    EXPERIMENTS.append({
        "label"  : f"ConcatFC ({freeze}, CT-CLIP)",
        "arch"   : "ConcatFC",
        "freeze" : freeze,
        "root"   : _root_ctclip,
        "pattern": r"ConcatFC_",
        "ct_src" : "CT-CLIP",
    })

# CTFM embeddings
for freeze in ["unfrozen", "ms-frozen"]:
    EXPERIMENTS.append({
        "label"  : f"ConcatFC ({freeze}, CTFM)",
        "arch"   : "ConcatFC",
        "freeze" : freeze,
        "root"   : _root_ctfm,
        "pattern": r"ConcatFC_",
        "ct_src" : "CTFM",
    })

# ---------------------------------------------------------------------------
# 1. ABLATION TABLE  (0-0-0 results = no missing data at test time)
# ---------------------------------------------------------------------------
ablation_rows = []

for exp in EXPERIMENTS:
    exp_dir = find_exp_dir(exp["root"], exp["pattern"], TARGET, exp["freeze"])
    if exp_dir is None:
        print(f"[SKIP – not found] {exp['label']}")
        continue

    sub_name = get_results_subname(exp_dir)
    if sub_name is None:
        print(f"[SKIP – no results dir] {exp['label']}")
        continue

    n_folders = count_missing_subfolders(exp_dir, sub_name)
    if n_folders <= 2:
        print(f"[SKIP – only {n_folders} missing-% folders] {exp['label']}")
        continue

    csv_path = os.path.join(
        exp_dir, "provided_provided", "results", sub_name,
        "0", "0-0-0", "unbalanced", "test", "averages_classes_average_performance.csv"
    )
    row = read_metric_row(csv_path)
    if row is None:
        print(f"[SKIP – missing CSV] {exp['label']}")
        continue

    se = row[(METRIC, "std")] / np.sqrt(5)
    ablation_rows.append({
        "Model"        : exp["label"],
        "Architecture" : exp["arch"],
        "Freeze"       : exp["freeze"],
        "CT embeddings": exp["ct_src"],
        f"{METRIC_NAME} mean": round(row[(METRIC, "mean")], 3),
        f"{METRIC_NAME} SE"  : round(se, 3),
    })
    print(f"[OK] {exp['label']}  C-index={row[(METRIC,'mean')]:.3f} ± {row[(METRIC,'std')]:.3f}")

df_ablation = pd.DataFrame(ablation_rows)
ablation_csv = os.path.join(OUTPUT_PATH, "ablation_wsi_ct_tab_table.csv")
df_ablation.to_csv(ablation_csv, index=False)
print(f"\nAblation table saved -> {ablation_csv}")
print(df_ablation.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. MISSING-MODALITY PERFORMANCE PLOTS (one plot per experiment)
# ---------------------------------------------------------------------------
mapper = {0: 0, 25: 25, 50: 50, 75: 75, 100: 100}

for exp in EXPERIMENTS:
    exp_dir = find_exp_dir(exp["root"], exp["pattern"], TARGET, exp["freeze"])
    if exp_dir is None:
        continue

    sub_name = get_results_subname(exp_dir)
    if sub_name is None:
        continue

    n_folders = count_missing_subfolders(exp_dir, sub_name)
    if n_folders <= 2:
        continue

    results = []
    for subfolder in FOLDERS_3:
        csv_path = os.path.join(
            exp_dir, "provided_provided", "results", sub_name,
            "0", subfolder, "unbalanced", "test", "averages_classes_average_performance.csv"
        )
        row = read_metric_row(csv_path)
        if row is None:
            continue
        parts = [int(x) for x in subfolder.split("-")]
        entry = {
            "subfolder": subfolder,
            "mean"     : row[(METRIC, "mean")],
            "std"      : row[(METRIC, "std")],
        }
        for idx, mod_id in enumerate(MOD_LIST):
            entry[f"modality_{MODAL_NAME[mod_id]}_missing%"] = parts[idx]
        results.append(entry)

    if not results:
        continue

    df_r = pd.DataFrame(results)
    df_r["setting"] = df_r["subfolder"].str.replace("-", "_")

    # assign which modality is being masked
    df_r["missing_modality"] = df_r["subfolder"].apply(
        lambda s: MOD_LIST[np.argmax([int(x) for x in s.split("-")])]
        if s != "0-0-0" else MOD_LIST[0]
    )

    df_zero = df_r[df_r["subfolder"] == "0-0-0"].copy()
    df_nonzero = df_r[df_r["subfolder"] != "0-0-0"].copy()

    df_plot = df_nonzero.copy()
    for mod_id in MOD_LIST:
        z = df_zero.copy()
        z["missing_modality"] = mod_id
        df_plot = pd.concat([df_plot, z], ignore_index=True)

    df_plot["missing_percentage"] = df_plot["subfolder"].apply(
        lambda s: mapper[max(int(x) for x in s.split("-"))]
    )

    # WSI x-axis adjustments (same as original script)
    df_plot = df_plot[~((df_plot["missing_modality"] == 1) & (df_plot["missing_percentage"].isin([25, 50])))]
    df_plot.loc[(df_plot["missing_modality"] == 1) & (df_plot["missing_percentage"] == 0), "missing_percentage"] = 64
    df_plot.loc[(df_plot["missing_modality"] == 2) & (df_plot["missing_percentage"] == 0), "missing_percentage"] = 5
    df_plot = df_plot.sort_values(["missing_modality", "missing_percentage"])

    fig = go.Figure()
    for mod_id in sorted(df_plot["missing_modality"].unique()):
        subset = df_plot[df_plot["missing_modality"] == mod_id]
        sem = subset["std"] / np.sqrt(5)
        hex_col = COLORS[mod_id].lstrip("#")
        r, g, b = int(hex_col[0:2], 16), int(hex_col[2:4], 16), int(hex_col[4:6], 16)
        line_color  = f"rgba({r},{g},{b},0.85)"
        error_color = f"rgba({r},{g},{b},0.45)"
        fill_color  = f"rgba({r},{g},{b},0.12)"
        # shaded SE band
        x_fwd = list(subset["missing_percentage"])
        x_rev = x_fwd[::-1]
        y_up  = list(subset["mean"] + sem)
        y_dn  = list(subset["mean"] - sem)
        fig.add_trace(go.Scatter(
            x=x_fwd + x_rev,
            y=y_up + y_dn[::-1],
            fill="toself",
            fillcolor=fill_color,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=subset["missing_percentage"],
            y=subset["mean"],
            mode="markers+lines",
            marker=dict(size=10, color=line_color,
                        line=dict(width=1, color="rgba(0,0,0,0.6)"), symbol="circle"),
            name=MODAL_NAME[mod_id],
            line=dict(width=2.5, color=line_color, dash="solid"),
        ))

    fig.update_layout(
        xaxis=dict(
            title="<b>% Missing</b>",
            tickmode="array", tickvals=[0, 25, 50, 75, 100],
            showgrid=True, gridcolor="#e8e8e8",
            zeroline=False,
            title_font=dict(size=28, family=FONT, color="black"),
            tickfont=dict(size=23, family=FONT, color="black"),
            linecolor="black", linewidth=1, mirror=True,
        ),
        yaxis=dict(
            title=f"<b>{METRIC_NAME} (Mean ± SE)</b>",
            showgrid=True, gridcolor="#e8e8e8",
            zeroline=False,
            title_font=dict(size=28, family=FONT, color="black"),
            tickfont=dict(size=23, family=FONT, color="black"),
            linecolor="black", linewidth=1, mirror=True,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            title=dict(
                text="<b>Masked Modality</b>",
                font=dict(size=24, family=FONT, color="black"),
            ),
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=22, family=FONT, color="black"),
            itemsizing="constant",
            itemwidth=50,
        ),
        font=dict(size=22, family=FONT, color="black"),
        width=980, height=580,
        margin=dict(l=100, r=30, t=30, b=130),
    )

    safe_label = re.sub(r"[^\w\-]", "_", exp["label"])
    out_png = os.path.join(OUTPUT_PATH, f"{TARGET}_{safe_label}_missing_perf.png")
    fig.write_image(out_png, scale=2)
    print(f"Saved plot -> {out_png}")

# ---------------------------------------------------------------------------
# 3. COMBINED COMPARISON PLOT  — all valid experiments overlaid (0-0-0 only)
# ---------------------------------------------------------------------------
# (Bar chart of mean c-index with error bars, grouped by CT embedding source)

if not df_ablation.empty:
    # Color bars by CT embedding source
    source_palettes = {
        "Merlin" : ["#1f77b4", "#4e9fd6", "#7bbce0", "#a8d4ef", "#5ba3c9", "#2e86c1", "#85c1e9"],
        "CT-CLIP": ["#2ca02c", "#5abf5a"],
        "CTFM"   : ["#e67e22", "#f0a050"],
    }
    source_counts = {}

    # Sort by mean descending for readability
    df_plot_bar = df_ablation.sort_values(f"{METRIC_NAME} mean", ascending=True).reset_index(drop=True)

    bar_colors = []
    for _, r in df_plot_bar.iterrows():
        src = r["CT embeddings"]
        idx = source_counts.get(src, 0)
        pal = source_palettes.get(src, ["#888888"])
        bar_colors.append(pal[idx % len(pal)])
        source_counts[src] = idx + 1

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_plot_bar[f"{METRIC_NAME} mean"],
        y=df_plot_bar["Model"],
        orientation="h",
        error_x=dict(
            type="data",
            array=df_plot_bar[f"{METRIC_NAME} SE"],
            visible=True, thickness=1.5, width=5,
            color="rgba(0,0,0,0.55)",
        ),
        marker_color=bar_colors,
        marker_line=dict(width=0.8, color="rgba(0,0,0,0.3)"),
        opacity=0.88,
        showlegend=False,
    ))

    # add subtle value annotations at end of bars
    for _, r in df_plot_bar.iterrows():
        fig_bar.add_annotation(
            x=r[f"{METRIC_NAME} mean"] + r[f"{METRIC_NAME} SE"] + 0.3,
            y=r["Model"],
            text=f"{r[f'{METRIC_NAME} mean']:.1f}",
            showarrow=False,
            font=dict(size=13, family=FONT, color="black"),
            xanchor="left",
        )

    fig_bar.update_layout(
        xaxis=dict(
            title=f"<b>{METRIC_NAME} (no missing data, Mean ± SE)</b>",
            showgrid=True, gridcolor="#e8e8e8",
            zeroline=False,
            range=[55, max(df_plot_bar[f"{METRIC_NAME} mean"]) + 6],
            title_font=dict(size=20, family=FONT, color="black"),
            tickfont=dict(size=16, family=FONT, color="black"),
            linecolor="black", linewidth=1, mirror=True,
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=15, family=FONT, color="black"),
            automargin=True,
            linecolor="black", linewidth=1, mirror=True,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=15, family=FONT, color="black"),
        width=950,
        height=max(420, 55 * len(df_plot_bar)),
        margin=dict(l=20, r=80, t=50, b=60),
    )
    bar_png = os.path.join(OUTPUT_PATH, f"{TARGET}_ablation_bar_chart.png")
    fig_bar.write_image(bar_png, scale=2)
    print(f"Saved bar chart -> {bar_png}")

# ---------------------------------------------------------------------------
# 4. COMBINED TRIMODAL COMPARISON PLOT
#    One curve per CT embedding source (Merlin / CT-CLIP / CTFM).
#    Best model per source = highest 0-0-0 C-index mean.
#    All three share the same missing-modality x-axis; modalities encoded by
#    line style, sources encoded by colour.
# ---------------------------------------------------------------------------

SOURCE_COLORS = {"Merlin": "#1f77b4", "CT-CLIP": "#2ca02c", "CTFM": "#e67e22"}
SOURCE_DASH   = {"Merlin": "solid",   "CT-CLIP": "dash",    "CTFM": "dot"}
MOD_MARKERS   = {1: "circle", 2: "square", 3: "diamond"}

# Pick best (highest 0-0-0 mean) experiment per source
best_per_source = {}   # source -> exp dict
for exp in EXPERIMENTS:
    src = exp["ct_src"]
    exp_dir = find_exp_dir(exp["root"], exp["pattern"], TARGET, exp["freeze"])
    if exp_dir is None:
        continue
    sub_name = get_results_subname(exp_dir)
    if sub_name is None:
        continue
    if count_missing_subfolders(exp_dir, sub_name) <= 2:
        continue
    csv_0 = os.path.join(exp_dir, "provided_provided", "results", sub_name,
                         "0", "0-0-0", "unbalanced", "test",
                         "averages_classes_average_performance.csv")
    row0 = read_metric_row(csv_0)
    if row0 is None:
        continue
    mean0 = row0[(METRIC, "mean")]
    if src not in best_per_source or mean0 > best_per_source[src]["mean0"]:
        best_per_source[src] = {"exp": exp, "exp_dir": exp_dir,
                                "sub_name": sub_name, "mean0": mean0}

print("\n--- Best model per CT source ---")
for src, info in best_per_source.items():
    print(f"  {src}: {info['exp']['label']}  (C-index 0-miss = {info['mean0']:.3f})")

# Pre-load per-source data frames
source_data = {}   # src -> df_p (with missing_modality & missing_percentage)

for src, info in best_per_source.items():
    exp_dir  = info["exp_dir"]
    sub_name = info["sub_name"]

    results = []
    for subfolder in FOLDERS_3:
        csv_path = os.path.join(exp_dir, "provided_provided", "results", sub_name,
                                "0", subfolder, "unbalanced", "test",
                                "averages_classes_average_performance.csv")
        row = read_metric_row(csv_path)
        if row is None:
            continue
        parts = [int(x) for x in subfolder.split("-")]
        entry = {"subfolder": subfolder,
                 "mean": row[(METRIC, "mean")],
                 "std":  row[(METRIC, "std")]}
        for idx, mod_id in enumerate(MOD_LIST):
            entry[f"modality_{MODAL_NAME[mod_id]}_missing%"] = parts[idx]
        results.append(entry)

    if not results:
        continue

    df_s = pd.DataFrame(results)
    df_s["missing_modality"] = df_s["subfolder"].apply(
        lambda s: MOD_LIST[np.argmax([int(x) for x in s.split("-")])]
        if s != "0-0-0" else MOD_LIST[0]
    )
    df_zero    = df_s[df_s["subfolder"] == "0-0-0"].copy()
    df_nonzero = df_s[df_s["subfolder"] != "0-0-0"].copy()

    df_p = df_nonzero.copy()
    for mod_id in MOD_LIST:
        z = df_zero.copy()
        z["missing_modality"] = mod_id
        df_p = pd.concat([df_p, z], ignore_index=True)

    df_p["missing_percentage"] = df_p["subfolder"].apply(
        lambda s: mapper[max(int(x) for x in s.split("-"))]
    )
    df_p = df_p[~((df_p["missing_modality"] == 1) & (df_p["missing_percentage"].isin([25, 50])))]
    df_p.loc[(df_p["missing_modality"] == 1) & (df_p["missing_percentage"] == 0), "missing_percentage"] = 64
    df_p.loc[(df_p["missing_modality"] == 2) & (df_p["missing_percentage"] == 0), "missing_percentage"] = 5
    df_p = df_p.sort_values(["missing_modality", "missing_percentage"])
    source_data[src] = df_p

# --- Three-subplot figure: one column per masked modality ---
from plotly.subplots import make_subplots

subplot_titles = [f"<b>Masked: {MODAL_NAME[m]}</b>" for m in MOD_LIST]
fig_tri = make_subplots(
    rows=1, cols=3,
    shared_yaxes=True,
    subplot_titles=subplot_titles,
    horizontal_spacing=0.04,
)

sources_ordered = [s for s in ("Merlin", "CT-CLIP", "CTFM") if s in source_data]

for col_idx, mod_id in enumerate(MOD_LIST, start=1):
    for src in sources_ordered:
        df_p = source_data[src]
        subset = df_p[df_p["missing_modality"] == mod_id].copy()
        if subset.empty:
            continue

        hex_col    = SOURCE_COLORS[src].lstrip("#")
        r, g, b    = int(hex_col[0:2], 16), int(hex_col[2:4], 16), int(hex_col[4:6], 16)
        line_color = f"rgba({r},{g},{b},0.85)"
        fill_color = f"rgba({r},{g},{b},0.12)"
        sem        = subset["std"] / np.sqrt(5)

        x_fwd = list(subset["missing_percentage"])
        y_up  = list(subset["mean"] + sem)
        y_dn  = list(subset["mean"] - sem)

        # shaded SE band
        fig_tri.add_trace(go.Scatter(
            x=x_fwd + x_fwd[::-1],
            y=y_up + y_dn[::-1],
            fill="toself", fillcolor=fill_color,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col_idx)

        # main line — show legend only in first subplot to avoid duplicates
        fig_tri.add_trace(go.Scatter(
            x=subset["missing_percentage"],
            y=subset["mean"],
            mode="markers+lines",
            marker=dict(size=9, color=line_color,
                        line=dict(width=1, color="rgba(0,0,0,0.45)"),
                        symbol="circle"),
            name=src,
            legendgroup=src,
            showlegend=(col_idx == 1),
            line=dict(width=2.5, color=line_color, dash=SOURCE_DASH[src]),
        ), row=1, col=col_idx)

# Per-modality x-axis tick configuration
# WSI: 0% missing displayed at x=64, then 75, 100 (25/50 removed)
# CT:  0% missing displayed at x=5,  then 25, 50, 75, 100
# Tab: 0% missing displayed at x=0,  then 25, 50, 75, 100
mod_xaxis = {
    1: dict(tickvals=[64, 75, 100],           ticktext=["0%", "75%", "100%"]),
    2: dict(tickvals=[5, 25, 50, 75, 100],    ticktext=["0%", "25%", "50%", "75%", "100%"]),
    3: dict(tickvals=[0, 25, 50, 75, 100],    ticktext=["0%", "25%", "50%", "75%", "100%"]),
}

axis_common = dict(
    showgrid=True, gridcolor="#e8e8e8", zeroline=False,
    linecolor="black", linewidth=1, mirror=True,
    tickfont=dict(size=20, family=FONT, color="black"),
)
for col_idx, mod_id in enumerate(MOD_LIST, start=1):
    cfg = mod_xaxis[mod_id]
    fig_tri.update_xaxes(
        title_text="<b>% Missing</b>",
        title_font=dict(size=22, family=FONT, color="black"),
        tickmode="array",
        tickvals=cfg["tickvals"],
        ticktext=cfg["ticktext"],
        **axis_common,
        row=1, col=col_idx,
    )
fig_tri.update_yaxes(
    title_text=f"<b>{METRIC_NAME} (Mean ± SE)</b>",
    title_font=dict(size=22, family=FONT, color="black"),
    showgrid=True, gridcolor="#e8e8e8", zeroline=False,
    linecolor="black", linewidth=1, mirror=True,
    tickfont=dict(size=20, family=FONT, color="black"),
    row=1, col=1,
)

# Style subplot title annotations (modality names above each panel)
for ann in fig_tri.layout.annotations:
    ann.font = dict(size=22, family=FONT, color="black")

fig_tri.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(
        title=dict(text="<b>CT Embeddings</b>",
                   font=dict(size=22, family=FONT, color="black")),
        orientation="h",
        yanchor="top", y=-0.22,
        xanchor="center", x=0.5,
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        font=dict(size=22, family=FONT, color="black"),
        itemsizing="constant",
        itemwidth=60,
    ),
    font=dict(size=20, family=FONT, color="black"),
    width=1350, height=540,
    margin=dict(l=90, r=30, t=60, b=140),
)

tri_png = os.path.join(OUTPUT_PATH, f"{TARGET}_trimodal_comparison_Merlin_CTCLIP_CTFM.png")
fig_tri.write_image(tri_png, scale=2)
print(f"Saved trimodal comparison plot -> {tri_png}")
