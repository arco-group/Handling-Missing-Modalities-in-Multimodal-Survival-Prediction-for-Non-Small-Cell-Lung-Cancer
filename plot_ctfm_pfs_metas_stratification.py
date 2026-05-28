"""
PFS / Metastasis stratification for WSI+CT+Tab ConcatFC+ODST Unfrozen (CT-FM).

Loads the aggregate 5-fold predictions for the best multimodal CT-FM model,
merges with clinical data, computes the adaptive log-rank cutoff on 5-year OS,
and produces one combined KM figure per clinical grouping:

  Any_Progression — (recidiva=='SI') OR (metastasi=='SI')
  Metastasis      — metastasi=='SI'

Each figure shows four curves using two colour families:
  High Risk + YES  →  warm dark  (#E64B35)
  High Risk + NO   →  warm light (#F4A582)
  Low Risk  + YES  →  cool dark  (#2166AC)
  Low Risk  + NO   →  cool light (#92C5DE)

Output: paper/KM/ctfm/pfs_metas/
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AGG_CSV = Path(
    "outputs/fold_specific_ctfm"
    "/wsi_ct_tab_cox-label_cix"
    "/ConcatFC+ODST_WSI+CT+tabular_multimodal_OS_1d_regression_42"
    "_multimodal_joint_fusion_classification_with_missing_generation_over_test"
    "/provided_provided/predictions"
    "/ConcatFC_noimputation_normalize_categoricalencode_separate_modalities"
    "_previously_missing_MCAR_global"
    "/aggregate_5fold_test.csv"
)
ALL_DATA_PATH = Path("data/tabular/survival/AIDA/cross_validation/all_data.xlsx")
OUT_DIR       = Path("paper/KM/ctfm/pfs_metas")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS      = 5
LIMIT_DAYS = YEARS * 365

FONT = "Didot, GFS Didot, Bodoni MT, Times New Roman, serif"

COLORS = {
    "high_yes": "#E64B35",   # warm dark  — High Risk + clinical event
    "high_no":  "#F4A582",   # warm light — High Risk + no event
    "low_yes":  "#2166AC",   # cool dark  — Low Risk  + clinical event
    "low_no":   "#92C5DE",   # cool light — Low Risk  + no event
}

GROUPINGS = [
    {
        "col":     "Any_Progression",
        "label":   "Progression (PFS)",
        "short":   "Any_Progression",
        "yes_val": "YES",
        "no_val":  "NO",
    },
    {
        "col":     "metastasi",
        "label":   "Metastasis",
        "short":   "Metastasis",
        "yes_val": "SI",
        "no_val":  "NO",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_clinical() -> pd.DataFrame:
    df = pd.read_excel(ALL_DATA_PATH)
    df = df.rename(columns={"ID paziente": "ID"})
    df["ID"]      = df["ID"].astype(int)
    df["OS_Event"] = df["status"].map({"DOD": 1, "DOC": 1, "AWD": 0, "NED": 0}).fillna(0).astype(int)
    df["OS_Time"]  = df["days2OS"].astype(float)

    # 5-year truncation
    df.loc[df["OS_Time"] > LIMIT_DAYS, "OS_Event"] = 0

    # Combined progression flag
    df["Any_Progression"] = np.where(
        (df["recidiva"] == "SI") | (df["metastasi"] == "SI"), "YES", "NO"
    )

    return df.set_index("ID")[["OS_Time", "OS_Event", "Any_Progression", "metastasi"]]


def fix_score_direction(scores: np.ndarray, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Flip scores so that high score → shorter survival."""
    median_cut = np.median(scores)
    high_mask  = scores > median_cut
    low_mask   = ~high_mask
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return scores
    if time[high_mask].mean() > time[low_mask].mean():
        return -scores
    return scores


def find_optimal_cutoff_logrank(
    scores: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    min_percent: int = 10,
    max_percent: int = 90,
) -> float | None:
    percentiles       = np.arange(min_percent, max_percent + 1, 2)
    candidate_cutoffs = np.percentile(scores, percentiles)
    best_cutoff, best_stat = None, -np.inf
    for c in candidate_cutoffs:
        low_mask  = scores <= c
        high_mask = scores > c
        if low_mask.sum() < 10 or high_mask.sum() < 10:
            continue
        try:
            res = logrank_test(
                time[high_mask], time[low_mask],
                event_observed_A=event[high_mask],
                event_observed_B=event[low_mask],
            )
            if res.test_statistic > best_stat:
                best_stat   = res.test_statistic
                best_cutoff = c
        except Exception:
            continue
    return best_cutoff


def overall_logrank_p(subgroups: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Compute overall log-rank p-value across multiple groups."""
    groups_arr, times_arr, events_arr = [], [], []
    for g_id, (t, e) in enumerate(subgroups):
        groups_arr.append(np.full(len(t), g_id))
        times_arr.append(t)
        events_arr.append(e)
    all_groups = np.concatenate(groups_arr)
    all_times  = np.concatenate(times_arr)
    all_events = np.concatenate(events_arr)
    try:
        res = multivariate_logrank_test(all_times, all_groups, all_events)
        return float(res.p_value)
    except Exception:
        return 1.0


def make_four_curve_figure(df: pd.DataFrame, grouping: dict) -> go.Figure:
    """
    Build a KM figure with 4 curves: {High,Low} Risk × {YES,NO} clinical event.
    df must have columns: OS_Time, OS_Event, Risk_Group, <grouping['col']>
    """
    col     = grouping["col"]
    label   = grouping["label"]
    yes_val = grouping["yes_val"]
    no_val  = grouping["no_val"]

    subgroups_def = [
        ("High Risk", yes_val, "High Risk + event",    COLORS["high_yes"]),
        ("High Risk", no_val,  "High Risk + no event", COLORS["high_no"]),
        ("Low Risk",  yes_val, "Low Risk + event",     COLORS["low_yes"]),
        ("Low Risk",  no_val,  "Low Risk + no event",  COLORS["low_no"]),
    ]

    # Collect (T, E) arrays for overall log-rank
    subgroup_data = []
    for risk, clin_val, _, _ in subgroups_def:
        mask = (df["Risk_Group"] == risk) & (df[col] == clin_val)
        subgroup_data.append((df.loc[mask, "OS_Time"].values, df.loc[mask, "OS_Event"].values))

    p_val = overall_logrank_p(subgroup_data)

    # Pairwise log-rank: YES vs NO within each risk group
    def pairwise_p(t1, e1, t2, e2) -> float:
        if len(t1) == 0 or len(t2) == 0:
            return 1.0
        try:
            return float(logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2).p_value)
        except Exception:
            return 1.0

    T_hy, E_hy = subgroup_data[0]   # High Risk + YES
    T_hn, E_hn = subgroup_data[1]   # High Risk + NO
    T_ly, E_ly = subgroup_data[2]   # Low Risk  + YES
    T_ln, E_ln = subgroup_data[3]   # Low Risk  + NO

    p_high = pairwise_p(T_hy, E_hy, T_hn, E_hn)
    p_low  = pairwise_p(T_ly, E_ly, T_ln, E_ln)

    fig = go.Figure()
    kmf = KaplanMeierFitter()

    for (risk, clin_val, trace_label, color), (T, E) in zip(subgroups_def, subgroup_data):
        if len(T) == 0:
            continue
        kmf.fit(T, E, label=trace_label)

        fig.add_trace(go.Scatter(
            x    = kmf.survival_function_.index.tolist(),
            y    = kmf.survival_function_[trace_label].tolist(),
            mode = "lines",
            name = trace_label,
            line = dict(color=color, width=4, shape="hv"),
        ))

        # Censoring tick-marks
        mask_cens = (df["Risk_Group"] == risk) & (df[col] == clin_val) & (df["OS_Event"] == 0)
        censored  = df[mask_cens]
        if not censored.empty:
            cens_probs = [float(kmf.predict(t)) for t in censored["OS_Time"]]
            fig.add_trace(go.Scatter(
                x          = censored["OS_Time"].tolist(),
                y          = cens_probs,
                mode       = "markers",
                showlegend = False,
                marker     = dict(symbol="line-ns-open", color=color, size=14,
                                  line=dict(width=2.5)),
            ))

    axis_style = dict(
        showgrid=True, gridcolor="#e8e8e8", gridwidth=1,
        zeroline=False, linecolor="black", linewidth=1.5, mirror=True,
        tickfont=dict(size=27, family=FONT, color="black"),
        title_font=dict(size=29, family=FONT, color="black"),
    )

    fig.update_layout(
        xaxis=dict(title="<b>Time (days)</b>", **axis_style),
        yaxis=dict(title="<b>Survival Probability</b>", range=[0, 1.05], **axis_style),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=23, family=FONT, color="black"),
        title=dict(
            text=f"<b>WSI+CT+Tab</b>",
            font=dict(size=30, family=FONT, color="black"),
            x=0.5, xanchor="center",
        ),
        width=1050, height=700,
        margin=dict(l=110, r=50, t=90, b=90),
        legend=dict(
            x=0.98, y=0.98, xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.90)", bordercolor="black", borderwidth=1,
            font=dict(size=22, family=FONT),
        ),
        annotations=[
            dict(
                x=0.5, y=0.98, xref="paper", yref="paper",
                xanchor="center", yanchor="top",
                text=(
                    "<span style='color:{c_hy}'><b>High Risk YES vs NO: p = {ph}{sh}</b></span><br>"
                    "<span style='color:{c_ly}'><b>Low Risk YES vs NO: p = {pl}{sl}</b></span>"
                ).format(
                    c_hy=COLORS["high_yes"], ph=f"{p_high:.3f}", sh="*" if p_high < 0.05 else "",
                    c_ly=COLORS["low_yes"],  pl=f"{p_low:.3f}",  sl="*" if p_low  < 0.05 else "",
                ),
                showarrow=False,
                font=dict(size=22, family=FONT, color="black"),
                bgcolor="rgba(255,255,255,0.88)", bordercolor="black", borderwidth=1,
            ),
        ],
    )

    return fig, p_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading clinical data...")
    clinical = load_clinical()
    print(f"  {len(clinical)} patients loaded.")

    print(f"\nLoading predictions from:\n  {AGG_CSV}")
    if not AGG_CSV.exists():
        print("ERROR: aggregate CSV not found. Run aggregate_5fold_ctfm.py first.")
        return

    preds = pd.read_csv(AGG_CSV)
    preds["ID"]    = preds["ID"].astype(int)
    preds["score"] = pd.to_numeric(preds["probability"], errors="coerce")
    preds = preds.set_index("ID")[["score"]].dropna()
    print(f"  {len(preds)} predictions loaded.")

    common_ids = clinical.index.intersection(preds.index)
    print(f"  {len(common_ids)} patients in common.")

    if len(common_ids) < 20:
        print("ERROR: Not enough patients in common.")
        return

    df = clinical.loc[common_ids].copy()
    df["score"] = preds.loc[common_ids, "score"]

    # Force direction: high score = high hazard
    df["score"] = fix_score_direction(
        df["score"].values, df["OS_Time"].values, df["OS_Event"].values
    )

    # Adaptive cutoff on 5-year OS
    cutoff = find_optimal_cutoff_logrank(
        df["score"].values, df["OS_Time"].values, df["OS_Event"].values
    )
    if cutoff is None:
        cutoff = float(np.median(df["score"].values))
        print("  [WARN] No valid adaptive cutoff found, using median.")
    else:
        print(f"  Adaptive cutoff: {cutoff:.4f}")

    df["Risk_Group"] = np.where(df["score"] > cutoff, "High Risk", "Low Risk")
    n_high = (df["Risk_Group"] == "High Risk").sum()
    n_low  = (df["Risk_Group"] == "Low Risk").sum()
    print(f"  High Risk: {n_high} patients | Low Risk: {n_low} patients\n")

    for grouping in GROUPINGS:
        fig, p_val = make_four_curve_figure(df, grouping)
        out_path   = OUT_DIR / f"WSI_CT_Tab_ConcatFC_ODST_Unfrozen_{grouping['short']}.png"
        fig.write_image(str(out_path), scale=2)
        p_str = f"{p_val:.3f}{'*' if p_val < 0.05 else ''}"
        print(f"  [SAVE] {out_path.name}  p={p_str}")

    print(f"\nDone. Plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
