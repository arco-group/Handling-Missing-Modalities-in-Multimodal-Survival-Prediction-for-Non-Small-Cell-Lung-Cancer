"""
Kaplan-Meier stratification plots for CT-FM aggregated predictions.

For each selected (modality_set × frozen) experiment, produces two separate figures:
  {name}_adaptive.png — optimal log-rank cutoff (10th–90th percentile sweep)
  {name}_median.png   — median score cutoff

Direction is forced: scores are flipped when high score correlates with better
(not worse) survival, guaranteeing High Risk always has the lower KM curve.

Only the 5-year survival window is considered.
Only the architecture combinations from SELECTED_ARCH are plotted (CT-FM only).

Output: paper/KM/ctfm/
"""

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR      = Path("outputs/fold_specific_ctfm")
ALL_DATA_PATH = Path("data/tabular/survival/AIDA/cross_validation/all_data.xlsx")
OUT_DIR       = Path("paper/KM/ctfm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET     = "OS_1d_regression"
YEARS      = 5
LIMIT_DAYS = YEARS * 365

FONT = "Didot, GFS Didot, Bodoni MT, Times New Roman, serif"

FREEZE_LABEL = {False: "Unfrozen", True: "Frozen_MS"}

# Regex to extract concat_layer and layer2 from experiment dir name
ARCH_RE = re.compile(r"^(Concat\w+)\+(\w+)_")

COLORS = {
    "high": "#E64B35",  # red  — High Risk
    "low" : "#4DBBD5",  # blue — Low Risk
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mod_key_to_label(mod_key: str) -> str:
    """'wsi_ct_tab_cox-label_cix' → 'WSI+CT+Tab'"""
    parts = mod_key.replace("_cox-label_cix", "").split("_")
    return "+".join(p.upper() if p in ("wsi", "ct") else p.capitalize() for p in parts)


def find_aggregate_csv(exp_dir: Path) -> Path | None:
    for provided_dir in exp_dir.iterdir():
        if not provided_dir.is_dir():
            continue
        pred_root = provided_dir / "predictions"
        if not pred_root.exists():
            continue
        for pred_subdir in pred_root.iterdir():
            if not pred_subdir.is_dir():
                continue
            csv = pred_subdir / "aggregate_5fold_test.csv"
            if csv.exists():
                return csv
    return None


def load_ground_truth() -> pd.DataFrame:
    df = pd.read_excel(ALL_DATA_PATH)
    df = df.rename(columns={"ID paziente": "ID"})
    df["ID"]    = df["ID"].astype(int)
    df["event"] = df["status"].map({"DOD": 1, "DOC": 1, "AWD": 0, "NED": 0}).fillna(0).astype(int)
    df["time"]  = df["days2OS"].astype(float)
    return df.set_index("ID")[["time", "event"]]


def fix_score_direction(scores: np.ndarray, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """
    Guarantee that high score = high hazard (lower survival).
    Strategy: split at median, compare median survival time of each half.
    If the high-score half lives longer on average, flip the sign.
    """
    median_cut  = np.median(scores)
    high_mask   = scores > median_cut
    low_mask    = ~high_mask

    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return scores

    # Use mean observed time as a proxy for survival (higher = better survival)
    mean_time_high = time[high_mask].mean()
    mean_time_low  = time[low_mask].mean()

    if mean_time_high > mean_time_low:
        # High-score group lives longer → score is inverted, flip it
        return -scores
    return scores


def find_optimal_cutoff_logrank(
    scores: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    min_percent: int = 10,
    max_percent: int = 90,
) -> tuple[float | None, float]:
    """Sweep 10th–90th percentile, return cutoff that maximises log-rank chi2."""
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
    return best_cutoff, best_stat


def make_km_figure(
    df: pd.DataFrame,
    cutoff: float,
    title: str,
) -> tuple[go.Figure, float, int, int]:
    """
    Build a single KM figure for the given score cutoff.
    Returns (fig, p_value, n_high, n_low).
    """
    df = df.copy()
    df["group"] = (df["score"] > cutoff).astype(int)

    T, E, G = df["time"], df["event"], df["group"]

    try:
        lr    = logrank_test(T[G == 1], T[G == 0],
                             event_observed_A=E[G == 1], event_observed_B=E[G == 0])
        p_val = lr.p_value
    except Exception:
        p_val = 1.0

    fig = go.Figure()
    kmf = KaplanMeierFitter()
    n_counts: dict[int, int] = {}

    for g, label, color in [(1, "High Risk", COLORS["high"]),
                             (0, "Low Risk",  COLORS["low"])]:
        mask = G == g
        n_counts[g] = int(mask.sum())
        if mask.sum() == 0:
            continue
        kmf.fit(T[mask], E[mask], label=label)

        fig.add_trace(go.Scatter(
            x    = kmf.survival_function_.index.tolist(),
            y    = kmf.survival_function_[label].tolist(),
            mode = "lines",
            name = label,
            line = dict(color=color, width=4, shape="hv"),
        ))

        # Censoring tick-marks
        censored = df[mask & (E == 0)]
        if not censored.empty:
            cens_probs = [float(kmf.predict(t)) for t in censored["time"]]
            fig.add_trace(go.Scatter(
                x          = censored["time"].tolist(),
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
            text=title,
            font=dict(size=30, family=FONT, color="black"),
            x=0.5, xanchor="center",
        ),
        width=900,
        height=700,
        margin=dict(l=110, r=50, t=80, b=90),
        showlegend=False,
        annotations=[dict(
            x=0.98, y=0.06,
            xref="paper", yref="paper",
            xanchor="right", yanchor="bottom",
            text=f"<b>Log-rank p = {p_val:.3f}{'*' if p_val < 0.05 else ''}</b>",
            showarrow=False,
            font=dict(size=27, family=FONT, color="black"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="black",
            borderwidth=1,
        )],
    )

    return fig, p_val, n_counts.get(1, 0), n_counts.get(0, 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gt = load_ground_truth()
    print(f"Ground truth loaded: {len(gt)} patients.\n")

    modality_dirs = sorted([
        d for d in BASE_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".") and "late" not in d.name.lower()
    ])

    for mod_dir_path in modality_dirs:
        mod_label = mod_key_to_label(mod_dir_path.name)

        exp_dirs = sorted([
            d for d in mod_dir_path.iterdir()
            if d.is_dir()
            and ARCH_RE.match(d.name)
            and TARGET in d.name
            and "copy" not in d.name.lower()
        ])

        if not exp_dirs:
            continue

        print(f"\n=== {mod_label} ===")

        for exp_dir in exp_dirs:
            m            = ARCH_RE.match(exp_dir.name)
            concat_layer = m.group(1)
            layer2       = m.group(2)
            frozen       = exp_dir.name.endswith("_ms-frozen")
            freeze_lbl   = FREEZE_LABEL[frozen]

            csv_path = find_aggregate_csv(exp_dir)
            if csv_path is None:
                print(f"  [SKIP] {concat_layer}+{layer2} {freeze_lbl}: aggregate_5fold_test.csv not found")
                continue

            # ----------------------------------------------------------------
            # Load & prepare
            # ----------------------------------------------------------------
            preds = pd.read_csv(csv_path)
            preds["ID"]    = preds["ID"].astype(int)
            preds["score"] = pd.to_numeric(preds["probability"], errors="coerce")
            preds = preds.set_index("ID")[["score"]].dropna()

            common_ids = gt.index.intersection(preds.index)
            if len(common_ids) < 20:
                print(f"  [SKIP] {concat_layer}+{layer2} {freeze_lbl}: only {len(common_ids)} common patients")
                continue

            df = preds.loc[common_ids].copy()
            df["time"]  = gt.loc[common_ids, "time"]
            df["event"] = gt.loc[common_ids, "event"]

            # 5-year truncation: patients surviving beyond limit are censored
            df.loc[df["time"] > LIMIT_DAYS, "event"] = 0

            # Force direction: high score = high hazard
            df["score"] = fix_score_direction(
                df["score"].values, df["time"].values, df["event"].values
            )

            scores = df["score"].values
            times  = df["time"].values
            events = df["event"].values

            # ----------------------------------------------------------------
            # Cutoffs
            # ----------------------------------------------------------------
            adaptive_cutoff, _ = find_optimal_cutoff_logrank(scores, times, events)
            median_cutoff      = float(np.median(scores))

            if adaptive_cutoff is None:
                print(f"  [WARN] {concat_layer}+{layer2} {freeze_lbl}: no valid adaptive cutoff, using median")
                adaptive_cutoff = median_cutoff

            safe_name = re.sub(r"[^\w\-]", "_", f"{mod_label}_{concat_layer}+{layer2}_{freeze_lbl}")

            # ----------------------------------------------------------------
            # Adaptive plot
            # ----------------------------------------------------------------
            fig_a, p_a, _, _ = make_km_figure(df, adaptive_cutoff,
                title=f"<b>{mod_label}</b>")
            out_a = OUT_DIR / f"{safe_name}_adaptive.png"
            fig_a.write_image(str(out_a), scale=2)
            print(f"  [SAVE] {out_a.name}  p={p_a:.3f}{'*' if p_a < 0.05 else ''}  n={len(df)}")

            # ----------------------------------------------------------------
            # Median plot
            # ----------------------------------------------------------------
            fig_m, p_m, _, _ = make_km_figure(df, median_cutoff,
                title=f"<b>{mod_label}</b>")
            out_m = OUT_DIR / f"{safe_name}_median.png"
            fig_m.write_image(str(out_m), scale=2)
            print(f"  [SAVE] {out_m.name}  p={p_m:.3f}{'*' if p_m < 0.05 else ''}  n={len(df)}")

    print(f"\nDone. Plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
