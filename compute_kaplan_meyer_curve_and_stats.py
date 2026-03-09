import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1) Helper Functions
# ============================================================

def find_optimal_cutoff_logrank(scores, time, event, min_percent=20, max_percent=80):
    """
    Finds the optimal cutoff to stratify patients based on maximum survival separation.
    Iterates through percentiles of the score distribution.
    """
    # We look for cutoffs between the 20th and 80th percentile to avoid tiny groups
    percentiles = np.arange(min_percent, max_percent + 1, 2)  # Step of 2 for speed
    candidate_cutoffs = np.percentile(scores, percentiles)

    best_cutoff = None
    best_stat = -np.inf

    for c in candidate_cutoffs:
        # Group 1: Low Risk (<= cutoff), Group 2: High Risk (> cutoff)
        low_mask = scores <= c
        high_mask = scores > c

        # Safety check: ensure both groups have at least 10 samples
        if low_mask.sum() < 10 or high_mask.sum() < 10:
            continue

        try:
            res = logrank_test(
                    time[high_mask], time[low_mask],
                    event_observed_A=event[high_mask],
                    event_observed_B=event[low_mask]
            )

            # We maximize the test statistic (Chi2)
            if res.test_statistic > best_stat:
                best_stat = res.test_statistic
                best_cutoff = c
        except:
            continue

    return best_cutoff, best_stat


# ============================================================
# 2) Setup & Data Loading
# ============================================================
path_reports = Path("outputs/fold_specific")
path_plots = Path("outputs/cox_stratified_plots")
path_plots.mkdir(parents=True, exist_ok=True)

# Find prediction files
path_reports = Path("outputs/predictions")
csv_files = list(path_reports.rglob("*pred_aggregated_score.csv"))
#csv_files = [f for f in csv_files if not 'frozen' in f.parts]


loaded_predictions = []
for file in csv_files:
    # Load predictions
    df = pd.read_csv(file, index_col='ID')

    # Handle duplicates if any (taking mean of predictions)
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    if df.index.duplicated().any():
        df = df.groupby(level=0)['score'].mean().to_frame()
    else:
        df = df[['score']]

    loaded_predictions.append((file.parts[2], df))

# Load Ground Truth
global_data_path = "data/tabular/survival/AIDA/cross_validation/all_data.xlsx"
time_and_events = pd.read_excel(global_data_path)

# Ensure ID column is handled correctly
if 'ID paziente' in time_and_events.columns:
    time_and_events = time_and_events.set_index('ID paziente')
time_and_events.index.name = 'ID'

# Map Status/Event
# 1 = Event (Death), 0 = Censored (Alive/NED)
time_and_events['status_binary'] = time_and_events['status'].map({'DOD': 1, 'DOC': 1, 'AWD': 0, 'NED': 0}).fillna(0)
time_and_events['OS_Time'] = time_and_events['days2OS']

# ============================================================
# 3) Main Analysis Loop
# ============================================================
# We analyze at specific time horizons (Years)
years_to_analyze = [5]

for y in years_to_analyze:
    limit_days = y * 365
    print(f"\n--- Processing {y}-Year Horizon ---")

    for (exp_name, df_preds) in loaded_predictions:

        # Merge Predictions with Ground Truth
        common_ids = time_and_events.index.intersection(df_preds.index)
        if len(common_ids) < 20:
            print(f"Skipping {exp_name}: Not enough common patients.")
            continue

        df_merged = df_preds.loc[common_ids].copy()
        df_merged['time'] = time_and_events.loc[common_ids, 'OS_Time']
        df_merged['event'] = time_and_events.loc[common_ids, 'status_binary']

        # --- A. Time Truncation (Analysis specific to Y-years) ---
        # If a patient lives > limit_days, they are censored at limit_days
        # If a patient dies > limit_days, they are censored at limit_days (for the context of N-year survival)
        analysis_df = df_merged.copy()
        mask_exceed = analysis_df['time'] > limit_days
        analysis_df.loc[mask_exceed, 'event'] = 0
        # analysis_df.loc[mask_exceed, 'time'] = limit_days

        # --- B. Find Optimal Cutoff ---
        # Since scores are -1 to 1 (Cox output), higher score = higher hazard
        cutoff = analysis_df['score'].median()  # Default fallback
        cutoff, stat = find_optimal_cutoff_logrank(
                analysis_df['score'].values,
                analysis_df['time'].values,
                analysis_df['event'].values
        )

        if cutoff is None:
            print(f"  {exp_name}: No valid cutoff found.")
            continue

        # --- C. Stratify ---
        # Group 1 (High Risk): Score > Cutoff
        # Group 0 (Low Risk): Score <= Cutoff
        analysis_df['group'] = (analysis_df['score'] > cutoff).astype(int)

        # --- D. Statistics ---
        # 1. Log-Rank Test
        T = analysis_df['time']
        E = analysis_df['event']
        G = analysis_df['group']

        try:
            lr_result = logrank_test(
                    T[G == 1], T[G == 0],
                    event_observed_A=E[G == 1], event_observed_B=E[G == 0]
            )
            p_val = lr_result.p_value
        except:
            p_val = 1.0

        # 2. Hazard Ratio (re-fitting a univariable Cox on the binary group)
        try:
            cph = CoxPHFitter()
            cph.fit(analysis_df[['time', 'event', 'group']], duration_col='time', event_col='event')
            hr = cph.hazard_ratios_['group']
        except:
            hr = np.nan

        # --- E. Plotting ---
        kmf = KaplanMeierFitter()
        fig = go.Figure()

        # Colors: Green for Low Risk (0), Red for High Risk (1)
        colors = {0: '#4DBBD5', 1: '#E64B35'}
        labels = {
                0: f"Low Risk",
                1: f"High Risk"
        }

        for g in [0, 1]:
            mask = analysis_df['group'] == g
            if mask.sum() == 0:
                continue

            kmf.fit(analysis_df.loc[mask, 'time'], analysis_df.loc[mask, 'event'])

            fig.add_trace(go.Scatter(
                    x=kmf.survival_function_.index,
                    y=kmf.survival_function_["KM_estimate"].values,
                    mode='lines',
                    name=f"{labels[g]}",
                    line=dict(color=colors[g], width=4, shape='hv')
            ))

            # Add Censoring Markers
            censored = analysis_df[mask & (analysis_df['event'] == 0)]
            if not censored.empty:
                # Map censored times to probabilities
                cens_times = censored['time']
                cens_probs = [kmf.predict(t) for t in cens_times]

                fig.add_trace(go.Scatter(
                        x=cens_times, y=cens_probs,
                        mode='markers',
                        showlegend=False,
                        marker=dict(symbol='line-ns-open', color=colors[g], size=14, line=dict(width=2))
                ))

        fig.update_layout(
                xaxis=dict(

                        showgrid=True, gridcolor='lightgrey',
                        tickfont=dict(size=30, family="Arial"),
                        title_font=dict(size=26, family="Arial", color="black"),
                ),
                yaxis=dict(
                        title="<b>Survival Probability</b>",
                        range=[0, 1.05],
                        showgrid=True, gridcolor='lightgrey',
                        tickfont=dict(size=30, family="Arial"),
                        title_font=dict(size=26, family="Arial", color="black"),
                ),
                plot_bgcolor='white',

                # 🔥 Bigger figure for a clean, professional KM curve
                width=1400,
                height=800,

                # 🔥 Better legend
                legend=dict(
                        x=0.02, y=0.05,
                        bgcolor='rgba(255,255,255,0.85)',
                        bordercolor='black',
                        borderwidth=1,
                        font=dict(size=28, family="Arial")
                ),

                # 🔥 Large, clear annotation box
                annotations=[
                        dict(
                                x=0.95, y=0.95, xref='paper', yref='paper',
                                text=f'<b>Log-Rank p = {p_val:.4f}</b>',
                                showarrow=False,
                                font=dict(size=28, family="Arial", color='black'),
                                bgcolor='rgba(255, 255, 255, 0.85)',
                                bordercolor='black',
                                borderwidth=1,
                                align="right"
                        )
                ],

                margin=dict(l=100, r=80, t=80, b=80)
        )

        out_path = path_plots / f"{exp_name}_Y{y}_Stratification.png"
        fig.write_image(out_path, scale=2)
        print(f"  Saved: {out_path.name} (p={p_val:.4e})")