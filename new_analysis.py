import pandas as pd
import numpy as np
from scipy.stats import chi2
from lifelines.statistics import logrank_test
import plotly.graph_objects as go
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. Helper Functions (Stats & Math)
# ============================================================

def calculate_logrank(t1, e1, t2, e2):
    """
    Calculates the log-rank test p-value for two groups (Manual calculation).
    """
    # Combine data
    df1 = pd.DataFrame({'time': t1, 'event': e1, 'group': 1})
    df2 = pd.DataFrame({'time': t2, 'event': e2, 'group': 2})
    df = pd.concat([df1, df2])

    # Get unique event times
    times = df[df['event'] == 1]['time'].unique()
    times.sort()

    observed_1 = 0
    expected_1 = 0
    variance_1 = 0

    for t in times:
        n1 = len(df[(df['group'] == 1) & (df['time'] >= t)])
        n2 = len(df[(df['group'] == 2) & (df['time'] >= t)])
        n = n1 + n2

        o1 = len(df[(df['group'] == 1) & (df['time'] == t) & (df['event'] == 1)])
        o2 = len(df[(df['group'] == 2) & (df['time'] == t) & (df['event'] == 1)])
        o = o1 + o2

        if n > 0:
            e1_t = o * (n1 / n)
            v1_t = (e1_t * (n2 / n) * (n - o)) / (n - 1) if n > 1 else 0

            observed_1 += o1
            expected_1 += e1_t
            variance_1 += v1_t

    if variance_1 > 0:
        z = (observed_1 - expected_1) / np.sqrt(variance_1)
        chi2_stat = z ** 2
        p_val = 1 - chi2.cdf(chi2_stat, df=1)
    else:
        p_val = 1.0

    return p_val


def get_km_curve(t, e):
    """Calculates Kaplan-Meier survival curve coordinates."""
    df = pd.DataFrame({'time': t, 'event': e})
    df = df.sort_values('time')
    times = [0]
    probs = [1.0]
    unique_times = df['time'].unique()
    current_prob = 1.0

    for time in unique_times:
        at_risk = len(df[df['time'] >= time])
        events = len(df[(df['time'] == time) & (df['event'] == 1)])
        if at_risk > 0:
            current_prob *= (1 - events / at_risk)
        times.append(time)
        probs.append(current_prob)

    return times, probs


def find_optimal_cutoff_logrank(scores, time, event, min_percent=20, max_percent=80):
    """Finds cutoff based on best logrank separation."""
    percentiles = np.arange(min_percent, max_percent + 1, 1)
    candidate_cutoffs = np.percentile(scores, percentiles)
    best_cutoff = None
    best_stat = -np.inf

    for c in candidate_cutoffs:
        low_mask = scores <= c
        high_mask = scores > c
        if low_mask.sum() < 5 or high_mask.sum() < 5:
            continue

        # Using lifelines for speed in optimization loop
        res = logrank_test(
                time[high_mask], time[low_mask],
                event_observed_A=event[high_mask],
                event_observed_B=event[low_mask]
        )
        if res.test_statistic > best_stat:
            best_stat = res.test_statistic
            best_cutoff = c

    return best_cutoff


# ============================================================
# 2. Data Loading
# ============================================================

# A. Clinical Data
print("Loading Clinical Data...")
clinical_path = 'data/tabular/survival/AIDA/cross_validation/all_data.xlsx'
df_all_data = pd.read_excel(clinical_path)

# Prepare Standard Survival Columns
df_all_data['OS_Event'] = df_all_data['status'].apply(lambda x: 1 if x in ['DOC', 'DOD'] else 0)
df_all_data['OS_Time'] = df_all_data['days2OS']

# Time Truncation (as per your first script)
df_all_data.loc[df_all_data['OS_Time'] > 1825, 'OS_Event'] = 0
df_all_data['OS_Time'] = np.where(df_all_data['OS_Time'] > 1825, 1825, df_all_data['OS_Time'])

# Define Progression Groups
df_all_data['Any_Progression'] = np.where(
        (df_all_data['recidiva'] == 'SI') | (df_all_data['metastasi'] == 'SI'), 'YES', 'NO'
)

# Ensure ID is the index for merging
if 'ID paziente' in df_all_data.columns:
    df_all_data = df_all_data.set_index('ID paziente')
df_all_data.index.name = 'ID'

# B. Model Predictions
print("Loading Model Predictions...")
path_reports = Path("outputs/predictions")
csv_files = list(path_reports.rglob("*pred_aggregated_score.csv"))

#csv_files = [f for f in csv_files if not 'frozen' in f.parts]

loaded_predictions = []
for file in csv_files:
    df = pd.read_csv(file, index_col='ID')
    df['score'] = pd.to_numeric(df['score'], errors='coerce')

    # Handle duplicates if any
    if df.index.duplicated().any():
        df = df.groupby(level=0)['score'].mean().to_frame()
    else:
        df = df[['score']]

    loaded_predictions.append((file.parts[2], df))

# ============================================================
# 3. Main Analysis Loop
# ============================================================

# Define the clinical comparisons we want to make
groupings = [
        {'col': 'Any_Progression', 'label': 'Progression', 'yes_val': 'YES', 'no_val': 'NO'},
        {'col': 'metastasi', 'label': 'Metastasis', 'yes_val': 'SI', 'no_val': 'NO'}
]

# Create output directory
output_dir = Path("outputs/risk_stratified_plots")
output_dir.mkdir(parents=True, exist_ok=True)

for (exp_name, df_preds) in loaded_predictions:
    print(f"\n--- Processing Model: {exp_name} ---")

    # Merge Clinical Data with Model Scores
    common_ids = df_all_data.index.intersection(df_preds.index)
    if len(common_ids) < 10:
        print("Not enough intersecting patients.")
        continue

    df_merged = df_all_data.loc[common_ids].copy()
    df_merged['score'] = df_preds.loc[common_ids, 'score']

    # Calculate Optimal Cutoff for this Model
    cutoff = find_optimal_cutoff_logrank(
            df_merged['score'].values,
            df_merged['OS_Time'].values,
            df_merged['OS_Event'].values
    )

    if cutoff is None:
        print("Could not find optimal cutoff.")
        continue

    # Assign Risk Groups
    df_merged['Risk_Group'] = np.where(df_merged['score'] > cutoff, 'High Risk', 'Low Risk')

    # --- PLOTTING LOOP ---
    # We create plots for High Risk patients, then Low Risk patients
    for risk_cat in ['High Risk', 'Low Risk']:

        # Filter for the specific risk group
        df_subset = df_merged[df_merged['Risk_Group'] == risk_cat].copy()

        if len(df_subset) < 5:
            print(f"Skipping {risk_cat} group (too few patients).")
            continue

        for group in groupings:
            col = group['col']
            label = group['label']
            yes_val = group['yes_val']
            no_val = group['no_val']

            # Prepare file name
            filename = output_dir / f"{exp_name}_{risk_cat.replace(' ', '')}_{label.replace(' ', '')}.png"

            # Split Data (YES Event vs NO Event within the Risk Group)
            mask_yes = df_subset[col] == yes_val
            mask_no = df_subset[col] == no_val

            T_yes = df_subset[mask_yes]['OS_Time'].values
            E_yes = df_subset[mask_yes]['OS_Event'].values

            T_no = df_subset[mask_no]['OS_Time'].values
            E_no = df_subset[mask_no]['OS_Event'].values

            if len(T_yes) == 0 or len(T_no) == 0:
                continue

            # Calculate Stats (Log-rank)
            p_val = calculate_logrank(T_yes, E_yes, T_no, E_no)

            # Calculate KM Curves
            times_yes, probs_yes = get_km_curve(T_yes, E_yes)
            times_no, probs_no = get_km_curve(T_no, E_no)

            # --- PLOTLY VISUALIZATION (Strictly following Script 1 style) ---
            fig = go.Figure()

            color_yes = '#E64B35'  # Red/Orange
            color_no = '#4DBBD5'  # Blue/Teal

            # 1. Trace for YES Group
            fig.add_trace(go.Scatter(
                    x=times_yes, y=probs_yes, mode='lines', name=f'{label}: YES',
                    line=dict(shape='hv', color=color_yes, width=3)
            ))

            # 2. Trace for NO Group
            fig.add_trace(go.Scatter(
                    x=times_no, y=probs_no, mode='lines', name=f'{label}: NO',
                    line=dict(shape='hv', color=color_no, width=3)
            ))

            # 3. Censoring Markers (YES)
            censored_yes_times = df_subset[mask_yes & (df_subset['OS_Event'] == 0)]['OS_Time']
            if len(censored_yes_times) > 0:
                censored_yes_probs = [probs_yes[np.searchsorted(times_yes, t, side='right') - 1] for t in censored_yes_times]
                fig.add_trace(go.Scatter(
                        x=censored_yes_times, y=censored_yes_probs, mode='markers',
                        name='Censored (YES)', showlegend=False,
                        marker=dict(symbol='line-ns-open', color=color_yes, size=10, line=dict(width=2))
                ))

            # 4. Censoring Markers (NO)
            censored_no_times = df_subset[mask_no & (df_subset['OS_Event'] == 0)]['OS_Time']
            if len(censored_no_times) > 0:
                censored_no_probs = [probs_no[np.searchsorted(times_no, t, side='right') - 1] for t in censored_no_times]
                fig.add_trace(go.Scatter(
                        x=censored_no_times, y=censored_no_probs, mode='markers',
                        name='Censored (NO)', showlegend=False,
                        marker=dict(symbol='line-ns-open', color=color_no, size=10, line=dict(width=2))
                ))

            # 5. Layout (Nature Journal Style)

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

            # Save static image
            fig.write_image(str(filename), scale=3)
            print(f"Saved: {filename.name} (p={p_val:.4f})")