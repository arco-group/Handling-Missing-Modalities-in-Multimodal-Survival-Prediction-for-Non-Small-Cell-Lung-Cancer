import pandas as pd
import numpy as np
from lifelines.statistics import logrank_test
import plotly.graph_objects as go
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. Configuration & Data Loading
# ============================================================
# File path from your previous snippet
file_path = "/Users/filruff/Desktop/PHD/PROGETTI/Multimodal_MARIA_AIDA/data/tabular/survival/AIDA/tabular/clinical_data_III_stage.csv"

# Output folder
output_dir = Path("outputs/clinical_km_plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Load Data
df = pd.read_csv(file_path, header=0)

# Clean column names (remove spaces for easier handling)
df.columns = df.columns.str.strip()

# ============================================================
# 2. Preprocessing Outcomes
# ============================================================
# 1. Define Event (1=Dead, 0=Alive)
# Adjust 'status' values based on your dataset (DOC/DOD = Dead)
df['OS_Event'] = df['status'].apply(lambda x: 1 if x in ['DOC', 'DOD'] else 0)

# 2. Define Time
# Assuming 'days2OS' exists based on your workflow. If named differently, update here.
if 'days2OS' in df.columns:
    df['OS_Time'] = df['days2OS']
else:
    # Fallback if specific column name differs, try to find it
    possible_time_cols = [c for c in df.columns if 'days' in c.lower() or 'os' in c.lower()]
    print(f"Warning: 'days2OS' not found. Using {possible_time_cols[0]}")
    df['OS_Time'] = df[possible_time_cols[0]]

# 3. Truncate Time (Optional, matching previous logic) at 5 years
MAX_DAYS = 1825
df.loc[df['OS_Time'] > MAX_DAYS, 'OS_Event'] = 0  # Censor events after cutoff
df['OS_Time'] = np.where(df['OS_Time'] > MAX_DAYS, MAX_DAYS, df['OS_Time'])


# ============================================================
# 3. Helper Functions
# ============================================================

def get_km_curve(t, e):
    """Calculates KM coordinates manually for Plotly."""
    df_km = pd.DataFrame({'time': t, 'event': e}).sort_values('time')
    times = [0]
    probs = [1.0]
    current_prob = 1.0

    unique_times = df_km['time'].unique()

    for time in unique_times:
        at_risk = len(df_km[df_km['time'] >= time])
        events = len(df_km[(df_km['time'] == time) & (df_km['event'] == 1)])

        if at_risk > 0:
            current_prob *= (1 - events / at_risk)

        times.append(time)
        probs.append(current_prob)

    return times, probs


def determine_risk_direction(df_subset, col, val1, val2):
    """
    Determines which value represents 'High Risk' (worse survival)
    by comparing the survival rate at the median time point.
    Returns: (high_risk_val, low_risk_val)
    """
    t_median = df_subset['OS_Time'].median()

    # Calculate simple survival prob at median time for both
    def get_surv_at_t(v):
        mask = df_subset[col] == v
        t = df_subset[mask]['OS_Time'].values
        e = df_subset[mask]['OS_Event'].values
        if len(t) == 0:
            return 0
        _, probs = get_km_curve(t, e)
        # Return last prob
        return probs[-1]

    surv1 = get_surv_at_t(val1)
    surv2 = get_surv_at_t(val2)

    # Lower survival = High Risk
    if surv1 < surv2:
        return val1, val2
    else:
        return val2, val1


# ============================================================
# 4. Clinical Features to Plot
# ============================================================
# Format: (Column Name, Display Label)
clinical_features = [
        ("Sesso", "Sex"),
        ("Diagnosi", "Histology"),
        ("Tecnica", "RT Technique"),
        ("Immuno_post_radiochemio_(adj)", "Adjuvant Immunotherapy")
]

print("Generating Clinical Feature Plots...")

for col, label in clinical_features:
    if col not in df.columns:
        print(f"Skipping {label}: Column '{col}' not found.")
        continue

    # Drop NaNs for this specific column
    df_plot = df.dropna(subset=[col, 'OS_Time', 'OS_Event']).copy()

    # Get Unique Groups
    groups = df_plot[col].unique()

    # Logic for selecting groups:
    # If exactly 2 groups, compare them.
    # If > 2 groups, take the top 2 most frequent (to maintain the Red/Blue binary style)
    if len(groups) > 2:
        top_2 = df_plot[col].value_counts().head(2).index.tolist()
        df_plot = df_plot[df_plot[col].isin(top_2)]
        groups = top_2
        print(f"  Note: {label} has >2 categories. Comparing top 2: {groups}")
    elif len(groups) < 2:
        print(f"  Skipping {label}: Only 1 category found.")
        continue

    val_a, val_b = groups[0], groups[1]

    # Determine which is High Risk (Red) and Low Risk (Blue)
    high_risk_val, low_risk_val = determine_risk_direction(df_plot, col, val_a, val_b)

    # Extract Data
    mask_high = df_plot[col] == high_risk_val
    mask_low = df_plot[col] == low_risk_val

    T_high, E_high = df_plot[mask_high]['OS_Time'], df_plot[mask_high]['OS_Event']
    T_low, E_low = df_plot[mask_low]['OS_Time'], df_plot[mask_low]['OS_Event']

    # Log-Rank Test
    res = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)
    p_val = res.p_value

    # Get KM Curves
    times_high, probs_high = get_km_curve(T_high, E_high)
    times_low, probs_low = get_km_curve(T_low, E_low)

    # --- PLOTTING ---
    fig = go.Figure()

    color_high = '#E64B35'  # Red (Nature Style)
    color_low = '#4DBBD5'  # Blue (Nature Style)

    # 1. High Risk Trace (Worse Survival)
    fig.add_trace(go.Scatter(
            x=times_high, y=probs_high, mode='lines',
            name=f'{high_risk_val} (High Risk)',
            line=dict(shape='hv', color=color_high, width=3)
    ))

    # 2. Low Risk Trace (Better Survival)
    fig.add_trace(go.Scatter(
            x=times_low, y=probs_low, mode='lines',
            name=f'{low_risk_val} (Low Risk)',
            line=dict(shape='hv', color=color_low, width=3)
    ))

    # 3. Censoring Markers (High Risk)
    cens_high_t = T_high[E_high == 0]
    if len(cens_high_t) > 0:
        cens_high_p = [probs_high[np.searchsorted(times_high, t, side='right') - 1] for t in cens_high_t]
        fig.add_trace(go.Scatter(
                x=cens_high_t, y=cens_high_p, mode='markers', showlegend=False,
                marker=dict(symbol='line-ns-open', color=color_high, size=10, line=dict(width=2))
        ))

    # 4. Censoring Markers (Low Risk)
    cens_low_t = T_low[E_low == 0]
    if len(cens_low_t) > 0:
        cens_low_p = [probs_low[np.searchsorted(times_low, t, side='right') - 1] for t in cens_low_t]
        fig.add_trace(go.Scatter(
                x=cens_low_t, y=cens_low_p, mode='markers', showlegend=False,
                marker=dict(symbol='line-ns-open', color=color_low, size=10, line=dict(width=2))
        ))

    # 5. Layout
    fig.update_layout(
            title=dict(
                    text=f'<b>Overall Survival by {label}</b>',
                    font=dict(size=20, family="Arial")
            ),
            xaxis=dict(title="Time (Days)", showgrid=True, gridcolor='lightgrey', linecolor='black', ticks='outside'),
            yaxis=dict(title="Survival Probability", range=[0, 1.05], showgrid=True, gridcolor='lightgrey', linecolor='black', ticks='outside'),
            plot_bgcolor='white',
            width=800, height=600,
            legend=dict(
                    x=0.02, y=0.02,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black', borderwidth=1
            ),
            annotations=[
                    dict(
                            x=0.95, y=0.95, xref='paper', yref='paper',
                            text=f'<b>Log-Rank p = {p_val:.1e}</b>',
                            showarrow=False,
                            font=dict(size=14, color='black'),
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='black', borderwidth=1
                    )
            ]
    )

    # Save
    clean_label = label.replace(" ", "_").replace("(", "").replace(")", "")
    filename = output_dir / f"KM_Clinical_{clean_label}.png"
    fig.write_image(str(filename), scale=3)
    print(f"Saved: {filename.name} | High Risk defined as: {high_risk_val}")

print("\nProcessing Complete.")