import pandas as pd
import numpy as np
from scipy.stats import chi2
import plotly.graph_objects as go


# --- 1. Helper Functions (Unchanged) ---

def calculate_logrank(t1, e1, t2, e2):
    """
    Calculates the log-rank test p-value for two groups.
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
        # At risk
        n1 = len(df[(df['group'] == 1) & (df['time'] >= t)])
        n2 = len(df[(df['group'] == 2) & (df['time'] >= t)])
        n = n1 + n2

        # Events at this time
        o1 = len(df[(df['group'] == 1) & (df['time'] == t) & (df['event'] == 1)])
        o2 = len(df[(df['group'] == 2) & (df['time'] == t) & (df['event'] == 1)])
        o = o1 + o2

        if n > 0:
            e1_t = o * (n1 / n)
            v1_t = (e1_t * (n2 / n) * (n - o)) / (n - 1) if n > 1 else 0

            observed_1 += o1
            expected_1 += e1_t
            variance_1 += v1_t

    # Chi-square statistic
    if variance_1 > 0:
        z = (observed_1 - expected_1) / np.sqrt(variance_1)
        chi2_stat = z ** 2
        p_val = 1 - chi2.cdf(chi2_stat, df=1)
    else:
        p_val = 1.0

    return p_val


def get_km_curve(t, e):
    """
    Calculates Kaplan-Meier survival curve.
    """
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


# --- 2. Main Logic & Plotly Visualization ---

# Load the dataframe.
# Ensure this path is correct for your local environment
df_all_data = pd.read_excel('data/tabular/survival/AIDA/cross_validation/all_data.xlsx')

# Status mapping
df_all_data['OS_Event'] = df_all_data['status'].apply(lambda x: 1 if x in ['DOC', 'DOD'] else 0)
df_all_data['OS_Time'] = df_all_data['days2OS']
df_all_data['OS_Event'][df_all_data['OS_Time'] >1825] =0  # Ensure non-negative times
df_all_data['OS_Time'] = np.where(df_all_data['OS_Time'] > 1825, 1825, df_all_data['OS_Time'])


# PFS Group Logic (Any Progression)
df_all_data['Any_Progression'] = np.where(
        (df_all_data['recidiva'] == 'SI') | (df_all_data['metastasi'] == 'SI'),
        'YES',
        'NO'
)

groupings = [
        #{'col': 'Any_Progression', 'label': 'Progression (PFS Group)', 'yes_val': 'YES', 'no_val': 'NO', 'file': 'km_pfs.html'},
        {'col': 'metastasi', 'label': 'Metastasis', 'yes_val': 'SI', 'no_val': 'NO', 'file': 'km_metastasis.html'},
        {'col': 'recidiva', 'label': 'Local Progression', 'yes_val': 'SI', 'no_val': 'NO', 'file': 'km_local_progression.html'}
]

print("Log-Rank Results:")

for group in groupings:
    col = group['col']
    label = group['label']
    yes_val = group['yes_val']
    no_val = group['no_val']
    filename = group['file']

    # Split Data
    T_yes = df_all_data[df_all_data[col] == yes_val]['OS_Time'].values
    E_yes = df_all_data[df_all_data[col] == yes_val]['OS_Event'].values

    T_no = df_all_data[df_all_data[col] == no_val]['OS_Time'].values
    E_no = df_all_data[df_all_data[col] == no_val]['OS_Event'].values

    # Calculate Stats
    p_val = calculate_logrank(T_yes, E_yes, T_no, E_no)
    print(f"{label}: p-value = {p_val:.5f}")

    # Calculate Curves
    times_yes, probs_yes = get_km_curve(T_yes, E_yes)
    times_no, probs_no = get_km_curve(T_no, E_no)

    # --- PLOTLY VISUALIZATION ---
    fig = go.Figure()

    # Define Colors (Nature Style: High Contrast)
    color_yes = '#E64B35'  # Red/Orange
    color_no = '#4DBBD5'  # Blue/Teal

    # 1. Trace for YES Group (Line)
    fig.add_trace(go.Scatter(
            x=times_yes, y=probs_yes,
            mode='lines',
            name=f'{label}: YES',
            line=dict(shape='hv', color=color_yes, width=3)  # 'hv' creates the step function
    ))

    # 2. Trace for NO Group (Line)
    fig.add_trace(go.Scatter(
            x=times_no, y=probs_no,
            mode='lines',
            name=f'{label}: NO',
            line=dict(shape='hv', color=color_no, width=3)
    ))

    # 3. Censoring Markers (YES Group)
    censored_yes_times = df_all_data[(df_all_data[col] == yes_val) & (df_all_data['OS_Event'] == 0)]['OS_Time']
    # Map censored times to their probability on the curve
    censored_yes_probs = [probs_yes[np.searchsorted(times_yes, t, side='right') - 1] for t in censored_yes_times]

    fig.add_trace(go.Scatter(
            x=censored_yes_times, y=censored_yes_probs,
            mode='markers',
            name='Censored (YES)',
            showlegend=False,
            marker=dict(symbol='line-ns-open', color=color_yes, size=10, line=dict(width=2))
    ))

    # 4. Censoring Markers (NO Group)
    censored_no_times = df_all_data[(df_all_data[col] == no_val) & (df_all_data['OS_Event'] == 0)]['OS_Time']
    censored_no_probs = [probs_no[np.searchsorted(times_no, t, side='right') - 1] for t in censored_no_times]

    fig.add_trace(go.Scatter(
            x=censored_no_times, y=censored_no_probs,
            mode='markers',
            name='Censored (NO)',
            showlegend=False,
            marker=dict(symbol='line-ns-open', color=color_no, size=10, line=dict(width=2))
    ))

    # 5. Layout and Styling (Nature Journal Style)
    fig.update_layout(
            title=dict(
                    text=f'<b>Overall Survival by {label}</b>',
                    font=dict(size=20, family="Arial")
            ),
            xaxis=dict(
                    title="Time (Days)",
                    showgrid=True,
                    gridcolor='lightgrey',
                    linecolor='black',
                    ticks='outside'
            ),
            yaxis=dict(
                    title="Survival Probability",
                    range=[0, 1.05],
                    showgrid=True,
                    gridcolor='lightgrey',
                    linecolor='black',
                    ticks='outside'
            ),
            plot_bgcolor='white',  # Clean white background
            width=800,
            height=600,
            legend=dict(
                    x=0.02, y=0.02,  # Bottom left position
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1
            ),
            # Add P-value Annotation
            annotations=[
                    dict(
                            x=0.95, y=0.95,
                            xref='paper', yref='paper',
                            text=f'<b>Log-Rank p = {p_val:.4f}</b>',
                            showarrow=False,
                            font=dict(size=14, color='black'),
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='black',
                            borderwidth=1
                    )
            ]
    )

    # Save
    # write_html saves an interactive file. If you need static images for a paper,
    # use
    fig.write_image(filename.replace('.html', '.png'), scale=3)
    #fig.write_html(filename)
    print(f"Saved plot to {filename}")

    # To display in notebook:
    # fig.show()