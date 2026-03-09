import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Data Preparation (Unchanged)
# ---------------------------------------------------------
TOTAL_N = 179

# --- Data for Panel A (Modalities) ---
df_modality = pd.DataFrame({
    'Modality': ['CT', 'Tabular', 'WSI'],
    'Available_Pct': [97.2, 100.0, 33.5]
})
df_modality['Missing_Pct'] = 100 - df_modality['Available_Pct']
df_modality['Available_N'] = (df_modality['Available_Pct'] / 100 * TOTAL_N).round().astype(int)
df_modality['Missing_N'] = TOTAL_N - df_modality['Available_N']
df_modality = df_modality.sort_values('Available_Pct', ascending=False)

# --- Data for Panel B (Clinical Variables) ---
data_clinical_specific = [
    ("MET Status", 3.4), ("EGFR Status", 26.3), ("ALK Status", 26.8),
    ("Comorbidity 2", 28.5), ("Induction CT", 30.0), ("PD-L1", 40.0),
    ("Comorbidity 1", 41.9), ("Smoking Status", 44.7), ("Esophageal Tox.", 46.9),
    ("ECOG PS", 47.5), ("Weight", 48.0), ("Height", 48.0),
    ("Concomitant CT", 72.0), ("RT Technique", 79.3), ("Histology", 99.0)
]
data_clinical_aggregated = [
    ("Others", 100.0)
]

df_clinical = pd.DataFrame(data_clinical_specific + data_clinical_aggregated, columns=['Variable', 'Available_Pct'])
df_clinical['Missing_Pct'] = 100 - df_clinical['Available_Pct']
df_clinical = df_clinical.sort_values('Available_Pct', ascending=False)

# ---------------------------------------------------------
# 2. Plot Construction (Split)
# ---------------------------------------------------------

# Define Colors & Fonts
COLOR_AVAIL = '#A2E2AE'
COLOR_MISS  = '#FFB7B2'
FONT_COLOR = "#2c3e50"
FONT_FAMILY = "Computer Modern"
FONT_SIZE = 24  # Increased from 14

# ==========================================
# FIGURE 1: Data Availability by Modality
# ==========================================
fig1 = go.Figure()

# Available Traces
fig1.add_trace(go.Bar(
    x=df_modality['Modality'],
    y=df_modality['Available_Pct'],
    name='Available',
    marker=dict(color=COLOR_AVAIL, line=dict(color='grey', width=0.5)),
    text=[f"{n}<br>({p:.1f}%)" for n, p in zip(df_modality['Available_N'], df_modality['Available_Pct'])],
    textposition='auto',
    showlegend=True
))

# Missing Traces
fig1.add_trace(go.Bar(
    x=df_modality['Modality'],
    y=df_modality['Missing_Pct'],
    name='Missing',
    marker=dict(color=COLOR_MISS, line=dict(color='grey', width=0.5)),
    text=[f"{n}<br>({p:.1f}%)" if p > 5 else "" for n, p in zip(df_modality['Missing_N'], df_modality['Missing_Pct'])],
    textposition='auto',
    showlegend=True
))

# Layout for Figure 1
fig1.update_layout(
    #title="<b>a) Data Availability by Modality</b>",
    barmode='stack',
    plot_bgcolor='white',
    font=dict(family=FONT_FAMILY, size=FONT_SIZE, color=FONT_COLOR, weight="bold"),
    height=500, # Adjusted height for single plot
    width=800,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=80, r=50, t=80, b=50)
)
fig1.update_yaxes(title_text="% Patients", range=[0, 100], showgrid=True, gridcolor='#ececec')


# ==========================================
# FIGURE 2: Missingness in Structured Clinical Variables
# ==========================================
fig2 = go.Figure()

# Available Traces
fig2.add_trace(go.Bar(
    x=df_clinical['Variable'],
    y=df_clinical['Available_Pct'],
    name='Available',
    marker=dict(color=COLOR_AVAIL, line=dict(color='grey', width=0.5)),
    text=[f"{p:.1f}%" if p > 10 else "" for p in df_clinical['Available_Pct']],
    textposition='inside',
    showlegend=True # Enabled legend for standalone figure
))

# Missing Traces
fig2.add_trace(go.Bar(
    x=df_clinical['Variable'],
    y=df_clinical['Missing_Pct'],
    name='Missing',
    marker=dict(color=COLOR_MISS, line=dict(color='grey', width=0.5)),
    text=[f"{p:.1f}%" if p > 10 else "" for p in df_clinical['Missing_Pct']],
    textposition='inside',
    showlegend=True # Enabled legend for standalone figure
))

# Layout for Figure 2
fig2.update_layout(
    #title="<b>b) Missingness in Structured Clinical Variables</b>",
    barmode='stack',
    plot_bgcolor='white',
    font=dict(family=FONT_FAMILY, size=FONT_SIZE, color=FONT_COLOR, weight="bold"),
    height=600, # Adjusted height
    width=1000,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=80, r=50, t=80, b=100)
)
fig2.update_yaxes(title_text="% Patients", range=[0, 100], showgrid=True, gridcolor='#ececec')
fig2.update_xaxes(tickangle=-45)

# Show or Save
#fig1.show()
#fig2.show()

fig1.write_image("figure1_modality.png", scale=3)
fig2.write_image("figure2_clinical.png", scale=3)