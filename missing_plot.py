import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# 2. Plot Construction (Separate figures)
# ---------------------------------------------------------

# Define Colors & Fonts
COLOR_AVAIL = '#B8D8BA'   # pastel sage green
COLOR_MISS  = '#F4B9B8'   # pastel rose
FONT_COLOR  = "#2c3e50"
FONT_FAMILY = "Didot, GFS Didot, Bodoni MT, Palatino, serif"
FONT_SIZE   = 18

COMMON_LAYOUT = dict(
    barmode='stack',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family=FONT_FAMILY, size=FONT_SIZE, color=FONT_COLOR),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        font=dict(size=FONT_SIZE),
    ),
    bargap=0.35,
)

# ==========================================
# FIGURE 1: Data Availability by Modality (vertical bars)
# ==========================================
fig1 = go.Figure()

fig1.add_trace(go.Bar(
    y=df_modality['Modality'],
    x=df_modality['Available_Pct'],
    name='Available',
    orientation='h',
    marker=dict(color=COLOR_AVAIL, line=dict(color='white', width=1.0)),
    text=[f"{n}  ({p:.1f}%)" for n, p in zip(df_modality['Available_N'], df_modality['Available_Pct'])],
    textposition='inside',
    insidetextanchor='middle',
    showlegend=True,
))

fig1.add_trace(go.Bar(
    y=df_modality['Modality'],
    x=df_modality['Missing_Pct'],
    name='Missing',
    orientation='h',
    marker=dict(color=COLOR_MISS, line=dict(color='white', width=1.0)),
    text=[f"{n}  ({p:.1f}%)" if p > 5 else "" for n, p in zip(df_modality['Missing_N'], df_modality['Missing_Pct'])],
    textposition='inside',
    insidetextanchor='middle',
    showlegend=True,
))

fig1.update_layout(
    **COMMON_LAYOUT,
    height=300,
    width=700,
    margin=dict(l=20, r=30, t=70, b=60),
)
fig1.update_xaxes(title_text="Missing Modality (%)", range=[0, 100],
                  showgrid=True, gridcolor='#ececec', zeroline=False)
fig1.update_yaxes(showgrid=False)


# ==========================================
# FIGURE 2: Missingness in Clinical Variables (vertical bars)
# ==========================================
fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=df_clinical['Variable'],
    y=df_clinical['Available_Pct'],
    name='Available',
    marker=dict(color=COLOR_AVAIL, line=dict(color='white', width=0.8)),
    text=[f"<b>{p:.1f}%</b>" if p > 8 else "" for p in df_clinical['Available_Pct']],
    textposition='inside',
    insidetextanchor='middle',
    textangle=0,
    textfont=dict(size=26, color=FONT_COLOR, family=FONT_FAMILY),
    showlegend=True,
))

fig2.add_trace(go.Bar(
    x=df_clinical['Variable'],
    y=df_clinical['Missing_Pct'],
    name='Missing',
    marker=dict(color=COLOR_MISS, line=dict(color='white', width=0.8)),
    text=[f"<b>{p:.1f}%</b>" if p > 8 else "" for p in df_clinical['Missing_Pct']],
    textposition='inside',
    insidetextanchor='middle',
    textangle=0,
    textfont=dict(size=26, color=FONT_COLOR, family=FONT_FAMILY),
    showlegend=True,
))

fig2.update_layout(
    **COMMON_LAYOUT,
    height=700,
    width=1400,
    margin=dict(l=80, r=30, t=100, b=200),
)
fig2.update_layout(
    font=dict(family=FONT_FAMILY, size=28, color=FONT_COLOR),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        font=dict(size=30, family=FONT_FAMILY),
    ),
)
fig2.update_xaxes(
    tickangle=-45,
    tickfont=dict(size=26, family=FONT_FAMILY, color=FONT_COLOR),
    showgrid=False,
)
fig2.update_yaxes(
    title_text="% Patients",
    range=[0, 100],
    showgrid=True,
    gridcolor='#ececec',
    zeroline=False,
    title_font=dict(size=30, family=FONT_FAMILY, color=FONT_COLOR),
    tickfont=dict(size=26, family=FONT_FAMILY, color=FONT_COLOR),
)

# Save
fig1.write_image("figure1_modality.png", scale=3)
fig2.write_image("figure2_clinical.png", scale=3)