import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import ttest_ind
import umap.umap_ as umap
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIG
# ============================================================
SEED = 42

# CT Dataset
CT_DATASET = {
        "emb_path" : "data/tabular/survival/AIDA/imaging/embeddings_ctfm",
        "meta_path": "data/tabular/survival/AIDA/imaging/CT_embeddings_ctfm.xlsx"
}

# WSI Dataset
WSI_EMB_PATH = "data/tabular/survival/AIDA/wsi/embeddings"
WSI_META_PATH = "data/tabular/survival/AIDA/wsi/WSI_embeddings.xlsx"

# Tabular Dataset
TABULAR_DATA_PATH = "data/tabular/survival/AIDA/tabular/clinical_data_III_stage.csv"


# ============================================================
# FUNCTIONS
# ============================================================

def load_embeddings_and_metadata(emb_path, meta_path):
    """Loads, z-score normalizes, aligns embeddings and metadata."""
    npy_files = sorted(Path(emb_path).glob("*.npy"))
    ids_in_files = [int(f.stem.split("_")[-1]) for f in npy_files]

    emb_list = [np.load(f)[np.newaxis, :] for f in npy_files]
    X = np.concatenate(emb_list, axis=0).astype(np.float32)

    # Z-score
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    df = pd.read_excel(meta_path)
    df["ID"] = df["ID paziente"].astype(int)
    df = df[df["ID"].isin(ids_in_files)].copy()

    # Align order
    id_map = {pid: i for i, pid in enumerate(ids_in_files)}
    idx = [id_map[x] for x in df["ID"]]
    X = X[idx]

    # Labels
    df["Status"] = df["OS"].map({"uncensored": "Non-Survivor", "censored": "Survivor"})
    y = df["OS"].map({"uncensored": 1, "censored": 0}).values

    return X, df, y


def preprocess_tabular_data(csv_path):
    """
    Load and preprocess tabular clinical data:
    - Imputation for missing values
    - Standardization for continuous variables
    - Label encoding for categorical variables
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Extract survival status
    df["Status"] = df["OS"].map({"uncensored": "Non-Survivor", "censored": "Survivor"})
    y = df["OS"].map({"uncensored": 1, "censored": 0}).values

    # Select relevant clinical features (exclude IDs, outcomes, and imaging availability)
    exclude_cols = [
            'ID paziente', 'days2recidiva', 'days2metastasi', 'days2OS',
            '6months2recidiva', '6months2metastasi', '6months2OS',
            'recidiva', 'metastasi', 'status', 'PFSevent', 'days2PFS',
            '6months2PFS', 'OS', 'PFS', 'EHR', 'WSI', 'CT', 'Status'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    df_features = df[feature_cols].copy()

    # Identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []

    for col in df_features.columns:
        # Check if column has numeric type or can be converted to numeric
        try:
            pd.to_numeric(df_features[col], errors='raise')
            numerical_cols.append(col)
        except (ValueError, TypeError):
            categorical_cols.append(col)

    # Process categorical columns
    df_encoded = df_features.copy()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        # Handle missing values by treating them as a separate category
        df_encoded[col] = df_encoded[col].astype(str)
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Convert all to numeric
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')

    # Imputation for missing values
    # Use median for numerical features
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(df_encoded)

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, df, y


def compute_pvalue(X, y):
    """T-test on PC1."""
    pca = PCA(n_components=1, random_state=SEED).fit(X)
    pc1 = pca.transform(X).squeeze()
    t, p = ttest_ind(pc1[y == 0], pc1[y == 1], equal_var=False)
    return p


def make_umap(X, y, **kwargs):
    reducer = umap.UMAP(
            n_neighbors=5, min_dist=0.5, n_components=2, random_state=SEED
    )
    return reducer.fit_transform(X, y=y)

def make_tsne(X, n_components=2):
    """Create t-SNE projection."""
    tsne = TSNE(n_components=n_components, random_state=SEED, perplexity=30)
    return tsne.fit_transform(X)


def add_plotly_umap_2d(fig, col, X2d, df, p_value, title, position):
    """Add 2D UMAP visualization to subplot."""
    surv = df[df["Status"] == "Survivor"]
    nons = df[df["Status"] == "Non-Survivor"]

    X_surv = X2d[df["Status"] == "Survivor"]
    X_nons = X2d[df["Status"] == "Non-Survivor"]

    # Density contours
    fig.add_trace(go.Histogram2dContour(
            x=X_surv[:, 0], y=X_surv[:, 1],
            ncontours=8, opacity=0.25, colorscale="Greens",
            showscale=False, hoverinfo="skip",
            showlegend=False,
    ), row=1, col=col)

    fig.add_trace(go.Histogram2dContour(
            x=X_nons[:, 0], y=X_nons[:, 1],
            ncontours=8, opacity=0.25, colorscale="Reds",
            showscale=False, hoverinfo="skip",
            showlegend=False,
    ), row=1, col=col)

    # Scatter points
    fig.add_trace(go.Scatter(
            x=X_surv[:, 0], y=X_surv[:, 1],
            mode="markers",
            marker=dict(color="green", size=7, line=dict(width=0.5, color="white")),
            name="Survivor",
            showlegend=(col == 1),
    ), row=1, col=col)

    fig.add_trace(go.Scatter(
            x=X_nons[:, 0], y=X_nons[:, 1],
            mode="markers",
            marker=dict(color="red", size=7, line=dict(width=0.5, color="white")),
            name="Non-Survivor",
            showlegend=(col == 1),
    ), row=1, col=col)

    # Centroids
    cent_surv = X_surv.mean(axis=0)
    cent_nons = X_nons.mean(axis=0)

    fig.add_trace(go.Scatter(
            x=[cent_surv[0]], y=[cent_surv[1]],
            mode="markers",
            marker=dict(symbol="x", size=24, color="darkgreen",
                        line=dict(width=3, color="white")),
            name="Centroid Survivor",
            showlegend=False,
    ), row=1, col=col)

    fig.add_trace(go.Scatter(
            x=[cent_nons[0]], y=[cent_nons[1]],
            mode="markers",
            marker=dict(symbol="x", size=24, color="darkred",
                        line=dict(width=3, color="white")),
            name="Centroid Non-Survivor",
            showlegend=False,
    ), row=1, col=col)

    # Axis labels
    fig.update_xaxes(title_text=f"{title}-1", row=1, col=col, showticklabels=False, ticks="")
    fig.update_yaxes(title_text=f"{title}-2", row=1, col=col, showticklabels=False, ticks="")

    # Annotation with p-value
    fig.add_annotation(
            x=position[0] if isinstance(position, tuple) else 0.5,
            y=position[1] if isinstance(position, tuple) else 1.15,
            text=(
                    f"<b>Group separation</b><br>"
                    f"p = {p_value:.4f}{'*' if p_value < 0.05 else ''}"
            ),
            showarrow=False,
            font=dict(size=24),
            align="left",
            bordercolor="gray",
            borderwidth=1.5,
            borderpad=6,
            bgcolor="rgba(255,255,255,0.85)",
            row=1, col=col
    )


def add_plotly_3d(fig, row, col, X3d, df, p_value, title):
    """Add 3D visualization to subplot."""
    surv = df[df["Status"] == "Survivor"]
    nons = df[df["Status"] == "Non-Survivor"]

    X_surv = X3d[df["Status"] == "Survivor"]
    X_nons = X3d[df["Status"] == "Non-Survivor"]

    # Scatter points
    fig.add_trace(go.Scatter3d(
            x=X_surv[:, 0], y=X_surv[:, 1], z=X_surv[:, 2],
            mode="markers",
            marker=dict(color="green", size=4, line=dict(width=0.3, color="white")),
            name="Survivor",
            showlegend=(col == 1),
    ), row=row, col=col)

    fig.add_trace(go.Scatter3d(
            x=X_nons[:, 0], y=X_nons[:, 1], z=X_nons[:, 2],
            mode="markers",
            marker=dict(color="red", size=4, line=dict(width=0.3, color="white")),
            name="Non-Survivor",
            showlegend=(col == 1),
    ), row=row, col=col)

    # Centroids
    cent_surv = X_surv.mean(axis=0)
    cent_nons = X_nons.mean(axis=0)

    fig.add_trace(go.Scatter3d(
            x=[cent_surv[0]], y=[cent_surv[1]], z=[cent_surv[2]],
            mode="markers",
            marker=dict(symbol="x", size=12, color="darkgreen",
                        line=dict(width=2, color="white")),
            name="Centroid Survivor",
            showlegend=False,
    ), row=row, col=col)

    fig.add_trace(go.Scatter3d(
            x=[cent_nons[0]], y=[cent_nons[1]], z=[cent_nons[2]],
            mode="markers",
            marker=dict(symbol="x", size=12, color="darkred",
                        line=dict(width=2, color="white")),
            name="Centroid Non-Survivor",
            showlegend=False,
    ), row=row, col=col)


# ============================================================
# MAIN EXECUTION: UMAP 2D (3 panels)
# ============================================================

print("=" * 60)
print("CREATING UMAP 2D VISUALIZATION (CT, Tabular, WSI)")
print("=" * 60)

fig_umap_2d = make_subplots(
        rows=1, cols=3,
        subplot_titles=("WSI", "CT ", "Tabular" )
)

# Style subplot titles
for ann in fig_umap_2d['layout']['annotations']:
    ann['font'] = dict(size=26, family="Didot", color="black", weight="bold")
# --- WSI Dataset ---
print("\n[3/3] Processing WSI Dataset...")
X_wsi, df_wsi, y_wsi = load_embeddings_and_metadata(WSI_EMB_PATH, WSI_META_PATH)
p_wsi = compute_pvalue(X_wsi, y_wsi)
X_wsi_umap = make_umap(X_wsi, y_wsi)
add_plotly_umap_2d(fig_umap_2d, 1, X_wsi_umap, df_wsi, p_wsi, "UMAP", (15, -18))
print(f"   ✓ WSI: p-value = {p_wsi:.4e}")
# --- CT Dataset ---
print("\n[1/3] Processing CT Dataset...")
X_ct, df_ct, y_ct = load_embeddings_and_metadata(CT_DATASET["emb_path"], CT_DATASET["meta_path"])
p_ct = compute_pvalue(X_ct, y_ct)
X_ct_umap = make_umap(X_ct, y_ct)
add_plotly_umap_2d(fig_umap_2d, 2, X_ct_umap, df_ct, p_ct, "UMAP", (16, -18))
print(f"   ✓ CT: p-value = {p_ct:.4e}")

# --- Tabular Dataset ---
print("\n[2/3] Processing Tabular Dataset...")
X_tab, df_tab, y_tab = preprocess_tabular_data(TABULAR_DATA_PATH)
p_tab = compute_pvalue(X_tab, y_tab)
X_tab_umap = make_umap(X_tab, y_tab)
add_plotly_umap_2d(fig_umap_2d, 3, X_tab_umap, df_tab, p_tab, "UMAP", (12, 0))
print(f"   ✓ Tabular: p-value = {p_tab:.4e}")



# Layout for UMAP 2D
fig_umap_2d.update_layout(
        height=600,
        width=1900,
        template="plotly_white",
        font=dict(family="Didot", size=28, color="#333333"),
        title_font=dict(family="Didot", size=28, color="#000000"),
        legend=dict(
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center",
                font=dict(family="Didot", size=28, color="#000000")
        ),
        margin=dict(l=50, r=50, t=80, b=120)
)

fig_umap_2d.update_xaxes(title_font=dict(size=28, family="Didot"), showticklabels=False, ticks="")
fig_umap_2d.update_yaxes(title_font=dict(size=28, family="Didot"), showticklabels=False, ticks="")

print("\n✓ UMAP 2D figure created")
fig_umap_2d.write_image("umap_2d_ct_tabular_wsi.png", scale=3)
print("✓ Saved: umap_2d_ct_tabular_wsi.png")

"""# ============================================================
# t-SNE 2D (3 panels)
# ============================================================

print("\n" + "=" * 60)
print("CREATING t-SNE 2D VISUALIZATION")
print("=" * 60)

fig_tsne_2d = make_subplots(
        rows=1, cols=3,
        subplot_titles=("CT (Whole)", "Tabular Data", "WSI")
)

for ann in fig_tsne_2d['layout']['annotations']:
    ann['font'] = dict(size=26, family="Didot", color="black", weight="bold")

# --- CT Dataset ---
print("\n[1/3] Computing t-SNE for CT...")
X_ct_tsne = make_tsne(X_ct, n_components=2)
add_plotly_umap_2d(fig_tsne_2d, 1, X_ct_tsne, df_ct, p_ct, "t-SNE", (-3, -1))

# --- Tabular Dataset ---
print("[2/3] Computing t-SNE for Tabular...")
X_tab_tsne = make_tsne(X_tab, n_components=2)
add_plotly_umap_2d(fig_tsne_2d, 2, X_tab_tsne, df_tab, p_tab, "t-SNE", (-2, 6))

# --- WSI Dataset ---
print("[3/3] Computing t-SNE for WSI...")
X_wsi_tsne = make_tsne(X_wsi, n_components=2)
add_plotly_umap_2d(fig_tsne_2d, 3, X_wsi_tsne, df_wsi, p_wsi, "t-SNE", (-15, -12))

# Layout for t-SNE 2D
fig_tsne_2d.update_layout(
        height=600,
        width=1900,
        template="plotly_white",
        font=dict(family="Didot", size=28, color="#333333"),
        title_font=dict(family="Didot", size=28, color="#000000"),
        legend=dict(
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center",
                font=dict(family="Didot", size=28, color="#000000")
        ),
        margin=dict(l=50, r=50, t=80, b=120)
)

fig_tsne_2d.update_xaxes(title_font=dict(size=28, family="Didot"))
fig_tsne_2d.update_yaxes(title_font=dict(size=28, family="Didot"))

print("\n✓ t-SNE 2D figure created")
fig_tsne_2d.write_image("tsne_2d_ct_tabular_wsi.png", scale=3)
print("✓ Saved: tsne_2d_ct_tabular_wsi.png")

# ============================================================
# t-SNE 3D (3 panels)
# ============================================================
"""
print("\nGenerated files:")
print("  1. umap_2d_ct_tabular_wsi.png")
print("  2. tsne_2d_ct_tabular_wsi.png")
