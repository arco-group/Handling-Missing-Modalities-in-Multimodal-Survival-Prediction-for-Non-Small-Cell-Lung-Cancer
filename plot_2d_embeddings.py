
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind

# ==============================
# CONFIGURATION
# ==============================
# Adjust paths as needed
DATASETS = {
        "CT (Whole)"     : {
                "emb_path" : "/Users/filruff/Desktop/PHD/PROGETTI/Multimodal_MARIA_AIDA/data/tabular/survival/AIDA/imaging/embeddings",
                "meta_path": "data/tabular/survival/AIDA/imaging/CT_embeddings.xlsx"
        },
        "CT (Lungs Only)": {
                "emb_path" : "/Users/filruff/Desktop/PHD/PROGETTI/Multimodal_MARIA_AIDA/data/tabular/survival/AIDA/imaging/embeddings_2",
                "meta_path": "data/tabular/survival/AIDA/imaging/CT_embeddings_2.xlsx"
        }
}
SEED = 42


def load_and_align_data(emb_path, meta_path):
    """Loads embeddings and metadata, ensuring they match."""
    # 1. Find Files
    npy_files = sorted(list(Path(emb_path).glob("*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files in {emb_path}")

    # 2. Load Embeddings
    # Assuming filename format 'patient_ID.npy' or similar where ID is the last number
    ids_in_files = [int(f.stem.split('_')[-1]) for f in npy_files]
    embeddings_list = [np.load(f)[np.newaxis, :] for f in npy_files]
    X = np.concatenate(embeddings_list, axis=0).astype(np.float32)

    # Z-score Normalization
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # 3. Load Metadata
    df = pd.read_excel(meta_path)
    df['ID'] = df['ID paziente'].astype(int)

    # 4. Align Data
    # Filter metadata to keep only IDs present in embeddings
    df = df[df['ID'].isin(ids_in_files)].copy()

    # Sort embeddings to match the dataframe order
    id_map = {id_: i for i, id_ in enumerate(ids_in_files)}
    indices = [id_map[uid] for uid in df['ID'].values]
    X_aligned = X[indices]

    # 5. Get Labels
    # 'uncensored' = Event (Death) = Non-Survivor (1)
    # 'censored' = No Event (Alive) = Survivor (0)
    df['Status'] = df['OS'].map({'uncensored': 'Non-Survivor', 'censored': 'Survivor'})
    y_numeric = df['OS'].map({'uncensored': 1, 'censored': 0}).values

    return X_aligned, df, y_numeric


def plot_density_and_centroids(ax, X_2d, labels, title, p_val):
    """Plots Supervised UMAP with KDE density regions and centroids."""

    # Define Style
    palette = {'Survivor': '#2ecc71', 'Non-Survivor': '#e74c3c'}  # Green / Red
    df_plot = pd.DataFrame(X_2d, columns=['UMAP1', 'UMAP2'])
    df_plot['Status'] = labels

    # 1. KDE Density (Background)
    for status, color in palette.items():
        subset = df_plot[df_plot['Status'] == status]
        if len(subset) > 1:
            sns.kdeplot(
                    data=subset, x='UMAP1', y='UMAP2',
                    fill=True, alpha=0.15, color=color,
                    levels=4, thresh=0.05, ax=ax
            )

    # 2. Scatter Points
    sns.scatterplot(
            data=df_plot, x='UMAP1', y='UMAP2', hue='Status',
            palette=palette, alpha=0.9, s=50, edgecolor='white', linewidth=0.5, ax=ax
    )

    # 3. Centroids (X Markers)
    for status, color in palette.items():
        subset = df_plot[df_plot['Status'] == status]
        centroid = subset[['UMAP1', 'UMAP2']].median()
        ax.scatter(
                centroid['UMAP1'], centroid['UMAP2'],
                color=color, s=250, marker='X', edgecolors='black', linewidth=1.5,
                label=f'{status} Centroid', zorder=10
        )

    # 4. Annotation (P-Value)
    # Displaying the statistical significance of the ORIGINAL features
    ax.text(
            0.05, 0.95, f'Group separation: $p = {f"{p_val:.4f}" + "*" if p_val < 0.05 else f"{p_val:.4f}"}$',
            transform=ax.transAxes, fontsize=20, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    )
    ax.set_xlabel('UMAP Dimension 1', fontsize=16)
    ax.set_ylabel('UMAP Dimension 2', fontsize=16)
    ax.legend(loc='lower right', frameon=True, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)


# ==============================
# MAIN EXECUTION
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for i, (name, paths) in enumerate(DATASETS.items()):
    print(f"Processing {name}...")

    try:
        # Load Data
        X, df, y = load_and_align_data(paths['emb_path'], paths['meta_path'])

        # A. Statistical Analysis (On Original Features)
        # We perform T-test on the 1st Principal Component to get a robust p-value
        pca = PCA(n_components=1, random_state=SEED)
        X_pca = pca.fit_transform(X)
        surv_vals = X_pca[y == 0]
        nonsurv_vals = X_pca[y == 1]
        t_stat, p_val = ttest_ind(surv_vals, nonsurv_vals, equal_var=False)
        p_val = p_val[0]

        print(f"  - Original Feature P-value (PC1): {p_val:.4e}")

        # B. Supervised UMAP (For Visualization)
        # passing 'y' makes it supervised -> clusters will be tighter
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.5, n_components=2, random_state=SEED)

        X_umap = reducer.fit_transform(X, y=y)



        # C. Plotting
        plot_density_and_centroids(axes[i], X_umap, df['Status'], name, p_val)

    except Exception as e:
        print(f"  ! Error: {e}")
        axes[i].text(0.5, 0.5, f"Data Error:\n{name}", ha='center')

#plt.suptitle("Supervised UMAP Projection of CT Embeddings (Survivor vs Non-Survivor)", fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig("Supervised_UMAP_Comparison.png", dpi=300, bbox_inches='tight')
plt.show()
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
import umap.umap_ as umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIG
# ============================================================
SEED = 42

DATASETS = {
    "CT (Whole)": {
        "emb_path": "data/tabular/survival/AIDA/imaging/embeddings",
        "meta_path": "data/tabular/survival/AIDA/imaging/CT_embeddings.xlsx"
    },
    "CT (Lungs Only)": {
        "emb_path": "data/tabular/survival/AIDA/imaging/embeddings_2",
        "meta_path": "data/tabular/survival/AIDA/imaging/CT_embeddings_2.xlsx"
    }
}

WSI_EMB_PATH  = "data/tabular/survival/AIDA/wsi/embeddings"
WSI_META_PATH = "data/tabular/survival/AIDA/wsi/WSI_embeddings.xlsx"

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


def compute_pvalue(X, y):
    """T-test on PC1."""
    pca = PCA(n_components=1, random_state=SEED).fit(X)
    pc1 = pca.transform(X).squeeze()
    t, p = ttest_ind(pc1[y == 0], pc1[y == 1], equal_var=False)
    return p


def make_umap(X, y):
    reducer = umap.UMAP(
        n_neighbors=5, min_dist=0.5, n_components=2, random_state=SEED
    )
    return reducer.fit_transform(X ,y=y)

def add_plotly_umap(fig, col, X2d, df, p_value, position):
    surv = df[df["Status"] == "Survivor"]
    nons = df[df["Status"] == "Non-Survivor"]

    X_surv = X2d[df["Status"] == "Survivor"]
    X_nons = X2d[df["Status"] == "Non-Survivor"]

    # Create unique legend items ONCE (for the whole figure)
    legend_map = {
        "Survivor": dict(color="green", name="Survivor"),
        "Non-Survivor": dict(color="red", name="Non-Survivor"),
    }

    # ------------- Density -------------
    fig.add_trace(go.Histogram2dContour(
        x=X_surv[:,0], y=X_surv[:,1],
        ncontours=8, opacity=0.25, colorscale="Greens",
        showscale=False, hoverinfo="skip",
        showlegend=False,   # remove per-panel legend
    ), row=1, col=col)

    fig.add_trace(go.Histogram2dContour(
        x=X_nons[:,0], y=X_nons[:,1],
        ncontours=8, opacity=0.25, colorscale="Reds",
        showscale=False, hoverinfo="skip",
        showlegend=False,
    ), row=1, col=col)

    # ------------- Points -------------
    fig.add_trace(go.Scatter(
        x=X_surv[:,0], y=X_surv[:,1],
        mode="markers",
        marker=dict(color="green", size=7, line=dict(width=0.5, color="white")),
        name="Survivor",
        showlegend=(col == 1),  # show only in first subplot
    ), row=1, col=col)

    fig.add_trace(go.Scatter(
        x=X_nons[:,0], y=X_nons[:,1],
        mode="markers",
        marker=dict(color="red", size=7, line=dict(width=0.5, color="white")),
        name="Non-Survivor",
        showlegend=(col == 1),
    ), row=1, col=col)

    # ------------- Centroids -------------
    cent_surv = X_surv.mean(axis=0)
    cent_nons = X_nons.mean(axis=0)

    fig.add_trace(go.Scatter(
        x=[cent_surv[0]], y=[cent_surv[1]],
        mode="markers",
        marker=dict(symbol="x", size=24, color="darkgreen",
                    line=dict(width=3, color="white")),
        name="Centroid Survivor",
        showlegend=False,  # avoid clutter
    ), row=1, col=col)

    fig.add_trace(go.Scatter(
        x=[cent_nons[0]], y=[cent_nons[1]],
        mode="markers",
        marker=dict(symbol="x", size=24, color="darkred",
                    line=dict(width=3, color="white")),
        name="Centroid Non-Survivor",
        showlegend=False,
    ), row=1, col=col)

    # ------------- Axis labels -------------
    fig.update_xaxes(title_text="UMAP-1", row=1, col=col)
    fig.update_yaxes(title_text="UMAP-2", row=1, col=col)

    # ------------- Annotation with title + p-value -------------
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

# ============================================================
# FIGURE (3 COLUMNS)
# ============================================================

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("CT (Whole)", "CT (Lungs Only)", "WSI")

)
# --- increase subplot title font size ---
fig.update_layout(
    font=dict(size=28),  # global font (optional)
    title_font=dict(size=28),
)
# 🔥 This specifically affects subplot titles

# Increase and bold subplot titles
for ann in fig['layout']['annotations']:
    ann['font'] = dict(
        size=26,          # increase font size
        family="Arial",   # or any font
        color="black",
        # 🔥 bold
        weight="bold"
    )

# ------------------------------------------------------------
# CT DATASETS (column 1 & 2)
# ------------------------------------------------------------
positions_ct = [(-3, -1), (-2, 6)]  # x,y positions for annotations
for col, (name, paths) in enumerate(DATASETS.items(), start=1):
    X, df, y = load_embeddings_and_metadata(paths["emb_path"], paths["meta_path"])
    p_value = compute_pvalue(X, y)

    X_umap = make_umap(X, y)

    add_plotly_umap(fig, col, X_umap, df, p_value, position=positions_ct[col-1])

# ------------------------------------------------------------
# WSI (column 3)
# ------------------------------------------------------------
X_wsi, df_wsi, y_wsi = load_embeddings_and_metadata(WSI_EMB_PATH, WSI_META_PATH)
p_wsi = compute_pvalue(X_wsi, y_wsi)
X_wsi_umap = make_umap(X_wsi, y_wsi)

add_plotly_umap(fig, 3, X_wsi_umap, df_wsi, p_wsi, position=(-15, -12))


# ============================================================
# FINAL LAYOUT
# ============================================================
fig.update_layout(
    height=950,
    width=1900,
    template="plotly_white",

    # 🔥 Improved global font (affects all text)
    font=dict(
        family="Arial",   # or "Helvetica", "Times New Roman"
        size=28,          # increase overall font size for clarity
        color="#333333"   # dark grey for professional look
    ),

    # 🔥 Improved subplot titles
    title_font=dict(
        family="Arial",
        size=28,
        color="#000000"
    ),

    # 🔥 Unified legend with better font
    legend=dict(
        orientation="h",
        y=-0.15,
        x=0.5,
        xanchor="center",
        font=dict(
            family="Arial",
            size=28,
            color="#000000"
        )
    ),

    margin=dict(l=50, r=50, t=80, b=120)
)

# Optional: Improve axis fonts as well
fig.update_xaxes(title_font=dict(size=28, family="Arial"))
fig.update_yaxes(title_font=dict(size=28, family="Arial"))
fig.show()
fig.write_image("umap_ct_ct2_wsi.png", scale=3)
