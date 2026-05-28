"""
Generate a LaTeX table of C-index results for all Concat*+* architectures
from outputs/fold_specific_ctfm/.

Rows   : (ConcatLayer, Layer2, Freeze MS) — 4 architecture combos × 2 freeze states = 8 rows
Columns: modality combinations (CT+Tab, WSI+CT, WSI+CT+Tab, WSI+Tab)
Cells  : mean ± SE over 5 folds
"""

import os
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = "outputs/fold_specific_ctfm"
TARGET     = "OS_1d_regression"
METRIC     = "c_index"
N_FOLDS    = 5
OUTPUT_TEX = "tests/table_ctfm_architectures.tex"
OUTPUT_CSV = "tests/table_ctfm_architectures.csv"

os.makedirs("tests", exist_ok=True)

# Modality directory -> display name (column headers)
MODALITY_DIRS = {
    "ct_tab_cox-label_cix"     : "CT+Tab",
    "wsi_ct_cox-label_cix"     : "WSI+CT",
    "wsi_ct_tab_cox-label_cix" : "WSI+CT+Tab",
    "wsi_tab_cox-label_cix"    : "WSI+Tab",
}

# Row order: (concat_layer, layer2)
ARCH_ORDER = [
    ("ConcatFC",   "FC"),
    ("ConcatFC",   "ODST"),
    ("ConcatODST", "FC"),
    ("ConcatODST", "ODST"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_results_subdir(exp_dir):
    """Return the single subdirectory name inside provided_provided/results/."""
    results_root = os.path.join(exp_dir, "provided_provided", "results")
    if not os.path.isdir(results_root):
        return None
    names = [n for n in os.listdir(results_root)
             if os.path.isdir(os.path.join(results_root, n))]
    return names[0] if names else None


def find_zero_missing_subfolder(exp_dir, sub):
    """
    Find the all-zeros missing subfolder (e.g. '0-0' or '0-0-0') dynamically.
    Matches any folder name that is all zeros separated by dashes.
    """
    fold0_dir = os.path.join(exp_dir, "provided_provided", "results", sub, "0")
    if not os.path.isdir(fold0_dir):
        return None
    for name in os.listdir(fold0_dir):
        if re.match(r"^0(-0)+$", name):
            return name
    return None


def read_cindex_stats(exp_dir):
    """
    Return (mean, se) for c_index at 0% missing, or (None, None).
    Uses averages_classes_average_performance.csv which has multi-level headers.
    Automatically detects '0-0' vs '0-0-0' subfolder depending on modality count.
    """
    sub = get_results_subdir(exp_dir)
    if sub is None:
        return None, None

    zero_folder = find_zero_missing_subfolder(exp_dir, sub)
    if zero_folder is None:
        return None, None

    csv_path = os.path.join(
        exp_dir, "provided_provided", "results", sub,
        "0", zero_folder, "unbalanced", "test",
        "averages_classes_average_performance.csv",
    )
    if not os.path.exists(csv_path):
        # Fall back to per-fold CSV
        csv_path = os.path.join(
            exp_dir, "provided_provided", "results", sub,
            "0", zero_folder, "unbalanced", "test",
            "all_test_performance.csv",
        )
        if not os.path.exists(csv_path):
            return None, None
        df = pd.read_csv(csv_path)
        vals = df[METRIC].dropna().values
        if len(vals) == 0:
            return None, None
        return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    df = pd.read_csv(csv_path, header=[0, 1], index_col=[0, 1])
    # columns are like (c_index, mean), (c_index, std), ...
    try:
        mean = float(df[(METRIC, "mean")].iloc[0])
        std  = float(df[(METRIC, "std")].iloc[0])
        return mean, std / np.sqrt(N_FOLDS)
    except KeyError:
        return None, None


def find_exp_dir(mod_dir_path, concat_layer, layer2, frozen):
    """
    Locate the experiment directory matching Concat{concat_layer}+{layer2}_*
    with or without _ms-frozen suffix.
    """
    pattern = rf"^{re.escape(concat_layer)}\+{re.escape(layer2)}_"
    candidates = []
    for name in os.listdir(mod_dir_path):
        if not re.match(pattern, name):
            continue
        if TARGET not in name:
            continue
        is_frozen = name.endswith("_ms-frozen")
        if frozen and is_frozen:
            candidates.append(name)
        elif not frozen and not is_frozen:
            candidates.append(name)
    if not candidates:
        return None
    return os.path.join(mod_dir_path, sorted(candidates)[-1])


# ---------------------------------------------------------------------------
# Collect results
# ---------------------------------------------------------------------------
# results[(concat_layer, layer2, frozen)][modality_label] = (mean, se)
results = {}

for arch in ARCH_ORDER:
    concat_layer, layer2 = arch
    for frozen in [False, True]:
        key = (concat_layer, layer2, frozen)
        results[key] = {}

        for mod_dir, mod_label in MODALITY_DIRS.items():
            mod_dir_path = os.path.join(BASE_DIR, mod_dir)
            if not os.path.isdir(mod_dir_path):
                print(f"[SKIP] directory not found: {mod_dir_path}")
                continue

            exp_dir = find_exp_dir(mod_dir_path, concat_layer, layer2, frozen)
            if exp_dir is None:
                print(f"[SKIP] no match for {concat_layer}+{layer2} "
                      f"frozen={frozen} in {mod_dir}")
                continue

            mean, se = read_cindex_stats(exp_dir)
            if mean is None:
                print(f"[SKIP] no stats for {os.path.basename(exp_dir)}")
                continue

            results[key][mod_label] = (mean, se)
            print(f"[OK] {concat_layer}+{layer2} frozen={frozen:5} "
                  f"{mod_label:15} C-index={mean:.2f} ± {se:.2f}")

# ---------------------------------------------------------------------------
# Build flat DataFrame for CSV export
# ---------------------------------------------------------------------------
mod_cols = list(MODALITY_DIRS.values())
rows_csv = []
for arch in ARCH_ORDER:
    concat_layer, layer2 = arch
    for frozen in [False, True]:
        key = (concat_layer, layer2, frozen)
        row = {
            "Layer Concat": concat_layer,
            "Layer 2"     : layer2,
            "Freeze MS"   : "Yes" if frozen else "No",
        }
        for mod_label in mod_cols:
            if mod_label in results[key]:
                m, s = results[key][mod_label]
                row[f"{mod_label} mean"] = round(m, 2)
                row[f"{mod_label} SE"]   = round(s, 2)
            else:
                row[f"{mod_label} mean"] = None
                row[f"{mod_label} SE"]   = None
        rows_csv.append(row)

df_csv = pd.DataFrame(rows_csv)
df_csv.to_csv(OUTPUT_CSV, index=False)
print(f"\nCSV saved -> {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# Build LaTeX table
# ---------------------------------------------------------------------------

def fmt_cell(key, mod_label):
    if mod_label not in results[key]:
        return "--"
    m, s = results[key][mod_label]
    return rf"{m:.2f} {{\small $\pm$ {s:.2f}}}"


# Column spec: 3 descriptor cols + len(mod_cols) data cols
col_spec = "lll" + "c" * len(mod_cols)

lines = []
lines.append(r"\begin{table}[ht]")
lines.append(r"  \centering")
lines.append(r"  \caption{C-index (mean $\pm$ SE, 5-fold CV) for all "
             r"Concat architecture variants using CT-FM embeddings. "
             r"Columns correspond to modality combinations.}")
lines.append(r"  \label{tab:ctfm_architectures}")
lines.append(r"  \resizebox{\linewidth}{!}{%")
lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
lines.append(r"    \toprule")

# Header row
header_mods = " & ".join(rf"\textbf{{{m}}}" for m in mod_cols)
lines.append(
    r"    \textbf{Layer Concat} & \textbf{Layer 2} & \textbf{Freeze MS} & "
    + header_mods + r" \\"
)
lines.append(r"    \midrule")

# Data rows — group by architecture (4 groups of 2 rows each)
for i, arch in enumerate(ARCH_ORDER):
    concat_layer, layer2 = arch
    if i > 0:
        lines.append(r"    \midrule")

    for j, frozen in enumerate([False, True]):
        key = (concat_layer, layer2, frozen)
        freeze_str = "Yes" if frozen else "No"

        cells = " & ".join(fmt_cell(key, m) for m in mod_cols)

        if j == 0:
            # First sub-row: show ConcatLayer and Layer2 with \multirow
            lines.append(
                rf"    \multirow{{2}}{{*}}{{\texttt{{{concat_layer}}}}} & "
                rf"\multirow{{2}}{{*}}{{\texttt{{{layer2}}}}} & "
                rf"{freeze_str} & {cells} \\"
            )
        else:
            lines.append(rf"    & & {freeze_str} & {cells} \\")

lines.append(r"    \bottomrule")
lines.append(r"  \end{tabular}}")
lines.append(r"  }")
lines.append(r"\end{table}")

latex_str = "\n".join(lines)

with open(OUTPUT_TEX, "w") as f:
    f.write(latex_str + "\n")

print(f"LaTeX table saved -> {OUTPUT_TEX}")
print()
print(latex_str)
