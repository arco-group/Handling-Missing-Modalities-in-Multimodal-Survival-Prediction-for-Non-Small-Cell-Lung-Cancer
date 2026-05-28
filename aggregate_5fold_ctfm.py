"""
Aggregate 5-fold test predictions from fold_specific_ctfm experiments.

For each experiment under outputs/fold_specific_ctfm/{modality_combo}/{experiment}/,
navigates to:
  provided_provided/predictions/{pred_subdir}/0/0-0/
and concatenates {0,1,2,3,4}_0_test.csv into a single file:
  provided_provided/predictions/{pred_subdir}/aggregate_5fold_test.csv

Duplicate patient IDs (column 'ID') are resolved by keeping the first occurrence.
The time component of the 'label' column (format "[event, time]") is replaced with
the true days2OS value from all_data.xlsx, matched by patient ID.
"""

import sys
import ast
import pandas as pd
from pathlib import Path


BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2023-6-336/AIDA_multimodal_F&C/outputs/fold_specific_ctfm")
ALL_DATA_PATH = Path("/mimer/NOBACKUP/groups/naiss2023-6-336/AIDA_multimodal_F&C/data/tabular/survival/AIDA/cross_validation/all_data.xlsx")
OUTPUT_FILENAME = "aggregate_5fold_test.csv"
FOLD_PATTERN = "{fold}_0_test.csv"
N_FOLDS = 5


def find_prediction_subdir(predictions_dir: Path) -> Path | None:
    """Return the single subdirectory inside predictions_dir, or None if not found."""
    subdirs = [d for d in predictions_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 0:
        print(f"  [WARN] No prediction subdir found in {predictions_dir}")
        return None
    if len(subdirs) > 1:
        print(f"  [WARN] Multiple prediction subdirs in {predictions_dir}: {[d.name for d in subdirs]}, using first")
    return subdirs[0]


def find_fold_dir(pred_subdir: Path) -> Path | None:
    """
    Locate the directory containing the per-fold test CSVs inside pred_subdir.

    Three layouts are supported:
      1. pred_subdir/0/0-0/       (bimodal: two modalities provided)
      2. pred_subdir/0/0-0-0/     (trimodal: three modalities provided)
      3. pred_subdir/              (flat: test CSVs directly inside pred_subdir)
    """
    # Case 3: test CSVs live directly in pred_subdir
    if any(pred_subdir.glob("*_test.csv")):
        return pred_subdir

    # Cases 1 & 2: navigate into 0/ then find the all-zeros subdir
    zero_missing_dir = pred_subdir / "0"
    if not zero_missing_dir.exists():
        return None

    # Find a subdir whose name consists only of "0" and "-" (e.g. "0-0", "0-0-0")
    all_zero_subdirs = [
        d for d in zero_missing_dir.iterdir()
        if d.is_dir() and all(c in "0-" for c in d.name) and d.name.startswith("0")
    ]
    if not all_zero_subdirs:
        return None
    if len(all_zero_subdirs) > 1:
        # Prefer the one with the most zeros (shouldn't happen in practice)
        all_zero_subdirs = sorted(all_zero_subdirs, key=lambda d: d.name)
    return all_zero_subdirs[0]


def fix_label_time(combined: pd.DataFrame, days2os_lookup: dict) -> pd.DataFrame:
    """Replace the time component in label "[event, time]" with days2OS from all_data.xlsx."""
    def _fix_row(row):
        patient_id = row["ID"]
        if patient_id not in days2os_lookup:
            return row["label"]
        days2os = days2os_lookup[patient_id]
        try:
            parsed = ast.literal_eval(str(row["label"]))
            event = parsed[0]
            return f"[{event}, {days2os}]"
        except Exception:
            return row["label"]

    combined["label"] = combined.apply(_fix_row, axis=1)
    return combined


def aggregate_experiment(exp_dir: Path, days2os_lookup: dict) -> None:
    # Discover the single subdirectory inside exp_dir (e.g. "provided_provided")
    exp_subdirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    if not exp_subdirs:
        print(f"  [SKIP] Empty experiment dir: {exp_dir}")
        return
    if len(exp_subdirs) > 1:
        raise RuntimeError(
            f"Multiple subdirs found in {exp_dir}: {[d.name for d in exp_subdirs]}. "
            "Expected exactly one."
        )
    provided_dir = exp_subdirs[0]

    predictions_dir = provided_dir / "predictions"
    if not predictions_dir.exists():
        print(f"  [SKIP] No predictions dir: {predictions_dir}")
        return

    pred_subdir = find_prediction_subdir(predictions_dir)
    if pred_subdir is None:
        return

    fold_dir = find_fold_dir(pred_subdir)
    if fold_dir is None:
        print(f"  [SKIP] Could not locate fold test files under: {pred_subdir}")
        return

    fold_files = []
    for fold in range(N_FOLDS):
        fpath = fold_dir / FOLD_PATTERN.format(fold=fold)
        if fpath.exists():
            fold_files.append(fpath)
        else:
            print(f"  [WARN] Missing fold file: {fpath}")

    if not fold_files:
        print(f"  [SKIP] No fold files found in {fold_dir}")
        return

    dfs = []
    for fpath in fold_files:
        try:
            df = pd.read_csv(fpath)
            dfs.append(df)
        except Exception as e:
            print(f"  [ERROR] Could not read {fpath}: {e}")

    if not dfs:
        return

    combined = pd.concat(dfs, ignore_index=True)
    n_before = len(combined)

    # Keep first occurrence of each patient ID
    duplicated_ids = combined[combined.duplicated(subset="ID", keep=False)]["ID"].unique()
    if len(duplicated_ids) > 0:
        print(f"  [INFO] Duplicate IDs found ({len(duplicated_ids)} unique IDs repeated), keeping first occurrence")
    combined = combined.drop_duplicates(subset="ID", keep="first")
    n_after = len(combined)

    if n_before != n_after:
        print(f"  [INFO] Removed {n_before - n_after} duplicate rows ({n_before} -> {n_after})")

    # Replace time in label with true days2OS from all_data.xlsx
    n_matched = combined["ID"].isin(days2os_lookup).sum()
    if n_matched < n_after:
        print(f"  [WARN] {n_after - n_matched} patients not found in all_data.xlsx, label time unchanged")
    combined = fix_label_time(combined, days2os_lookup)

    out_path = pred_subdir / OUTPUT_FILENAME
    combined.to_csv(out_path, index=False)
    print(f"  [OK] {n_after} patients -> {out_path.relative_to(BASE_DIR.parent)}")


def load_days2os_lookup() -> dict:
    """Load patient ID -> days2OS mapping from all_data.xlsx."""
    df = pd.read_excel(ALL_DATA_PATH, usecols=["ID paziente", "days2OS"])
    df = df.dropna(subset=["ID paziente", "days2OS"])
    df["ID paziente"] = df["ID paziente"].astype(int)
    df["days2OS"] = df["days2OS"].astype(int)
    return dict(zip(df["ID paziente"], df["days2OS"]))


def main():
    if not BASE_DIR.exists():
        print(f"ERROR: Base directory not found: {BASE_DIR}")
        sys.exit(1)

    print(f"Loading days2OS lookup from {ALL_DATA_PATH.name}...")
    days2os_lookup = load_days2os_lookup()
    print(f"  Loaded {len(days2os_lookup)} patients.\n")

    modality_dirs = sorted([
        d for d in BASE_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".") and "late" not in d.name.lower()
    ])

    total_experiments = 0
    for mod_dir in modality_dirs:
        print(f"\n=== {mod_dir.name} ===")
        exp_dirs = sorted([d for d in mod_dir.iterdir() if d.is_dir() and "late" not in d.name.lower()])
        for exp_dir in exp_dirs:
            print(f"  Processing: {exp_dir.name}")
            aggregate_experiment(exp_dir, days2os_lookup)
            total_experiments += 1

    print(f"\nDone. Processed {total_experiments} experiments across {len(modality_dirs)} modality combinations.")


if __name__ == "__main__":
    main()
