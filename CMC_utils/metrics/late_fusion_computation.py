import numpy as np
import pandas as pd
from typing import List
from CMC_utils import save_load

__all__ = ["get_decision_profile", "apply_late_fusion_approach"]

def get_decision_profile(files: List[str]) -> pd.DataFrame:
    decision_profile = pd.DataFrame()
    for file in files:
        fold_preds = save_load.load_params_table(file, index_col=0)  # .set_index("ID")
        decision_profile = pd.concat([decision_profile, fold_preds], axis=0)
    decision_profile = decision_profile.sort_index()
    return decision_profile


def apply_late_fusion_approach(decision_profile: pd.DataFrame, late_fusion_approach: str, task: str, classes: List[str] = None) -> pd.DataFrame:
    decision_profile = decision_profile.copy()

    fold_labels = decision_profile[["label"]].reset_index().drop_duplicates(subset="ID").set_index("ID")

    decision_profile = pd.DataFrame(decision_profile.probability.tolist(), index=decision_profile.index)
    thresholds = pd.Series([0.5]*fold_labels.shape[0], index=fold_labels.index)

    if late_fusion_approach == "mean":
        fold_probabilities = decision_profile.groupby(decision_profile.index).mean()
    elif late_fusion_approach == "max":
        fold_probabilities = decision_profile.groupby(decision_profile.index).max()
    elif late_fusion_approach == "min":
        fold_probabilities = decision_profile.groupby(decision_profile.index).min()
    elif late_fusion_approach == "median":
        fold_probabilities = decision_profile.groupby(decision_profile.index).median()
    elif late_fusion_approach == "majority_voting":
        fold_probabilities = decision_profile.groupby(decision_profile.index).sum()
        factors = decision_profile.groupby(decision_profile.index).apply(lambda x: x.notna().all(axis=1).sum(axis=0))
        thresholds = thresholds.mul(factors)
    elif late_fusion_approach == "rank_sum":
        # 1. Rank scores within each modality/column (higher score → higher rank)
        ranked = decision_profile.rank(method="average")
        # 2. Sum ranks across modalities for each patient
        fold_probabilities = ranked.groupby(ranked.index).sum()
    elif late_fusion_approach == "product":
        fold_probabilities = decision_profile.groupby(decision_profile.index).prod()
        factors = decision_profile.groupby(decision_profile.index).apply(lambda x: x.notna().all(axis=1).sum(axis=0))
        thresholds = thresholds.pow(factors)
    else:
        raise ValueError(f"Late fusion approach {late_fusion_approach} not recognized")

    if task in ("classification",):
        fold_probabilities = fold_probabilities.rpow(np.e, axis=1)
        fold_probabilities = fold_probabilities.div(fold_probabilities.sum(axis=1), axis=0)

        fold_predictions = fold_probabilities.idxmax(axis=1).rename("prediction")
        if classes is not None:
            fold_predictions = fold_predictions.map({i: classes[i] for i in range(len(classes))})
    elif task in ("multilabelclassification",):
        fold_predictions = fold_probabilities.gt(thresholds, axis=0)
        fold_predictions = fold_predictions.apply(lambda row: row.tolist(), axis=1).rename("prediction")
    else:
        fold_predictions = fold_probabilities[0].rename("prediction")
    fold_probabilities = fold_probabilities.apply(lambda row: row.tolist(), axis=1).rename("probability")
    fold_results = pd.concat([fold_labels, fold_predictions, fold_probabilities], axis=1)
    return fold_results


if __name__ == "__main__":
    pass
