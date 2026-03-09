import os
import logging
import numpy as np
import pandas as pd
from CMC_utils import save_load

log = logging.getLogger(__name__)

__all__ = ["target_normalization", "target_denormalization", "set_target_preprocessing", "apply_target_preprocessing"]


def target_normalization(label, target_min, target_max):
    # return labels.sub(target_min).div(target_max - target_min)
    return (label - target_min) / (target_max - target_min)


def target_denormalization(label, target_min, target_max):
    # return labels.prod(target_max - target_min).add(target_min)
    return label * (target_max - target_min) + target_min


def set_target_preprocessing(labels, test_fold: int, val_fold: int, preprocessing_paths: dict, dataset_name: str = None, **_) -> None:
    target_range = pd.Series({"min": labels.min().tolist(), "max": labels.max().tolist()})

    filename_wo_extension = f"{test_fold}_{val_fold}" + (f"_{dataset_name}" if dataset_name else "")

    save_load.save_table( target_range.reset_index().rename({"index": "params", 0: "values"}, axis=1), filename_wo_extension, preprocessing_paths["target"], extension="csv")
    log.info("Training target range saved")


def apply_target_preprocessing(labels, test_fold: int, val_fold: int, preprocessing_paths: dict, dataset_name: str = None) -> pd.DataFrame:
    filename_wo_extension = f"{test_fold}_{val_fold}" + (f"_{dataset_name}" if dataset_name else "")
    params_path = os.path.join(preprocessing_paths["target"], filename_wo_extension + ".csv")
    if os.path.exists(params_path):
        labels = labels.copy()
        params = save_load.load_params_table(params_path, index_col=0).squeeze().to_dict()
        labels = target_normalization(labels, np.array(params["min"]), np.array(params["max"]))
        log.info("Target normalization applied")
    return labels


if __name__ == "__main__":
    pass
