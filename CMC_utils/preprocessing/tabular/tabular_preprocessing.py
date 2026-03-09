import os
import logging
import numpy as np
import pandas as pd
from typing import Union, List
from omegaconf import DictConfig

from .functions import *
from CMC_utils import save_load

log = logging.getLogger(__name__)

__all__ = ["set_fold_preprocessing", "apply_preprocessing", "describe_features", "join_tabular_preprocessing_params"]


def set_fold_preprocessing(train, test_fold: int, val_fold: int, preprocessing_paths: Union[dict, DictConfig], preprocessing_params: dict, dataset_name: str = None, **kwargs) -> None:
    """
    Applies preprocessing to a fold and saves the parameters
    Parameters
    ----------
    train : pd.DataFrame
    test_fold : int
    val_fold : int
    preprocessing_paths : Union[dict, DictConfig]
    preprocessing_params : dict
    dataset_name: str
    kwargs : dict

    Returns
    -------
    None
    """
    train = train.copy()
    filename_wo_extension = f"{test_fold}_{val_fold}" + (f"_{dataset_name}" if dataset_name else "")

    if "categorical" in preprocessing_params.keys() and preprocessing_params["categorical_columns"]:
        categorical_path = os.path.join( preprocessing_paths["categorical"], filename_wo_extension + ".csv")
        if not os.path.exists( categorical_path ):

            train, train_categorical_params = categorical_preprocessing(train, unique_values=preprocessing_params["categorical_unique_values"], return_categories=True, **preprocessing_params["categorical"], **kwargs)

            save_load.save_table(pd.Series(train_categorical_params).reset_index().rename( {"index": "params", 0: "values"}, axis=1 ), filename_wo_extension, preprocessing_paths["categorical"], extension="csv")
            log.info("Categorical preprocessing params saved")
        else:
            categorical_params = save_load.load_params_table(categorical_path, index_col=0).squeeze().to_dict()
            train = categorical_preprocessing(train, **categorical_params)

    if "numerical" in preprocessing_params.keys():
        numerical_path = os.path.join(preprocessing_paths["numerical"], filename_wo_extension + ".csv")
        if not os.path.exists( numerical_path ):

            columns_to_preprocess = list(preprocessing_params["numerical_columns"].values())

            train, train_numerical_params = numerical_preprocessing(train, numerical_columns=columns_to_preprocess, return_params=True, **preprocessing_params["numerical"])

            save_load.save_table(pd.Series(train_numerical_params).reset_index().rename( {"index": "params", 0: "values"}, axis=1 ), filename_wo_extension, preprocessing_paths["numerical"], extension="csv")
            log.info("Numerical preprocessing params saved")

        else:
            numerical_params = save_load.load_params_table(numerical_path, index_col=0).squeeze().to_dict()
            train = numerical_preprocessing(train, **numerical_params)

    if "imputer" in preprocessing_params.keys():
        if not os.path.exists(os.path.join(preprocessing_paths["imputer"], filename_wo_extension + ".pkl")):
            train, train_imputer = impute_missing(train, features_info=preprocessing_params, return_imputer=True, **preprocessing_params["imputer"])

            save_load.save_model(train_imputer, filename_wo_extension, preprocessing_paths["imputer"], extension="pkl", model_params=pd.Series(preprocessing_params["imputer"]))
            log.info("Imputation model saved")


def apply_preprocessing(*sets: pd.DataFrame, preprocessing_paths: Union[dict, DictConfig], preprocessing_params: dict, test_fold: int = 0, val_fold: int = 0, copy: bool = True, dataset_name: str = None, skipna_samples: bool = True, **kwargs ) -> List[pd.DataFrame]:
    """
    Applies preprocessing to a fold
    Parameters
    ----------
    sets : pd.DataFrame
    preprocessing_paths : Union[dict, DictConfig]
    preprocessing_params : dict
    test_fold : int (default: 0)
    val_fold : int (default: 0)
    copy : bool (default: True)
    dataset_name: str
    skipna_samples: bool (default: False)
    kwargs : dict

    Returns
    -------
    List[pd.DataFrame]
    """
    if copy:
        sets = [fset.copy() for fset in sets]
    filename_wo_extension = f"{test_fold}_{val_fold}" + (f"_{dataset_name}" if dataset_name else "")

    categorical_path = os.path.join( preprocessing_paths["categorical"], filename_wo_extension + ".csv" )
    if os.path.exists(categorical_path):
        categorical_params = save_load.load_params_table(categorical_path, index_col=0).squeeze().to_dict()
        sets = [ categorical_preprocessing(fset, **categorical_params) for fset in sets ]

    numerical_path = os.path.join(preprocessing_paths["numerical"], filename_wo_extension + ".csv")
    if os.path.exists(numerical_path):
        numerical_params = save_load.load_params_table(numerical_path, index_col=0).squeeze().to_dict()
        sets = [ numerical_preprocessing(fset, **numerical_params) for fset in sets]

    imputer_path = os.path.join(preprocessing_paths["imputer"], filename_wo_extension + ".pkl")
    if os.path.exists(imputer_path):
        imputer = save_load.load_model(imputer_path)
        imputer_params = save_load.load_params_table( os.path.join( preprocessing_paths["imputer"], filename_wo_extension + ".csv" ), index_col=0).squeeze().to_dict()
        sets = [ impute_missing(fset, features_info=preprocessing_params, imputer=imputer, skipna_samples=skipna_samples, **imputer_params) for fset in sets ]

    return sets


def describe_features(data: pd.DataFrame, features_dtypes: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame()
    n_samples = data.shape[0]
    for feature, dtype in features_dtypes.items():

        nan_number = data[feature].isna().sum()
        feature_info = {"Feature": feature}

        if dtype in ["float", "int"]:
            mean = np.round(data[feature].mean(), 2)

            feature_info["Categories"] = [f"<{mean}", f"≥{mean}"]

            le_num = np.sum(data[feature] < mean)
            ge_num = np.sum(data[feature] >= mean)
            feature_info["Distribution"] = [ f"{le_num} ({np.round(100*(le_num/n_samples), 2)}%)", f"{ge_num} ({np.round(100*(ge_num/n_samples), 2)}%)" ]

        else:
            feat_info = data[feature].value_counts().sort_index().to_dict()
            feature_info["Categories"] = list(feat_info.keys())
            feature_info["Distribution"] = [ f"{v} ({np.round(100*(v/n_samples), 2)}%)" for v in list(feat_info.values())]

        feature_info = pd.DataFrame(feature_info)
        nan_col = pd.Series( [f"{nan_number} ({np.round(100 * (nan_number / n_samples), 2)}%)"] + [np.nan] * (len(feature_info["Categories"]) - 1))
        feature_info.insert(1, "Missing Data", nan_col )
        features = pd.concat([features, feature_info], axis=0, ignore_index=True)

    return features


def join_tabular_preprocessing_params(*params: dict) -> dict:
    """
    Joins the preprocessing parameters
    Parameters
    ----------
    params : dict

    Returns
    -------
    dict
    """
    preprocessing_params = params[0].copy()
    for param in params[1:]:
        n_features = len(preprocessing_params["features_types"])
        for key, value in param.items():
            if key not in preprocessing_params.keys():
                preprocessing_params[key] = value
            else:
                if key in ("numerical_columns", "categorical_columns"):
                    value = {k + n_features: v for k, v in value.items()}
                    preprocessing_params[key] = {**preprocessing_params[key], **value}
                elif key in ("categorical_unique_values", "features_types"):
                    preprocessing_params[key] = pd.concat([preprocessing_params[key], value])
                else:
                    assert preprocessing_params[key] == value, f"Preprocessing parameter {key} is different in the datasets"
    return preprocessing_params


if __name__ == "__main__":
    pass
