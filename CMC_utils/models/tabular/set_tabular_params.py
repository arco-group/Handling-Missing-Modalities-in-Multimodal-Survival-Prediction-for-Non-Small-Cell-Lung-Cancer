import os
import logging

import numpy as np
from typing import Tuple, Union

import pandas as pd
from hydra.utils import instantiate
from numpy.array_api import unique_values
from omegaconf import DictConfig, ListConfig
from CMC_utils.miscellaneous import recursive_cfg_search, recursive_cfg_substitute
from CMC_utils.datasets import SupervisedTabularDatasetTorch

__all__ = ["set_custom_mlp_params", "set_adaboost_params", "set_histgradientboostingtree_params", "set_naim_params", "set_maria_params", "set_tabnet_params", "set_tabnet_torch_params", "set_xgboost_params", "set_fttransformer_params", "set_tabtransformer_params", "set_survival_params", "set_custom_mlp_survival_regression_params", "set_multimodal_mlp_head_params"]


log = logging.getLogger(__name__)


def compute_categorical_idxs_dims(columns: list, preprocessing_params: Union[dict, DictConfig]) -> Tuple[list, list]:
    """
    Compute the indexes and dimensions of the categorical features in the dataset
    Parameters
    ----------
    columns : list
    preprocessing_params : Union[dict, DictConfig]

    Returns
    -------
    Tuple[list, list]
        categorical_idxs, categorical_dims
    """
    if isinstance(preprocessing_params, ListConfig):
        unique_values = pd.concat([params.categorical_unique_values for params in preprocessing_params])
        numerical_columns = list([val for params in preprocessing_params for val in params.numerical_columns.values()])
    else:
        unique_values = preprocessing_params.categorical_unique_values
        numerical_columns = list(preprocessing_params.numerical_columns.values())

    categorical_columns = tuple(unique_values.index.to_list())


    categorical_idxs = []
    categorical_dims = []
    for idx, col in enumerate(columns):
        if col in categorical_columns:
            categorical_idxs.append(idx)
            categorical_dims.append(len(unique_values[col]))
        elif col.startswith(categorical_columns) and not col in numerical_columns:
            categorical_idxs.append(idx)
            categorical_dims.append(2)
    return categorical_idxs, categorical_dims


def set_custom_mlp_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the CustomMLP model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    model_cfg["init_params"]["input_size"] = train_set.input_size
    model_cfg["init_params"]["output_size"] = train_set.output_size

    return model_cfg

def set_custom_mlp_survival_regression_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the CustomMLP model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    model_cfg["init_params"]["input_size"] = train_set.input_size
    model_cfg["init_params"]["num_events"] = preprocessing_params["num_events"]
    model_cfg["init_params"]["max_time"] = preprocessing_params["max_time"]
    #model_cfg["init_params"]["output_size"] = train_set.output_size

    return model_cfg

def set_adaboost_params(model_cfg: dict, **_) -> dict:
    """
    Set the parameters of the HistGradientBoostingTree model
    Parameters
    ----------
    model_cfg : dict

    Returns
    -------
    dict
        model_cfg
    """
    if model_cfg["init_params"].get("estimator", None) is not None:
        model_cfg["init_params"]["estimator"] = instantiate(model_cfg["init_params"]["estimator"])

    return model_cfg


def set_histgradientboostingtree_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the HistGradientBoostingTree model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    model_cfg["init_params"]["categorical_features"] = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)[0]

    return model_cfg


def set_naim_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, checkpoint_path: str = None, test_fold: int = None, val_fold: int = None, **_) -> dict:
    """
    Set the parameters of the NAIM model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    model_cfg["init_params"]["input_size"] = train_set.input_size
    model_cfg["init_params"]["output_size"] = train_set.output_size

    cat_idxs, cat_dims = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)
    model_cfg["init_params"]["cat_idxs"] = cat_idxs
    model_cfg["init_params"]["cat_dims"] = cat_dims

    searched_value, key_found = recursive_cfg_search(model_cfg, "d_token")
    if searched_value is None and key_found:
        d_token, _ = recursive_cfg_search(model_cfg, "input_size")  # cfg.model.init_params.input_size  # train_set.input_size  # train_set.input_size  # + train_set.input_size #//2
        log.info(f"Token dimension: {d_token}")  # + {int(np.ceil(np.log2(train_set.input_size)))}")
        model_cfg = recursive_cfg_substitute(model_cfg, {"d_token": d_token})

    searched_value, key_found = recursive_cfg_search(model_cfg, "num_heads")
    if searched_value is None and key_found:
        divisor = 2  # 2  # 1  # divisors(d_token)[1]  # [-2]
        log.info(f"N. attention heads: {divisor}")
        model_cfg = recursive_cfg_substitute(model_cfg, {"num_heads": divisor})

    searched_value, key_found = recursive_cfg_search(model_cfg, "checkpoint_path")
    if searched_value is None and key_found:
        model_name = model_cfg["name"]
        filename = f"{test_fold}_{val_fold}.{model_cfg['file_extension']}"
        basename = os.path.basename(checkpoint_path)
        if basename != "checkpoints":
            dirname = os.path.dirname(checkpoint_path)
            checkpoint_path = os.path.join(dirname, model_name, basename, filename)
        else:
            checkpoint_path = os.path.join(checkpoint_path, model_name, filename)
        model_cfg = recursive_cfg_substitute(model_cfg, {"checkpoint_path": checkpoint_path})

    return model_cfg


def set_maria_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the NAIM model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    model_cfg["init_params"]["input_size"] = train_set.input_size
    model_cfg["init_params"]["output_size"] = train_set.output_size
    model_cfg["init_params"]["n_modalities"] = len(train_set.input_size)

    cat_idxs, cat_dims = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)
    model_cfg["init_params"]["cat_idxs"] = cat_idxs
    model_cfg["init_params"]["cat_dims"] = cat_dims

    searched_value, key_found = recursive_cfg_search(model_cfg, "d_token")
    if searched_value is None and key_found:
        d_token, _ = recursive_cfg_search(model_cfg, "input_size")  # cfg.model.init_params.input_size  # train_set.input_size  # train_set.input_size  # + train_set.input_size #//2
        log.info(f"Token dimension: {d_token}")  # + {int(np.ceil(np.log2(train_set.input_size)))}")
        model_cfg = recursive_cfg_substitute(model_cfg, {"d_token": d_token})

    searched_value, key_found = recursive_cfg_search(model_cfg, "num_heads")
    if searched_value is None and key_found:
        divisor = 2  # 2  # 1  # divisors(d_token)[1]  # [-2]
        log.info(f"N. attention heads: {divisor}")
        model_cfg = recursive_cfg_substitute(model_cfg, {"num_heads": divisor})

    return model_cfg


def set_multimodal_mlp_head_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], shared_network_config: dict,  train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the NAIM model
    Parameters
    ----------
    shared_network_config
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    model_cfg["init_params"]["input_size"] = shared_network_config.input_size
    model_cfg["init_params"]["output_size"] = shared_network_config.output_size
    model_cfg["init_params"]["n_modalities"] = 1

    cat_idxs, cat_dims = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)
    model_cfg["init_params"]["cat_idxs"] = cat_idxs
    model_cfg["init_params"]["cat_dims"] = cat_dims

    searched_value, key_found = recursive_cfg_search(model_cfg, "d_token")
    if searched_value is None and key_found:
        d_token, _ = recursive_cfg_search(model_cfg, "input_size")  # cfg.model.init_params.input_size  # train_set.input_size  # train_set.input_size  # + train_set.input_size #//2
        log.info(f"Token dimension: {d_token}")  # + {int(np.ceil(np.log2(train_set.input_size)))}")
        model_cfg = recursive_cfg_substitute(model_cfg, {"d_token": d_token})

    searched_value, key_found = recursive_cfg_search(model_cfg, "num_heads")
    if searched_value is None and key_found:
        divisor = 2  # 2  # 1  # divisors(d_token)[1]  # [-2]
        log.info(f"N. attention heads: {divisor}")
        model_cfg = recursive_cfg_substitute(model_cfg, {"num_heads": divisor})

    return model_cfg

def set_tabnet_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, val_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the TabNet model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch
    val_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    cat_idxs, cat_dims = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)
    model_cfg["init_params"]["cat_idxs"] = cat_idxs
    model_cfg["init_params"]["cat_dims"] = cat_dims

    train_data, train_labels = train_set.get_data()[:-1]
    val_data, val_labels = val_set.get_data()[:-1]
    if train_set.label_type == "continuous":
        train_labels = np.expand_dims(train_labels, 1)
        val_labels = np.expand_dims(val_labels, 1)

    model_cfg["fit_params"]["eval_set"] = [(train_data, train_labels), (val_data, val_labels)]
    model_cfg["fit_params"]["eval_name"] = ["train", "val"]

    return model_cfg


def set_tabnet_torch_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the TabNet model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch
    val_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """

    model_cfg["init_params"]["input_dim"] = train_set.input_size
    model_cfg["init_params"]["output_dim"] = train_set.output_size

    cat_idxs, cat_dims = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)
    model_cfg["init_params"]["cat_idxs"] = cat_idxs
    model_cfg["init_params"]["cat_dims"] = cat_dims

    return model_cfg


def set_xgboost_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, val_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the XGBoost model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedTabularDatasetTorch
    val_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    cat_idxs = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)[0]
    model_cfg["init_params"]["feature_types"] = list("c" if n in cat_idxs else "q" for n in range(train_set.input_size))
    if train_set.label_type != "continuous":
        model_cfg["init_params"]["num_class"] = len(preprocessing_params["classes"])

    model_cfg["fit_params"]["eval_set"] = [train_set.get_data()[:-1], val_set.get_data()[:-1]]

    return model_cfg


def set_fttransformer_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    cat_idxs, cat_dims = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)
    model_cfg["init_params"]["cat_idxs"] = cat_idxs
    model_cfg["init_params"]["categories"] = cat_dims
    model_cfg["init_params"]["num_continuous"] = len(preprocessing_params["numerical_columns"])
    model_cfg["init_params"]["dim_out"] = len(preprocessing_params.get("classes", [1]))
    return model_cfg


def set_tabtransformer_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    cat_idxs, cat_dims = compute_categorical_idxs_dims(train_set.columns, preprocessing_params)
    model_cfg["init_params"]["cat_idxs"] = cat_idxs
    model_cfg["init_params"]["categories"] = cat_dims
    model_cfg["init_params"]["num_continuous"] = len(preprocessing_params["numerical_columns"])
    model_cfg["init_params"]["dim_out"] = len(preprocessing_params.get("classes", [1]))
    return model_cfg


def set_survival_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], **_) -> dict:
    """
    Set the parameters of the Survival model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig], optional

    Returns
    -------
    dict
        model_cfg
    """
    model_cfg["num_events"] = preprocessing_params["num_events"]
    model_cfg["max_time"] = preprocessing_params["max_time"]

    # model_cfg = set_model_label_type(model_cfg, preprocessing_params)
    return model_cfg


if __name__ == "__main__":
    pass
