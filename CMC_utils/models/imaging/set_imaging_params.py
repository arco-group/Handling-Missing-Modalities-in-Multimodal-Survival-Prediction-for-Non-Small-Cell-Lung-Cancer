import os
import logging
from typing import Union
from omegaconf import DictConfig
from CMC_utils.datasets import SupervisedTabularDatasetTorch
from CMC_utils.miscellaneous import recursive_cfg_search, recursive_cfg_substitute


__all__ = ["set_sota_model_params", "set_custom_cnn_params", "set_monai_params", "set_timm_params"]


log = logging.getLogger(__name__)


def set_sota_model_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, checkpoint_path: str = None,  test_fold: int = None, val_fold: int = None, **_) -> dict:
    """
    Set the parameters of the CustomMLP model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedImagingDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    model_cfg["init_params"]["num_classes"] = train_set.output_size

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


def set_custom_cnn_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the CustomCNN model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : Union[dict, DictConfig]
    train_set : SupervisedImagingDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    model_cfg["init_params"]["input_size"] = train_set.input_size[-2:]
    model_cfg["init_params"]["output_size"] = train_set.output_size

    return model_cfg


def set_monai_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:

    model_cfg["init_params"]["in_channels"] = train_set.input_size[-3]
    model_cfg["init_params"]["img_size"] = train_set.input_size[-2:]
    model_cfg["init_params"]["num_classes"] = train_set.output_size

    if model_cfg["init_params"]["in_channels"] == 1:
        model_cfg["init_params"]["spatial_dims"] = 2

    return model_cfg


def set_timm_params(model_cfg: dict, preprocessing_params: Union[dict, DictConfig], train_set: SupervisedTabularDatasetTorch, **_) -> dict:

    model_cfg["init_params"]["img_size"] = train_set.input_size[-1]
    model_cfg["init_params"]["num_classes"] = train_set.output_size

    return model_cfg


if __name__ == "__main__":
    pass
