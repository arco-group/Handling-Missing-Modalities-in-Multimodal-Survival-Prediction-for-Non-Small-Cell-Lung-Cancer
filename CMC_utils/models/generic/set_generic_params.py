import logging
from typing import Union, List, DefaultDict
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf, ListConfig
from CMC_utils.miscellaneous import join_preprocessing_params, recursive_cfg_substitute
from CMC_utils.datasets import MultimodalJointFusionTabularDatasetTorch

__all__ = ["set_survival_wrapper_params", "set_multimodal_params", "set_multimodal_survival_params"]


log = logging.getLogger(__name__)


def set_survival_wrapper_params(model_cfg: dict = {}, preprocessing_params: Union[dict, DictConfig] = {}, **kwargs) -> dict:
    """
    Set the parameters of the survival wrapper model.
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : dict
    kwargs : dict

    Returns
    -------
    dict
    """
    if isinstance(preprocessing_params, ListConfig):
        model_cfg["init_params"]["num_events"] = preprocessing_params[0]["num_events"]
        model_cfg["init_params"]["num_events"] = preprocessing_params[0]["max_time"]
    else:
        model_cfg["init_params"]["num_events"] = preprocessing_params["num_events"]
        model_cfg["init_params"]["max_time"] = preprocessing_params["max_time"]

    model_cfg["init_params"]["shared_net"] = call(model_cfg["init_params"]["shared_net"]["set_params_function"], model_cfg["init_params"]["shared_net"], preprocessing_params=preprocessing_params, **kwargs )
    model_cfg["init_params"]["cs_subnet"] = call(model_cfg["init_params"]["cs_subnet"]["set_params_function"], model_cfg["init_params"]["cs_subnet"], preprocessing_params=preprocessing_params, **kwargs )

    # model_cfg["actual_label_type"] = preprocessing_params["label_type"]
    return model_cfg


def set_multimodal_survival_params(model_cfg: dict, preprocessing_params: List[Union[dict, DictConfig]], train_set: MultimodalJointFusionTabularDatasetTorch, checkpoint_path: str = None, test_fold: int = None, val_fold: int = None, **kwargs) -> dict:
    """
    Set the parameters of the Multimodal model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : List[Union[dict, DictConfig]], optional
    train_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """

    for ms_model, params, fset in zip(model_cfg["init_params"]["ms_models"].keys(), preprocessing_params, train_set.datasets):
        model_cfg["init_params"]["ms_models"][ms_model] = call(model_cfg["init_params"]["ms_models"][ms_model]["set_params_function"], OmegaConf.create(model_cfg["init_params"]["ms_models"][ms_model]), preprocessing_params=params, train_set=fset, checkpoint_path=checkpoint_path, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

    if isinstance(preprocessing_params, ListConfig):
        model_cfg["init_params"]["num_events"] = preprocessing_params[0]["num_events"]
        model_cfg["init_params"]["max_time"] = preprocessing_params[0]["max_time"]
    else:
        model_cfg["init_params"]["num_events"] = preprocessing_params["num_events"]
        model_cfg["init_params"]["max_time"] = preprocessing_params["max_time"]


    model_cfg["init_params"]["shared_net"] = call(model_cfg["init_params"]["shared_net"]["set_params_function"], OmegaConf.create(model_cfg["init_params"]["shared_net"]), preprocessing_params=join_preprocessing_params(*preprocessing_params), train_set=train_set, _recursive_=False)
    shared_network_config = DefaultDict()

    shared_network_config["input_size"] = model_cfg['init_params']["shared_net"]['init_params']['feedforward_dim']
    shared_network_config['output_size'] = None
    model_cfg["init_params"]["cs_subnet"] = call(model_cfg["init_params"]["cs_subnet"]["set_params_function"], model_cfg["init_params"]["cs_subnet"], preprocessing_params=preprocessing_params, train_set=train_set, shared_network_config=shared_network_config, **kwargs)

    return model_cfg


def set_multimodal_params(model_cfg: dict, preprocessing_params: List[Union[dict, DictConfig]], train_set: MultimodalJointFusionTabularDatasetTorch, checkpoint_path: str = None, test_fold: int = None, val_fold: int = None, **_) -> dict:
    """
    Set the parameters of the Multimodal model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : List[Union[dict, DictConfig]], optional
    train_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    for ms_model, params, fset in zip(model_cfg["init_params"]["ms_models"].keys(), preprocessing_params, train_set.datasets):
        model_cfg["init_params"]["ms_models"][ms_model] = call(model_cfg["init_params"]["ms_models"][ms_model]["set_params_function"], OmegaConf.create(model_cfg["init_params"]["ms_models"][ms_model]), preprocessing_params=params, train_set=fset, checkpoint_path=checkpoint_path, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

    model_cfg["init_params"]["shared_net"] = call(model_cfg["init_params"]["shared_net"]["set_params_function"], OmegaConf.create(model_cfg["init_params"]["shared_net"]), preprocessing_params=join_preprocessing_params(*preprocessing_params), train_set=train_set, _recursive_=False)

    return model_cfg


if __name__ == "__main__":
    pass
