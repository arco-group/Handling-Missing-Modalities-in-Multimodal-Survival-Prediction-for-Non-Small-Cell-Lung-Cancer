import logging
import os.path

import torch.nn
from torch.nn.init import *
from typing import Union, List
from collections import OrderedDict
from torch.nn import Sequential, ModuleList, Conv2d, BatchNorm2d, Linear, Module

__all__ = ["list_model_modules", "initialize_weights", "set_device", "set_pretrained_attribute", "freeze_params", "remove_DataParallel_module", "load_pretrained_weights"]


def list_model_modules(model: Union[Module, ModuleList, Sequential]) -> List[Module]:
    """
    List all modules in a model
    Parameters
    ----------
    model : Union[Module, ModuleList, Sequential]    model to list modules

    Returns
    -------
    List[Module]   list of modules
    """
    modules = []
    for module in model.children():
        if isinstance(module, (Sequential, ModuleList)):
            modules.extend(list_model_modules(module))
        else:
            modules.append(module)
    return modules

def load_pretrained_weights(model: Union[Module, ModuleList, Sequential], pretrained_model_parts: dict, fold: int=0) -> None:
    """
    Load pretrained weights into the components of a model
    Parameters
    ----------
    model
    pretrained_model_parts
    fold

    Returns
    -------

    """

    freeze_module_params = pretrained_model_parts['freeze_model_params'] if 'freeze_model_params' in pretrained_model_parts else False
    pretrained_model_parts = {k:v for k,v in pretrained_model_parts.items() if 'model' in k}
    for index, (module_name, module_info_path) in enumerate(pretrained_model_parts.items()):
        if isinstance(module_info_path, bool):
            continue

        
        modules = list(model.ms_models.named_modules())[0][-1]
        if 'ms_models' in module_name or 'model' in module_name:

            module = modules[index-1]

            parts = module_info_path.split('/')[-1].split("CTIx" if "CTIx" in module_info_path else "Cixregression")
            if len(parts) == 2:
                left, right = parts[0], parts[1]
                left = left.strip("_")
                right = right.strip("_")

                candidates = []
                for d in os.listdir(os.path.dirname(module_info_path)):
                    if left in d and right in d:
                        candidates.append(d)
            else:
                print("⚠️ 'CTIx' not found in filename")
            if len(candidates) == 0:
                print(f"⚠️ No candidates found for {module_name} in {module_info_path} || dir name parts: {os.listdir(module_info_path)}")
                #continue
            module_info_path = os.path.join(os.path.dirname(module_info_path), candidates[0])

            if os.path.exists(module_info_path):
                cv_setting = os.listdir(module_info_path)[0]
                module_info_path_cv = os.path.join(module_info_path, cv_setting)
                if not os.path.exists(module_info_path_cv):
                    continue
                else:
                    module_info_path_cv_saved = os.path.join(module_info_path_cv, "saved_models")
                    if not os.path.exists(module_info_path_cv_saved):
                        continue
                    else:

                        exp_dir = os.listdir(module_info_path_cv_saved)[0]
                        saved_models = os.path.join(module_info_path_cv_saved, exp_dir)
                        if not os.path.exists(saved_models):
                            continue
                        else:
                            model_exp_path = os.path.join(saved_models, f"{fold}_0.pth")

            module_pretrained = torch.load(model_exp_path, map_location="cpu", weights_only=False)
            state_dict = module_pretrained if "state_dict" not in module_pretrained else module_pretrained["state_dict"]
            state_dict = {module_name.replace('[', '.').replace( ']', '.') + k:v for k,v in state_dict.items() if k in module.state_dict() and v.size() == module.state_dict()[k].size()}

            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            for name, param in model.named_parameters():
                if freeze_module_params and module_name.replace('[', '.').replace( ']', '.')  in name:
                    param.requires_grad = False

            for name, param in model.named_parameters():
                if param.requires_grad:
                    #print(f"{name:<60} {param.numel():>10,}")
                    pass


            print(f"|||| Loaded pretrained weights for {module_name} from {model_exp_path} |||")
            assert len(unexpected) == 0, f"Unexpected keys in state_dict: {unexpected}"
    return model

def initialize_weights(model: Union[Module, ModuleList, Sequential], initializer) -> None:
    """
    Initialize weights of a model
    Parameters
    ----------
    model : Union[Module, ModuleList, Sequential]     model to initialize
    initializer : Union[Dict, str]      dictionary with name and params of the initializer or string with the name of the initializer

    Returns
    -------
    None
    """
    initializer_options = dict(uniform=uniform_,
                               normal=normal_,
                               constant=constant_,
                               ones=ones_,
                               zeros=zeros_,
                               eye=eye_,
                               dirac=dirac_,
                               xavier_normal=xavier_normal_,
                               xavier_uniform=xavier_uniform_,
                               kaiming_uniform=kaiming_uniform_,
                               kaiming_normal=kaiming_normal_,
                               trunc_normal=trunc_normal_,
                               orthogonal=orthogonal_,
                               sparse=sparse_)
    for m in model.modules():
        if getattr(m, "is_pretrained", False):
            continue

        if isinstance(m, Conv2d):
            initializer_options[initializer.name](m.weight, **initializer.params)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, BatchNorm2d):
            constant_(m.weight, 1)
            constant_(m.bias, 0)
        elif isinstance(m, Linear):
            initializer_options[initializer.name](m.weight, **initializer.params)

            if m.bias is not None:
                constant_(m.bias, 0)


def set_device(device: str) -> torch.device:
    """
    Set device for training
    Parameters
    ----------
    device : str   device to use

    Returns
    -------
    torch.device  device to use
    """
    device_options = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}  # "mps": torch.has_mps,
    device = device_options[device] * device + (not device_options[device]) * "cpu"
    device = torch.device(device)
    return device


def set_pretrained_attribute(model: torch.nn.Module, value: bool = True) -> None:
    for module in model.modules():
        module.is_pretrained = value


def freeze_params(model: torch.nn.Module, freeze = True) -> None:
    """
    Set requires_grad=False for all model parameters if freeze_model_params is True
    Parameters
    ----------
    model : torch.nn.Module
    freeze : bool
    kwargs : dict

    Returns
    -------
    None
    """
    for param in model.parameters():
        param.requires_grad = not freeze


def remove_DataParallel_module(model_state_dict):
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k.replace('module.', '')  # remove 'module.' prefix
        new_state_dict[name] = v
    model_state_dict = new_state_dict
    return model_state_dict


if __name__ == "__main__":
    pass
