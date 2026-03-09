import torch
import torchxrayvision as xrv

from CMC_utils.miscellaneous import do_really_nothing
from CMC_utils.models import Extractor, freeze_params, set_pretrained_attribute

__all__ = ["get_CXRdensenet", "get_CXRresnet"]

def get_CXRdensenet(weights="densenet121-res224-all", num_classes=None, freeze_model_params=False, extractor=False, op_threshs=None, apply_sigmoid=False, **kwargs) -> torch.nn.Module:
    """
    Get DenseNet model pretrained on CXR data
    Parameters
    ----------
    weights : str
    num_classes : int
    freeze_model_params : bool
    extractor : bool
    d_token : int
    op_threshs : list
    apply_sigmoid : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    
    model = xrv.models.DenseNet(weights=weights, op_threshs=op_threshs, apply_sigmoid=apply_sigmoid)
    setattr(model, "input_size", (1, 224, 224))
    setattr(model, "d_token", 1024)

    set_pretrained_attribute(model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
        model.op_threshs = None if len(model.op_threshs) == 18 else model.op_threshs
        setattr(model, "output_size", num_classes)
    elif extractor:
        num_ftrs = model.classifier.in_features
        setattr(model, "output_size", (num_ftrs,))
        model.classifier = torch.nn.Identity()
        model.op_threshs = None if len(model.op_threshs) == 18 else model.op_threshs
        model = Extractor(model, num_ftrs, **kwargs)
        return model
    else:
        setattr(model, "output_size", model.classifier.out_features)

    return model


def get_CXRresnet(weights="resnet50-res512-all", num_classes=None, freeze_model_params=False, extractor=False, apply_sigmoid=False, **kwargs) -> torch.nn.Module:
    """
    Get DenseNet model pretrained on CXR data
    Parameters
    ----------
    weights : str
    num_classes : int
    freeze_model_params : bool
    extractor : bool
    d_token : int
    op_threshs : list
    apply_sigmoid : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """

    model = xrv.models.ResNet(weights=weights, apply_sigmoid=apply_sigmoid)
    setattr(model, "input_size", (1, 512, 512))
    setattr(model, "d_token", 2048)

    set_pretrained_attribute(model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.model.fc.in_features
        model.model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
        model.op_threshs = None
        setattr(model, "output_size", num_classes)
    elif extractor:
        num_ftrs = model.model.fc.in_features
        setattr(model, "output_size", (num_ftrs,))
        model.model.fc = torch.nn.Identity()
        model.op_threshs = None
        model = Extractor(model, num_ftrs, **kwargs)
        return model
    else:
        setattr(model, "output_size", model.model.fc.out_features)

    return model


if __name__ == "__main__":
    pass
