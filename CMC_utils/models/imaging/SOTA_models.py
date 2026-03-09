import torch
from torchvision import models

from CMC_utils.miscellaneous import do_really_nothing
from CMC_utils.models import Extractor, set_pretrained_attribute, freeze_params, remove_DataParallel_module

__all__ = ["get_alexnet", "get_densenet", "get_efficientnet", "get_googlenet", "get_inception", "get_mnasnet", "get_mobilenet", "get_resnet", "get_resnext", "get_shufflenet", "get_squeezenet", "get_vgg", "get_wide_resnet", "get_vit", "get_dino_vit"]


def get_alexnet(num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, **kwargs) -> torch.nn.Module:
    """
    Get AlexNet model
    Parameters
    ----------
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    params_options = {False: {}, True: {"weights": models.AlexNet_Weights.DEFAULT}}

    model = models.alexnet(**params_options[load_weights])

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)

    if extractor:
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


def get_densenet(name="densenet121", num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, d_token=1024, **kwargs) -> torch.nn.Module:
    """
    Get DenseNet model
    Parameters
    ----------
    name : str
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    d_token : int
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    models_dict = {"densenet121": models.densenet121, "densenet161": models.densenet161,
                   "densenet169": models.densenet169, "densenet201": models.densenet201}

    models_params = {"densenet121": models.DenseNet121_Weights, "densenet161": models.DenseNet161_Weights,
                     "densenet169": models.DenseNet169_Weights, "densenet201": models.DenseNet201_Weights}

    params_options = {False: {}, True: {"weights": models_params[name].DEFAULT}}

    model = models_dict[name](**params_options[load_weights])
    setattr(model, "input_size", (3, 224, 224))
    setattr(model, "d_token", d_token)

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
        setattr(model, "output_size", num_classes)

    if extractor:
        num_ftrs = model.classifier.in_features
        setattr(model, "output_size", (num_ftrs,))
        model.classifier = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


def get_efficientnet(name="efficientnet_b0", num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, d_token=1024, **kwargs) -> torch.nn.Module:
    """
    Get Wide EfficientNet model
    Parameters
    ----------
    name : str
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    d_token : int
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    from torchvision.models._api import WeightsEnum
    from torch.hub import load_state_dict_from_url

    def get_state_dict(self, *args, **kwargs):
        kwargs["check_hash"] = False  # kwargs.pop("check_hash")
        return load_state_dict_from_url(self.url, *args, **kwargs)

    WeightsEnum.get_state_dict = get_state_dict

    models_dict = {"efficientnet_b0": models.efficientnet_b0, "efficientnet_b1": models.efficientnet_b1, "efficientnet_b2": models.efficientnet_b2, "efficientnet_b3": models.efficientnet_b3, "efficientnet_b4": models.efficientnet_b4, "efficientnet_b5": models.efficientnet_b5, "efficientnet_b6": models.efficientnet_b6, "efficientnet_b7": models.efficientnet_b7}

    models_params = {"efficientnet_b0": models.EfficientNet_B0_Weights,
                     "efficientnet_b1": models.EfficientNet_B1_Weights,
                     "efficientnet_b2": models.EfficientNet_B2_Weights,
                     "efficientnet_b3": models.EfficientNet_B3_Weights,
                     "efficientnet_b4": models.EfficientNet_B4_Weights,
                     "efficientnet_b5": models.EfficientNet_B5_Weights,
                     "efficientnet_b6": models.EfficientNet_B6_Weights,
                     "efficientnet_b7": models.EfficientNet_B7_Weights}

    params_options = {False: {}, True: {"weights": models_params[name].DEFAULT}}

    model = models_dict[name](**params_options[load_weights])

    setattr(model, "input_size", (3, 224, 224))
    setattr(model, "d_token", d_token)

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.classifier[-1].in_features
        model.classifier = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
        setattr(model, "output_size", num_classes)

    if extractor:
        num_ftrs = model.classifier[-1].in_features
        model.classifier = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


def get_googlenet(num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, **kwargs) -> torch.nn.Module:
    """
    Get GoogLeNet model
    Parameters
    ----------
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    params_options = {False: {}, True: {"weights": models.GoogLeNet_Weights.DEFAULT}}

    model = models.googlenet(**params_options[load_weights])

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.fc.in_features
        model.fc.in_features = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)

    if extractor:
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


def get_inception(num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, **kwargs) -> torch.nn.Module:
    """
    Get Inception model
    Parameters
    ----------
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    params_options = {False: {}, True: {"weights": models.Inception_V3_Weights.DEFAULT}}

    model = models.inception_v3(**params_options[load_weights])

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)

    if extractor:
        num_ftrs1 = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = torch.nn.Identity()
        num_ftrs2 = model.fc.in_features
        model.fc = torch.nn.Identity()
        model = Extractor(model, num_ftrs1, num_ftrs2, **kwargs)
        return model

    return model


def get_mnasnet(name="mnasnet0_5", num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, **kwargs) -> torch.nn.Module:
    """
    Get MNASNet model
    Parameters
    ----------
    name : str
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    models_dict = {"mnasnet0_5": models.mnasnet0_5, "mnasnet0_75": models.mnasnet0_75,
                   "mnasnet1_0": models.mnasnet1_0, "mnasnet1_3": models.mnasnet1_3}

    models_params = {"mnasnet0_5": models.MNASNet0_5_Weights, "mnasnet0_75": models.MNASNet0_75_Weights,
                     "mnasnet1_0": models.MNASNet1_0_Weights, "mnasnet1_3": models.MNASNet1_3_Weights}

    params_options = {False: {}, True: {"weights": models_params[name].DEFAULT}}

    model = models_dict[name](**params_options[load_weights])

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)

    if extractor:
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


def get_mobilenet(name="mobilenet_v2", num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, **kwargs) -> torch.nn.Module:
    """
    Get MobileNet model
    Parameters
    ----------
    name : str
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    models_dict = {"mobilenet_v2": models.mobilenet_v2, "mobilenet_v3_small": models.mobilenet_v3_small,
                   "mobilenet_v3_large": models.mobilenet_v3_large}

    models_params = {"mobilenet_v2": models.MobileNet_V2_Weights,
                     "mobilenet_v3_small": models.MobileNet_V3_Small_Weights,
                     "mobilenet_v3_large": models.MobileNet_V3_Large_Weights}

    params_options = {False: {}, True: {"weights": models_params[name].DEFAULT}}

    model = models_dict[name](**params_options[load_weights])

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)

    if extractor:
        num_ftrs = model.classifier[-1].in_features
        model.classifier = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model

# TODO generalize load weights through checkpoint_path for all models
def get_resnet(name="resnet18", num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, checkpoint_path: str = None, **kwargs) -> torch.nn.Module:
    """
    Get ResNet model
    Parameters
    ----------
    name : str
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    models_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34,
                   "resnet50": models.resnet50, "resnet101": models.resnet101,
                   "resnet152": models.resnet152}

    models_params = {"resnet18": models.ResNet18_Weights, "resnet34": models.ResNet34_Weights,
                     "resnet50": models.ResNet50_Weights, "resnet101": models.ResNet101_Weights,
                     "resnet152": models.ResNet152_Weights}

    params_options = {False: {}, True: {"weights": models_params[name].DEFAULT}}

    model = models_dict[name](**params_options[load_weights and checkpoint_path is not None])

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes:  # and not extractor:
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)

    if load_weights and checkpoint_path is not None:
        model_state_dict = torch.load(checkpoint_path, map_location="cpu")
        model_state_dict = remove_DataParallel_module(model_state_dict)
        model.load_state_dict(model_state_dict)

        load_options = {False: do_really_nothing, True: set_pretrained_attribute}
        load_options[load_weights](model)

        freeze_options = {False: do_really_nothing, True: freeze_params}
        freeze_options[freeze_model_params](model)

    if extractor:
        num_ftrs = model.fc.in_features

        setattr(model, "input_size", (3, 224, 224))
        setattr(model, "d_token", num_ftrs)

        model.fc = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


def get_resnext(name="resnext50_32x4d", num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, **kwargs) -> torch.nn.Module:
    """
    Get ResNeXt model
    Parameters
    ----------
    name : str
    num_classes : int
    load_weights : bool
    extractor : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    models_dict = {"resnext50_32x4d": models.resnext50_32x4d, "resnext101_32x8d": models.resnext101_32x8d,
                   "resnext101_64x4d": models.resnext101_64x4d}

    models_params = {"resnext50_32x4d": models.ResNeXt50_32X4D_Weights,
                     "resnext101_32x8d": models.ResNeXt101_32X8D_Weights,
                     "resnext101_64x4d": models.ResNeXt101_64X4D_Weights}

    params_options = {False: {}, True: {"weights": models_params[name].DEFAULT}}

    model = models_dict[name](**params_options[load_weights])

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)

    if extractor:
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


def get_shufflenet(name="shufflenet_v2_x0_5", num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, **kwargs) -> torch.nn.Module:
    """
    Get ShuffleNet model
    Parameters
    ----------
    name : str
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    models_dict = {"shufflenet_v2_x0_5": models.shufflenet_v2_x0_5, "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
                   "shufflenet_v2_x1_5": models.shufflenet_v2_x1_5, "shufflenet_v2_x2_0": models.shufflenet_v2_x2_0}

    models_params = {"shufflenet_v2_x0_5": models.ShuffleNet_V2_X0_5_Weights,
                     "shufflenet_v2_x1_0": models.ShuffleNet_V2_X1_0_Weights,
                     "shufflenet_v2_x1_5": models.ShuffleNet_V2_X1_5_Weights,
                     "shufflenet_v2_x2_0": models.ShuffleNet_V2_X2_0_Weights}

    params_options = {False: {}, True: {"weights": models_params[name].DEFAULT}}

    model = models_dict[name](**params_options[load_weights])

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)

    if extractor:
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


def get_squeezenet(name="squeezenet1_0", num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, **kwargs) -> torch.nn.Module:
    """
    Get SqueezeNet model
    Parameters
    ----------
    name : str
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    models_dict = {"squeezenet1_0": models.squeezenet1_0, "squeezenet1_1": models.squeezenet1_1}

    models_params = {"squeezenet1_0": models.SqueezeNet1_0_Weights, "squeezenet1_1": models.SqueezeNet1_1_Weights}

    params_options = {False: {}, True: {"weights": models_params[name].DEFAULT}}

    model = models_dict[name](**params_options[load_weights])

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.classifier[1].in_channels
        model.classifier[1] = torch.nn.Conv2d(in_channels=num_ftrs, out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1))

    if extractor:
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


def get_vgg(name="vgg11", num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, **kwargs) -> torch.nn.Module:
    """
    Get VGG model
    Parameters
    ----------
    name : str
    num_classes : int
    load_weights : bool
    extractor : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    models_dict = {"vgg11": models.vgg11, "vgg11_bn": models.vgg11_bn,
                   "vgg13": models.vgg13, "vgg13_bn": models.vgg13_bn,
                   "vgg16": models.vgg16, "vgg16_bn": models.vgg16_bn,
                   "vgg19": models.vgg19, "vgg19_bn": models.vgg19_bn}

    models_params = {"vgg11": models.VGG11_Weights, "vgg11_bn": models.VGG11_BN_Weights,
                     "vgg13": models.VGG13_Weights, "vgg13_bn": models.VGG13_BN_Weights,
                     "vgg16": models.VGG16_Weights, "vgg16_bn": models.VGG16_BN_Weights,
                     "vgg19": models.VGG19_Weights, "vgg19_bn": models.VGG19_BN_Weights}

    params_options = {False: {}, True: {"weights": models_params[name].DEFAULT}}

    model = models_dict[name](**params_options[load_weights])

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)

    if extractor:
        num_ftrs = model.classifier[0].in_features
        model.classifier = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


def get_wide_resnet(name="wide_resnet50_2", num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, **kwargs) -> torch.nn.Module:
    """
    Get Wide ResNet model
    Parameters
    ----------
    name : str
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    models_dict = {"wide_resnet50_2": models.wide_resnet50_2, "wide_resnet101_2": models.wide_resnet101_2}

    models_params = {"wide_resnet50_2": models.Wide_ResNet50_2_Weights,
                     "wide_resnet101_2": models.Wide_ResNet101_2_Weights}

    params_options = {False: {}, True: {"weights": models_params[name].DEFAULT}}

    model = models_dict[name](**params_options[load_weights])

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)

    if extractor:
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


def get_vit(name="vit_b_16", num_classes=None, load_weights=False, freeze_model_params=False, extractor=False, d_token=768, **kwargs) -> torch.nn.Module:
    """
    Get ViT model
    Parameters
    ----------
    name : str
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    d_token : int

    patch_size=16
    num_layers=24
    num_heads=16
    hidden_dim=1024
    mlp_dim=4096
    weights=weights
    progress=progress

    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """

    models_dict = {"vit_b_16": models.vit_b_16, "vit_b_32": models.vit_b_32, "vit_l_16": models.vit_l_16, "vit_l_32": models.vit_l_32, "vit_h_14": models.vit_h_14}

    models_params = {"vit_b_16": models.ViT_B_16_Weights,
                     "vit_b_32": models.ViT_B_32_Weights,
                     "vit_l_16": models.ViT_L_16_Weights,
                     "vit_l_32": models.ViT_L_32_Weights,
                     "vit_h_14": models.ViT_H_14_Weights}

    params_options = {False: {}, True: {"weights": models_params[name].DEFAULT}}

    model = models_dict[name](**params_options[load_weights])

    setattr(model, "input_size", (3, 224, 224))
    setattr(model, "d_token", d_token)

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.heads[0].in_features
        model.heads = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
        setattr(model, "output_size", num_classes)

    if extractor:
        def extractor_forward(self, data, **_):
            # Reshape and permute the input tensor
            data = self._process_input(data)
            n = data.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            data = torch.cat([batch_class_token, data], dim=1)

            emb = self.encoder(data)
            emb = emb.view(n, -1)
            return emb

        model.heads = torch.nn.Identity()
        # num_ftrs = model.heads[0].in_features

        num_ftrs = ((model.input_size[-2]*model.input_size[-1])/(model.conv_proj.kernel_size[0]*model.conv_proj.kernel_size[1])) * model.d_token + model.d_token
        model.forward = extractor_forward.__get__(model, torch.nn.Module)

        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


def get_dino_vit(name: str = "dinov2_vitg14", num_classes: int = None, freeze_model_params: bool = False, checkpoint_path: str = None, extractor: bool = False, **kwargs) -> torch.nn.Module:
    """
    Get ViT model
    Parameters
    ----------
    name : str
    num_classes : int
    freeze_model_params : bool
    checkpoint_path : str
    extractor : bool

    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """

    model = torch.hub.load('facebookresearch/dinov2', name)
    d_token = model.num_features
    setattr(model, "input_size", (3, 224, 224))
    setattr(model, "d_token", d_token)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=False)

    set_pretrained_attribute(model)

    freeze_params(model, freeze=freeze_model_params)

    if num_classes and not extractor:
        model.head = torch.nn.Linear(in_features=d_token, out_features=num_classes)
        setattr(model, "output_size", num_classes)

    if extractor:
        def extractor_forward(self, data, **_):
            # Reshape and permute the input tensor
            emb = self.patch_embed(data)
            n = data.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.cls_token.expand(n, -1, -1)
            emb = torch.cat([batch_class_token, emb], dim=1)

            for blk in self.blocks:
                emb = blk(emb)

            emb = emb.view(n, -1)
            return emb

        num_ftrs = ((model.input_size[-2] * model.input_size[-1]) / (
                    model.patch_embed.proj.kernel_size[0] * model.patch_embed.proj.kernel_size[1])) * d_token + d_token

        model.forward = extractor_forward.__get__(model, torch.nn.Module)

        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


if __name__ == "__main__":
    pass
