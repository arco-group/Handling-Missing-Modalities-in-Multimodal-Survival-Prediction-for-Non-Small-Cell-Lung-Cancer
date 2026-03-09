import timm
import torch

from CMC_utils.miscellaneous import do_really_nothing
from CMC_utils.models import set_pretrained_attribute, freeze_params

__all__ = ["get_timm_model"]


def get_timm_model(name: str = 'vit_base_patch16_224_in21k', img_size: int = 512, num_classes: int = 2, pretrained: bool = False, freeze_model_params: bool = False, checkpoint_path: str = None, extractor: bool = False, **kwargs):
    # Load a pre-trained model from config file
    model = timm.create_model(name, num_classes=num_classes, img_size=img_size, pretrained=pretrained, features_only=extractor)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[pretrained or (checkpoint_path is not None)](model)

    freeze_params(model, freeze=freeze_model_params)

    return model


if __name__ == "__main__":
    pass
