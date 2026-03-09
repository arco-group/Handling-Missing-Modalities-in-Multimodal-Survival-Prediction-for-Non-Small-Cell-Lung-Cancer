import torch
from monai.networks.nets import ViT
from CMC_utils.models import freeze_params, Extractor
from CMC_utils.miscellaneous import do_really_nothing

__all__ = ["MonaiViT"]


class MonaiViT(ViT):

    def __init__(self, freeze_model_params: bool = False, extractor: bool = False, **kwargs):

        d_token = kwargs.get("hidden_size")

        super(MonaiViT, self).__init__(**kwargs)

        freeze_options = {False: do_really_nothing, True: freeze_params}
        freeze_options[freeze_model_params](self)

        self.input_size = (kwargs.get("in_channels"), *kwargs.get("img_size")) if kwargs.get("in_channels") != 1 else kwargs.get("img_size")
        n_patches = kwargs.get("img_size")[0] // kwargs.get("patch_size") * kwargs.get("img_size")[1] // kwargs.get("patch_size")
        if extractor:
            self.output_size = (d_token * (n_patches + self.classification),)
        else:
            self.output_size = kwargs.get("num_classes")
            self.fc = torch.nn.Linear(d_token * (n_patches + self.classification), self.output_size)
        self.d_token = d_token
        self.extractor = extractor

    def forward(self, x):
        nan_map = torch.all(torch.isnan(x), dim=-1).to(x.device)
        while len(nan_map.shape) > 1:
            nan_map = torch.all(nan_map, dim=-1)
        x_wo_nans = torch.nan_to_num(x.clone(), nan=0.0)

        out, h_repr = super(MonaiViT, self).forward(x_wo_nans)

        h_repr = h_repr[-1]
        h_repr = self.norm(h_repr)
        if self.extractor:

            nmap = torch.ones_like(h_repr).to(h_repr.device)
            nmap[nan_map] = 0
            h_repr = nmap * h_repr

            out = h_repr.view(x.shape[0], -1)
        else:
            if hasattr(self, "fc"):
                out = self.fc(h_repr.view(x.shape[0], -1))

        return out


if __name__ == "__main__":
    pass
