import torch
from CMC_utils.models import list_model_modules

__all__ = ["Extractor"]

class Extractor(torch.nn.Module):
    """
    Extractor class, used to extract features from a model
    """
    def __init__(self, model, *num_ftrs, **kwargs):
        super(Extractor, self).__init__()

        self.feature_extractor = model  # torch.nn.Sequential(*list(model.children()))

        self.input_size = model.input_size
        if "d_token" in model.__dict__:
            self.d_token = model.d_token

        output_size = kwargs.get("output_size", None)

        self.output_layer = output_size is not None

        if output_size:
            self.fc = torch.nn.Linear(in_features=sum(num_ftrs), out_features=output_size)
            self.output_size = (output_size,)
        else:
            self.output_size = num_ftrs

    def forward(self, inputs):
        nan_map = torch.all(torch.isnan(inputs), dim=-1).to(inputs.device)

        while len(nan_map.shape) > 1:
            nan_map = torch.all(nan_map, dim=-1)

        # for i in range(inputs.shape[0]):
        #    if torch.isnan(inputs[i]).all():
        #        outputs.append(torch.zeros((inputs[i].shape[0], self.output_size[0]), device=inputs.device))
        #        continue
        # inputs[nan_map] = torch.zeros([torch.sum(nan_map), *inputs.shape[1:]]).to(inputs.device)
        inputs_wo_nans = torch.nan_to_num(inputs.clone(), nan=0.0)

        x = self.feature_extractor(inputs_wo_nans)

        nmap = torch.ones_like(x).to(x.device)
        nmap[nan_map] = 0
        x = nmap * x
        # x[nan_map] = torch.zeros([torch.sum(nan_map), *x.shape[1:]]).to(x.device)

        if self.output_layer:
            x = self.fc(x.squeeze())

        # x = torch.cat(outputs, dim=1)

        return x


if __name__ == "__main__":
    pass
