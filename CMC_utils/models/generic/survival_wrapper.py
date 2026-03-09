import torch
from typing import Union
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import instantiate

__all__ = ["SurvivalWrapper", "SurvivalWrapperRegression"]


class SurvivalWrapperRegression(torch.nn.Module):
    """
    Wrapper for survival models. It takes a shared network and a list of subnetworks, one for each event.
    The shared network is used to extract features from the input data, and the subnetworks are used to predict the
    probability of each event at each time step.
    """

    def __init__(self, num_events: int, max_time: int, shared_net: Union[dict, DictConfig], cs_subnet: Union[dict, DictConfig], extractor: bool = False, **kwargs):
        super(SurvivalWrapperRegression, self).__init__()

        self.shared_net = instantiate(shared_net["init_params"])

        self.input_size = self.shared_net.output_size
        self.output_size = 1

        self.num_events = num_events
        self.max_time = max_time

        self.extractor = extractor


        cs_subnet["init_params"]["input_size"] = self.input_size
        cs_subnet["init_params"]["output_size"] = self.output_size
        remove_mlp_head = kwargs.get("remove_mlp_head", False)

        self.cs_subnet = instantiate(cs_subnet["init_params"])
        if extractor and not remove_mlp_head:

            self.output_size = self.cs_subnet.out.in_features,
            self.cs_subnet.out = torch.nn.Identity()
            cs_subnet["init_params"]["output_size"] = self.output_size
        elif extractor and remove_mlp_head:

            self.cs_subnet = torch.nn.Identity()
            self.output_size = self.shared_net.output_size
            cs_subnet["init_params"]["output_size"] = self.output_size


    def forward(self, inputs):

        x = self.shared_net(inputs)
        x = x.view(x.size(0), -1)

        y = self.cs_subnet(x)
        return y


class SurvivalWrapper(torch.nn.Module):
    """
    Wrapper for survival models. It takes a shared network and a list of subnetworks, one for each event.
    The shared network is used to extract features from the input data, and the subnetworks are used to predict the
    probability of each event at each time step.
    """

    def __init__(self, num_events: int, max_time: int, shared_net: Union[dict, DictConfig], cs_subnet: Union[dict, DictConfig], extractor: bool = False, **kwargs):
        super(SurvivalWrapper, self).__init__()

        self.shared_net = instantiate(shared_net["init_params"])

        self.input_size = self.shared_net.input_size
        self.output_size = self.shared_net.output_size

        self.num_events = num_events
        self.max_time = max_time

        self.extractor = extractor
        if not extractor:

            self.output_size = num_events * max_time
            cs_subnet["init_params"]["input_size"] = self.shared_net.output_size
            cs_subnet["init_params"]["output_size"] = self.output_size
            self.cs_subnets = torch.nn.ModuleList()
            for k in range(num_events):
                subnet = instantiate(cs_subnet["init_params"])
                self.cs_subnets.append(subnet)

    def forward(self, inputs):

        x = self.shared_net(inputs)
        x = x.view(x.size(0), -1)

        if not self.extractor:
            y = []

            for subnet in self.cs_subnets:
                x_CS = subnet(x)
                y_CS = F.softmax(x_CS, dim=-1)
                y.append(y_CS)
            y = torch.cat(y, dim=-1)
        else:
            y = x
        return y


if __name__ == "__main__":
    pass
