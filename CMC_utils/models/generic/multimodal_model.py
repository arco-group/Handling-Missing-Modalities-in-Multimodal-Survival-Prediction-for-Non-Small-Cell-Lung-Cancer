import torch
from hydra.utils import instantiate

from CMC_utils.miscellaneous import recursive_cfg_substitute
import torch.nn.functional as F
import logging
log = logging.getLogger(__name__)

__all__ = ["MultimodalLearner", "MultimodalSurvivalLearner"]



class MultimodalSurvivalLearner(torch.nn.Module):
    """
    MultimodalLearner is a generic class for multimodal learning. It takes as input a list of models and outputs a
    single prediction. The models can be of any type, as long as they have a forward method that takes a single input
    and returns a single output. The output of the models is concatenated and fed to a fully connected layer.
    """
    def __init__(self, ms_models=None, shared_net=None, cs_subnet=None, num_events=1, max_time=1, projector_dim=None, **_):
        if ms_models is None or shared_net is None:
            raise ValueError("ms_models and shared_net must be provided")
        super(MultimodalSurvivalLearner, self).__init__()
        remove_mlp_head = False
        for model_id in ms_models.keys():
            if ms_models[model_id]["name"].startswith("TabNet"):
                param_dict = dict(output_dim=ms_models[model_id]["init_params"].get("input_dim", None))
            elif ms_models[model_id]["name"].startswith(("TabTransformer", "FTTransformer")):
                param_dict = dict( dim_out=ms_models[model_id]["init_params"].get("num_continuous", 0) + len(ms_models[model_id]["init_params"].get("cat_idxs", [])), extractor=True )
            elif shared_net["name"].startswith("maria"):
                param_dict = dict( extractor=True)
                remove_mlp_head = False


            else:
                param_dict = dict(extractor=True)

            ms_models[model_id] = recursive_cfg_substitute(ms_models[model_id], param_dict)

        self.ms_models = torch.nn.ModuleList()
        for model_id, model in ms_models.items():
            model['init_params']['extractor'] = True
            model['init_params']['remove_mlp_head'] = remove_mlp_head
            if projector_dim[int(model_id)] > 0:
                model['init_params']['shared_net']['init_params']['projector_dim'] = projector_dim[int(model_id)]
            self.ms_models.append(instantiate(model["init_params"], _recursive_=False))

        self.input_size = [model.input_size for model in self.ms_models]
        self.ms_output_sizes = [model.output_size[0] for model in self.ms_models]
        if shared_net["name"].startswith("TabNet"):
            param_dict = dict( cat_idxs = [], cat_dims = [], input_dim= sum(self.input_size) )
        elif shared_net["name"].startswith(("TabTransformer", "FTTransformer")):
            param_dict = dict( cat_idxs = [], categories = [], num_continuous = sum(self.ms_output_sizes), embed_input= False )
        elif shared_net["name"].startswith("naim"):
            d_token = self.ms_models[0].d_token
            param_dict = dict(embed_input= False, d_token=d_token, input_size=torch.sum(torch.tensor([s[0] for s in self.ms_output_sizes]))//d_token)
        elif shared_net["name"].startswith("maria"):
            d_token = shared_net["init_params"]['d_token']

            tokens_per_modality = list(map(lambda x: x/d_token, self.ms_output_sizes))
            tokens_per_modality = torch.tensor(tokens_per_modality, dtype=torch.int64)

            param_dict = dict(embed_input= False, d_token=d_token, input_size=torch.sum(tokens_per_modality), ntokens_per_modality=tokens_per_modality)

        elif shared_net["name"].startswith("MLP"):
            param_dict = dict(input_size= sum(self.ms_output_sizes))
        else:
            param_dict = dict(input_size= sum(self.ms_output_sizes))
        shared_net = recursive_cfg_substitute(shared_net, param_dict)
        self.shared_net = instantiate(shared_net["init_params"], _recursive_=False)

        self.input_size = self.shared_net.input_size
        self.output_size = 1

        cs_subnet["init_params"]["input_size"] = self.shared_net.output_size
        cs_subnet["init_params"]["output_size"] = self.output_size
        self.cs_subnets = instantiate(cs_subnet["init_params"])

    def forward(self, *multiple_inputs):
        hidden_representations = list()
        for inputs, model in zip(multiple_inputs, self.ms_models):
            if 'NAIM' in model.shared_net.__class__.__name__:
                modality_output = model(inputs)
                hidden_representations.append(modality_output)
                continue
            nan_mask = torch.isnan(inputs)

            # 1. Select only valid rows
            not_nan_indices = torch.nonzero(~nan_mask[:, 0], as_tuple=False).squeeze()
            inputs_clean = inputs[not_nan_indices, :]


            # 2. Forward pass only on valid samples
            modality_output = model(inputs_clean)  # shape [N_valid, D]

            if len(modality_output.shape) == 1:
                modality_output = modality_output.unsqueeze(0)


            # 3. Recompose back into [N_total, D]
            modality_output_with_mask = torch.full(
                    (nan_mask.shape[0], modality_output.shape[1]),
                    float('nan'),  # or 0.0 if you want stable grads
                    device=modality_output.device,
                    dtype=modality_output.dtype
            )

            modality_output_with_mask.index_copy_(0, not_nan_indices, modality_output)

            hidden_representations.append(modality_output_with_mask)


        B = hidden_representations[0].shape[0]
        hidden_representations = torch.cat([hidden.view(B, -1) for hidden in hidden_representations], dim=1)

        out = self.shared_net(hidden_representations, multiple_inputs)
        out = out.squeeze()
        y = self.cs_subnets(out)
        return y


class MultimodalLearner(torch.nn.Module):
    """
    MultimodalLearner is a generic class for multimodal learning. It takes as input a list of models and outputs a
    single prediction. The models can be of any type, as long as they have a forward method that takes a single input
    and returns a single output. The output of the models is concatenated and fed to a fully connected layer.
    """
    def __init__(self, ms_models=None, shared_net=None, **_):
        if ms_models is None or shared_net is None:
            raise ValueError("ms_models and shared_net must be provided")
        super(MultimodalLearner, self).__init__()

        for model_id in ms_models.keys():
            if ms_models[model_id]["name"].startswith("TabNet"):
                param_dict = dict(output_dim=ms_models[model_id]["init_params"].get("input_dim", None))
            elif ms_models[model_id]["name"].startswith(("TabTransformer", "FTTransformer")):
                param_dict = dict( dim_out=ms_models[model_id]["init_params"].get("num_continuous", 0) + len(ms_models[model_id]["init_params"].get("cat_idxs", [])), extractor=True )
            else:
                param_dict = dict(extractor=True)

            ms_models[model_id] = recursive_cfg_substitute(ms_models[model_id], param_dict)

        self.ms_models = torch.nn.ModuleList()
        for model_id, model in ms_models.items():
            self.ms_models.append(instantiate(model["init_params"], _recursive_=False))

        self.input_size = [model.input_size for model in self.ms_models]

        self.ms_output_sizes = [model.output_size for model in self.ms_models]


        if shared_net["name"].startswith("TabNet"):
            param_dict = dict( cat_idxs = [], cat_dims = [], input_dim= sum(self.input_size) )
        elif shared_net["name"].startswith(("TabTransformer", "FTTransformer")):
            param_dict = dict( cat_idxs = [], categories = [], num_continuous = sum(self.ms_output_sizes), embed_input= False )
        elif shared_net["name"].startswith("naim"):
            d_token = self.ms_models[0].d_token
            param_dict = dict(embed_input= False, d_token=d_token, input_size=torch.sum(torch.tensor([s[0] for s in self.ms_output_sizes]))//d_token)
        elif shared_net["name"].startswith("maria"):
            d_token = self.ms_models[0].d_token
            tokens_per_modality = torch.tensor([s[0] for s in self.ms_output_sizes], dtype=torch.int64)//d_token
            param_dict = dict(embed_input= False, d_token=d_token, input_size=torch.sum(tokens_per_modality), ntokens_per_modality=tokens_per_modality)
            # for i, (inp_size, out_size) in enumerate(zip(self.input_size, self.ms_output_sizes)):
                #if isinstance(inp_size, list) or isinstance(inp_size, tuple):
                #    if isinstance(out_size, list) or isinstance(out_size, tuple):
                #        param_dict["input_size"][i] = out_size[0]
                #    else:
                #        param_dict["input_size"][i] = out_size
                # param_dict["input_size"][i] = torch.sum(out_size[0])
            # ms_output_size = int(sum(self.ms_output_sizes) / d_token)
        elif shared_net["name"].startswith("MLP"):
            param_dict = dict(input_size= sum(self.ms_output_sizes))
        else:
            param_dict = dict()

        shared_net = recursive_cfg_substitute(shared_net, param_dict)
        self.shared_net = instantiate(shared_net["init_params"], _recursive_=False)

        self.output_size = self.shared_net.output_size

    def forward(self, *multiple_inputs):
        hidden_representations = list()
        for inputs, model in zip(multiple_inputs, self.ms_models):
            hidden_representations.append(model(inputs))

        B = hidden_representations[0].shape[0]
        hidden_representations = torch.cat([hidden.view(B, -1) for hidden in hidden_representations], dim=1)

        out = self.shared_net(hidden_representations, multiple_inputs)

        return out.squeeze()


if __name__ == "__main__":
    pass
