import torch
import numpy as np
from hydra.utils import instantiate
from tab_transformer_pytorch import TabTransformer as TABT, FTTransformer as FTT
from tab_transformer_pytorch.tab_transformer_pytorch import MLP

from einops import rearrange, repeat


__all__ = ["TABTransformer", "FTTransformer"]


def exists(val):
    return val is not None


def check_non_zeros(model):
    zeros_found = False
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            zeros_found = zeros_found or m.in_features == 0
    return zeros_found


class TABTransformer(TABT):
    def __init__(self, *args, cat_idxs: list = None, embed_input: bool = True, **kwargs):
        kwargs["mlp_act"] = instantiate(kwargs["mlp_act"])
        super().__init__(*args, **kwargs)
        self.cat_idxs = cat_idxs
        self.input_size = len(cat_idxs) + kwargs["num_continuous"]
        self.output_size = kwargs["dim_out"]
        self.embed_input = embed_input

        null_layer_found = check_non_zeros(self.mlp)
        if null_layer_found:
            self.mlp = MLP([self.input_size, *[self.input_size]*len(kwargs["mlp_hidden_mults"]), self.output_size], act = kwargs["mlp_act"])

    def old_forward(self, inputs, *_):
        with torch.no_grad():
            n_features = inputs.shape[-1]
            cat_features_map = torch.BoolTensor(np.array(list(map(lambda x: x in self.cat_idxs, range(n_features))), dtype=bool) ).to(inputs.device)
            x_categ = inputs[:, cat_features_map].type(torch.LongTensor).to(inputs.device)
            x_cont = inputs[:, ~cat_features_map].to(inputs.device)
        return super().forward(x_categ, x_cont)

    def forward(self, inputs, *_, return_attn = False):
        if self.embed_input:
            n_features = inputs.shape[-1]
            cat_features_map = torch.BoolTensor(np.array(list(map(lambda inp: inp in self.cat_idxs, range(n_features))), dtype=bool)).to(inputs.device)
            x_categ = inputs[:, cat_features_map].type(torch.LongTensor).to(inputs.device)
            x_cont = inputs[:, ~cat_features_map].to(inputs.device)

            xs = []

            assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

            attns = None

            if self.num_unique_categories > 0:
                x_categ += self.categories_offset

                x, attns = self.transformer(x_categ, return_attn=True)

                flat_categ = x.flatten(1)
                xs.append(flat_categ)

            assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

            if self.num_continuous > 0:
                if exists(self.continuous_mean_std):
                    mean, std = self.continuous_mean_std.unbind(dim=-1)
                    x_cont = (x_cont - mean) / std

                normed_cont = self.norm(x_cont)
                xs.append(normed_cont)

            x = torch.cat(xs, dim=-1)
        else:
            x = inputs
        logits = self.mlp(x)

        if not return_attn:
            return logits

        return logits, attns


class FTTransformer(FTT):
    def __init__(self, *args, dim: int, cat_idxs: list = None, embed_input: bool = True, extractor: bool = False, **kwargs):
        super().__init__(*args, dim=dim, **kwargs)
        self.cat_idxs = cat_idxs
        self.input_size = len(cat_idxs) + kwargs["num_continuous"]
        self.output_size = kwargs["dim_out"]
        self.embed_input = embed_input
        self.d_token = dim
        self.extractor = extractor

    def old_forward(self, inputs, *_):
        with torch.no_grad():
            n_features = inputs.shape[-1]
            cat_features_map = torch.BoolTensor(np.array(list(map(lambda x: x in self.cat_idxs, range(n_features))), dtype=bool)).to(inputs.device)
            x_categ = inputs[:, cat_features_map].type(torch.LongTensor).to(inputs.device)
            x_numer = inputs[:, ~cat_features_map].to(inputs.device)
        return super().forward(x_categ, x_numer)

    def forward(self, inputs, *_, return_attn = False):
        if self.embed_input:

            n_features = inputs.shape[-1]
            cat_features_map = torch.BoolTensor(np.array(list(map(lambda inp: inp in self.cat_idxs, range(n_features))), dtype=bool)).to(inputs.device)
            x_categ = inputs[:, cat_features_map].type(torch.LongTensor).to(inputs.device)
            x_numer = inputs[:, ~cat_features_map].to(inputs.device)

            assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

            xs = []
            if self.num_unique_categories > 0:
                x_categ = x_categ + self.categories_offset

                x_categ = self.categorical_embeds(x_categ)

                xs.append(x_categ)

            # add numerically embedded tokens
            if self.num_continuous > 0:
                x_numer = self.numerical_embedder(x_numer)

                xs.append(x_numer)

            # concat categorical and numerical

            x = torch.cat(xs, dim = 1)
        else:
            x = inputs.view(inputs.shape[0], -1, self.d_token)
        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # attend

        x, attns = self.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))
        if self.extractor:
            return x

        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns


if __name__ == "__main__":
    pass
