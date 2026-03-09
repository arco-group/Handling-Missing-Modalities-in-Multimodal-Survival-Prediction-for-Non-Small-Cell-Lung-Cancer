import enum
import math
import torch
from torch import Tensor
from typing import List
from rtdl import CategoricalFeatureTokenizer


class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: str) -> '_TokenInitialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')

    def apply(self, x: Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            torch.nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            torch.nn.init.normal_(x, std=d_sqrt_inv)


class NAIMFeatureEmbedder(CategoricalFeatureTokenizer):
    def __init__(self, cardinalities: List[int], d_token: int, bias: bool, initialization: str, padding_idx: int):
        super(NAIMFeatureEmbedder, self).__init__(cardinalities, d_token, bias, initialization)
        initialization_ = _TokenInitialization.from_str(initialization)
        self.embeddings = torch.nn.Embedding(sum(cardinalities), d_token, padding_idx=padding_idx)
        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    def forward(self, x: Tensor, value: Tensor = None) -> Tensor:
        x = self.embeddings(x.long() + self.category_offsets[None])

        if value is not None:
            value = value.unsqueeze(0).unsqueeze(2).repeat(1, 1, x.shape[2])
            x = value * x

        if self.bias is not None:
            x = x + self.bias[None]
        return x


class NAIMEmbedder(torch.nn.Module):
    def __init__(self, input_size: int, cat_idxs: List[int], cat_dims: List[int], d_token: int, embedder_initialization: str, bias: bool):
        super(NAIMEmbedder, self).__init__()
        self.input_size = input_size
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.d_token = d_token
        self.embedder_initialization = embedder_initialization
        self.bias = bias

        j = 0
        self.embedders = torch.nn.ModuleList()
        common_params = dict(d_token=self.d_token, bias=self.bias, initialization=self.embedder_initialization, padding_idx=-1)
        for i in range(input_size):
            is_categorical_feature = i in self.cat_idxs
            if is_categorical_feature:
                feature_type_params = dict(cardinalities=[self.cat_dims[j] + 1])
                j = j + (is_categorical_feature * (i != self.cat_idxs[-1]))
            else:
                feature_type_params = dict(cardinalities=[2])
            embedding = NAIMFeatureEmbedder(**common_params, **feature_type_params)
            self.embedders.append(embedding)

            with torch.no_grad():
                self.embedders[-1].embeddings.weight[-1] = torch.zeros(self.d_token)

    def forward(self, x: Tensor) -> Tensor:
        j = 0
        embeddings = Tensor().to(x.device)
        for feature_idx in list(range(x.shape[1])):
            if feature_idx in self.cat_idxs:
                single_feature = torch.nan_to_num(x[:, feature_idx], nan=self.cat_dims[j]).to(torch.int64)
                feature_values = None
                j += 1
            else:
                single_feature = torch.isnan(x[:, feature_idx]).to(torch.int64)
                feature_values = torch.nan_to_num(x[:, feature_idx], nan=0)

            single_feature_embedding = self.embedders[feature_idx](single_feature, feature_values)

            single_feature_embedding = torch.swapaxes(single_feature_embedding, 0, 1)
            embeddings = torch.cat([embeddings, single_feature_embedding], dim=1)
        return embeddings


if __name__ == "__main__":
    pass
