import torch
from torch.nn import functional as F
from typing import Union, Callable, Optional
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .naim_masker import NAIMTabularMasker
from .naim_embedder import NAIMEmbedder
from .naim_attention import NAIMMultiHeadAttention

__all__ = ["NAIM"]


class NAIMEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu, layer_norm_eps: float = 1e-5,
                 batch_first: bool = True, norm_first: bool = False,
                 bias: bool = False, device=None, dtype=None):

        super(NAIMEncoderLayer, self).__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                      dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps,
                                                      batch_first=batch_first, norm_first=norm_first, bias=True, device=device, dtype=dtype)
        self.self_attn = NAIMMultiHeadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, bias=True, batch_first=batch_first, device=device, dtype=dtype)


class NAIM(torch.nn.Module):

    def __init__(self,input_size,
                 output_size,
                 cat_idxs: list,
                 cat_dims: list,
                 d_token: int,
                 embedder_initialization: str,
                 bias: bool,
                 missing_value: str = "-inf",
                 num_heads: int = 12,
                 feedforward_dim: int = 1000,
                 dropout_rate: float = 0.1,
                 activation: str = "relu",
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = True,
                 norm_first: bool = False,
                 device=None,
                 dtype=None,
                 num_layers: int = 12,  # enable_nested_tensor: bool = True, mask_check: bool = True,
                 embed_vector_fun: str = "cat",
                 limit_regression_output: bool = False,
                 extractor: bool = False):
        super(NAIM, self).__init__()
        self.input_size = input_size
        self.cat_idxs = cat_idxs if cat_idxs else [-1]
        self.cat_dims = cat_dims if cat_dims else [-1]
        self.d_token = d_token
        self.embedder_initialization = embedder_initialization
        self.bias = bias
        self.missing_value = missing_value
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.num_layers = num_layers
        self.embed_vector_fun = embed_vector_fun  # "cat", "mean", "clstoken"
        self.extractor = extractor

        device_options = {"cpu": True, "cuda": torch.cuda.is_available(),
                          "mps": torch.backends.mps.is_available()}  # "mps": torch.has_mps,
        device = device_options[device] * device + (not device_options[device]) * "cpu"

        self.cls_token = None
        if self.embed_vector_fun == "clstoken":
            self.cls_token = torch.nn.Parameter(torch.randn(1, 1, self.d_token))

        self.masker = NAIMTabularMasker(missing_value=missing_value, num_heads=num_heads)

        self.embedder = NAIMEmbedder(input_size=input_size, cat_idxs=cat_idxs, cat_dims=cat_dims, d_token=d_token, embedder_initialization=embedder_initialization, bias=False)

        encoder_layer = NAIMEncoderLayer(d_model=d_token,
                                         nhead=num_heads,
                                         dim_feedforward=feedforward_dim,
                                         dropout=dropout_rate,
                                         activation=activation,
                                         layer_norm_eps=layer_norm_eps,
                                         batch_first=batch_first,
                                         norm_first=norm_first,
                                         bias=bias,
                                         device=device,
                                         dtype=dtype)

        encoder_norm = torch.nn.LayerNorm(d_token, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype)

        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=num_layers,
                                          norm=encoder_norm)  # , enable_nested_tensor=enable_nested_tensor, mask_check=mask_check)

        self.repr_options = {"mean": self.mean_representation, "clstoken": self.clstoken_representation, "cat": self.concatenated_representation}

        if self.extractor:
            if self.embed_vector_fun == "cat":
                self.output_size = ( input_size*d_token, )
            else:
                self.output_size = ( d_token, )
            self.classifier = torch.nn.Identity()
        else:
            if self.embed_vector_fun == "cat":
                self.embed_size = input_size*d_token
            else:
                self.embed_size = d_token
            self.output_size = ( output_size, )

            if self.output_size[0] > 1:
                self.classifier = torch.nn.Sequential(torch.nn.Linear(self.embed_size, self.output_size[0]))
            else:
                if limit_regression_output:
                    self.classifier = torch.nn.Sequential(torch.nn.Linear(self.embed_size, self.output_size[0]), torch.nn.Sigmoid())
                else:
                    self.classifier = torch.nn.Sequential(torch.nn.Linear(self.embed_size, self.output_size[0]))

    def forward(self, x):
        embedding = self.embedder(x)

        if self.embed_vector_fun == "clstoken":
            cls_tokens = self.cls_token.repeat(embedding.shape[0], 1, 1).to(embedding.device)
            embedding = torch.cat((cls_tokens, embedding), dim=1)
            x = torch.cat([torch.ones(x.shape[0], 1).to(x.device), x], dim=1)

        mask = self.masker.mask(x)

        embedding = self.encoder(embedding, mask)

        embedding = self.repr_options[self.embed_vector_fun](embedding, mask[:embedding.shape[0]])

        if self.extractor:
            return embedding

        logits = self.classifier(embedding)

        return logits

    @staticmethod
    def mean_representation(embedding, mask):
        emb_masks = mask[:, 0] != 0
        embedding[emb_masks] = torch.nan
        embedding = embedding.nanmean(dim=1)
        embedding = torch.nan_to_num(embedding, nan=0)
        return embedding

    @staticmethod
    def clstoken_representation(embedding, *_):
        return embedding[:, 0]

    @staticmethod
    def concatenated_representation(embedding, *_):
        return embedding.view(embedding.shape[0], -1)


if __name__ == "__main__":
    pass
