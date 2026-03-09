import math
import torch
from typing import List

from pytorch_tabular.models.common.layers import ODST
# from skimage.filters.rank import modal
from torch import Tensor, nn
from torch.nn import Sigmoid
import torch.nn.functional as F
from .tabular_tokenizer import CategoricalFeatureTokenizer
from typing import Tuple, Optional

from CMC_utils.miscellaneous import do_really_nothing
from CMC_utils.models import Extractor, set_pretrained_attribute, freeze_params, remove_DataParallel_module

__all__ = ["NAIM", "MARIA", "get_naim_pretrained", "Embedder", "ConcatModelODST", "CrossAttentionMissingModality"]


def nanmean_pool1d(x, kernel_size: int, stride: int = None):
    """
    x: (batch, seq_len, hidden_dim)
    Pools along seq_len with NaN-aware averaging.
    """
    if stride is None:
        stride = kernel_size

    B, L, D = x.shape
    # reshape into groups
    new_len = L // stride
    x = x[:, :new_len * stride, :]  # trim excess if not divisible
    x = x.reshape(B, new_len, stride, D)  # (B, new_len, stride, D)

    # mask NaNs
    mask = ~torch.isnan(x)  # True if valid
    x = torch.where(mask, x, torch.tensor(0., device=x.device, dtype=x.dtype))

    # sum valid and divide by count
    sums = x.sum(dim=2)
    counts = mask.sum(dim=2).clamp(min=1)
    pooled = sums / counts
    return pooled


class TabularMasker:
    def __init__(self, missing_value: str = "-inf"):
        missing_value_options = {"-inf": -torch.inf, "~inf": -1e9}
        self.missing_value = missing_value_options[missing_value]

    def _tabular_sample_mask(self, sample: Tensor):
        mask = torch.clone(sample)
        mask[~torch.isnan(sample)] = 0
        mask[torch.isnan(sample)] = 1
        return mask

    def mask(self, data: Tensor, **_):
        masks = Tensor().to(data.device)

        for sample in data:
            sample_mask = self._tabular_sample_mask(sample).to(torch.bool)

            sample_mask = sample_mask.repeat(sample_mask.shape[0], 1)

            masks = torch.cat([masks, sample_mask.unsqueeze(dim=0)], dim=0)

        masks = torch.masked_fill(masks, masks.to(torch.bool), self.missing_value)

        return masks  # , masks.transpose(-2, -1)

    def modality_mask(self, embeddings: Tensor, data: Tensor, use_cls_token: bool = False, **_):
        masks = Tensor().to(embeddings.device)
        n_modalities = len(data) + int(use_cls_token)
        for sample_idx, embedding in enumerate(embeddings):

            if use_cls_token:
                sample_mask = torch.zeros(1, 1, n_modalities).to(embeddings.device)
            else:
                sample_mask = torch.Tensor().to(embeddings.device)

            for modality_data in data:
                sample = modality_data[sample_idx]
                if torch.isnan(sample).all():
                    ith_modality_mask = torch.ones(1, 1, n_modalities).to(embeddings.device)
                else:
                    ith_modality_mask = torch.zeros(1, 1, n_modalities).to(embeddings.device)
                sample_mask = torch.cat([sample_mask, ith_modality_mask], dim=1)
            masks = torch.cat([masks, sample_mask], dim=0)

        masks = torch.masked_fill(masks, masks.to(torch.bool), self.missing_value)

        return masks, masks.transpose(-2, -1)


class Attention(torch.nn.Module):

    def __init__(self, dropout_rate: float = 0.0):
        super(Attention, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, q: Tensor, k: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:

        B, Nt, E = q.shape
        q = q / math.sqrt(E)

        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
        else:
            attn = torch.bmm(q, k.transpose(-2, -1))

        attn = F.softmax(attn, dim=-1)

        attn = torch.add(attn, attn_mask.transpose(-2, -1))
        attn = F.relu(attn)

        if self.dropout_rate > 0.0:
            attn = F.dropout(attn, p=self.dropout_rate)

        return attn


class MultiHeadAttention(torch.nn.Module):

    def __init__(
            self,
            input_size: int,
            num_heads: int,
            bias: bool = True,
            activation: str = "relu",
            dropout_rate: float = 0.0
            ):

        super(MultiHeadAttention, self).__init__()
        assert input_size % num_heads == 0, f"`input_size`({input_size}) should be divisible by `num_heads`({num_heads})"

        self.input_size = input_size
        self.num_heads = num_heads
        self.bias = bias
        activation_options = dict(relu=F.relu, gelu=F.gelu, silu=F.silu, none=None)
        self.activation = activation_options[activation]
        self.dropout_rate = dropout_rate

        self.linear_q = torch.nn.Linear(input_size, input_size, bias)
        self.linear_k = torch.nn.Linear(input_size, input_size, bias)
        self.linear_v = torch.nn.Linear(input_size, input_size, bias)
        self.linear_o = torch.nn.Linear(input_size, input_size, bias)
        self.attn = Attention(dropout_rate)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            #v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        if mask is not None:
            mask = torch.repeat_interleave(mask, self.num_heads, 0)

        y, attn_scores = self._scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y, attn_scores

    def _reshape_to_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def _scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        attn = self.attn(q, k, attn_mask)
        """B, Nt, E = q.shape
        q = q / math.sqrt(E)

        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
        else:
            attn = torch.bmm(q, k.transpose(-2, -1))

        attn = F.softmax(attn, dim=-1)

        if attn_mask_2 is not None:
            attn = torch.add(attn, attn_mask_2)
            attn = F.relu(attn)

        if self.dropout_rate > 0.0:
            attn = F.dropout(attn, p=self.dropout_rate)"""

        output = torch.bmm(attn, v)

        return output, attn


class EncoderBlock(torch.nn.Module):
    """
    Encoder block of the Transformer.
    """

    def __init__(self, emb_dim, ff_dim, num_heads, bias: bool = False, activation: str = "relu", dropout_rate: float = 0.0):
        super(EncoderBlock, self).__init__()

        # Norm layers
        self.norm1 = torch.nn.RMSNorm(emb_dim)
        self.norm2 = torch.nn.RMSNorm(emb_dim)

        # Multi-head attention
        self.attn = MultiHeadAttention(
                input_size=emb_dim,
                num_heads=num_heads,
                bias=bias,
                activation=activation,
                dropout_rate=dropout_rate,
        )

        # Feed-forward block
        activation_options = dict(relu=torch.nn.ReLU, gelu=torch.nn.GELU, silu=torch.nn.SiLU)
        self.ff = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, ff_dim, bias=bias),
                activation_options[activation](),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(ff_dim, emb_dim, bias=bias),
        )

    def forward(self, x: Tensor, mask: Tensor = None):
        # Pre-Norm Attention
        normed_x = self.norm1(x)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x, mask=mask)
        x = x + attn_out  # residual connection

        # Pre-Norm Feedforward
        normed_x = self.norm2(x)
        ff_out = self.ff(normed_x)
        x = x + ff_out  # residual connection
        return x


class NAIM(torch.nn.Module):
    """
    NAIM model for tabular data.
    """

    def __init__(
            self,
            input_size,
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
            num_layers: int = 12,
            # embed_input: bool = True,
            embed_vector_fun: str = "cat",
            projector_dim: int = None,
            limit_regression_output: bool = False,
            extractor: bool = False
            ):

        super(NAIM, self).__init__()

        if projector_dim is not None:

            input_data_size = input_size
            input_size = projector_dim
            self.input_size = input_size
        else:
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
        # self.embed_input = embed_input
        self.embed_vector_fun = embed_vector_fun  # "cat", "mean", "clstoken"
        self.projector_dim = projector_dim
        self.limit_regression_output = limit_regression_output
        self.extractor = extractor

        self.cls_token = None

        if (self.embed_vector_fun in ("clstoken", "clstoken+mean")) or ("cls+pooling" in self.embed_vector_fun):
            self.cls_token = torch.nn.Parameter(torch.randn(1, 1, self.d_token))

        if self.extractor:
            if self.embed_vector_fun in ("clstoken", "mean"):
                self.output_size = (self.d_token,)
            elif self.embed_vector_fun == "clstoken+mean":
                self.output_size = (2 * self.d_token,)
            elif "cls+pooling" in self.embed_vector_fun:
                self.pooling_size = int(self.embed_vector_fun.split(":")[-1])
                n_tokens = math.floor(input_size / self.pooling_size)
                self.output_size = ((n_tokens + 1) * self.d_token,)
            else:
                self.output_size = (input_size * d_token,)
        else:
            if self.embed_vector_fun in ("clstoken", "mean"):
                self.embed_size = self.d_token
            elif self.embed_vector_fun == "clstoken+mean":
                self.embed_size = 2 * self.d_token
            elif "cls+pooling" in self.embed_vector_fun:
                self.pooling_size = int(self.embed_vector_fun.split(":")[-1])
                n_tokens = math.floor(input_size / self.pooling_size)
                self.embed_size = (n_tokens + 1) * self.d_token
            else:
                self.embed_size = input_size * d_token
            self.output_size = (output_size,)

        # EMBEDDERS initializations
        #if self.embed_input:
        j = 0
        self.embeddings = torch.nn.ModuleList()
        common_params = dict(d_token=self.d_token, bias=self.bias, initialization=self.embedder_initialization)
        for i in range(input_size):
            is_categorical_feature = i in self.cat_idxs
            feature_type_params = {True : dict(cardinalities=[self.cat_dims[j] + 1], padding_idx=self.cat_dims[j]),
                                   False: dict(cardinalities=[2], padding_idx=1)}

            j = j + (is_categorical_feature * (i != self.cat_idxs[-1]))
            embedding = CategoricalFeatureTokenizer(**common_params, **feature_type_params[is_categorical_feature])

            self.embeddings.append(embedding)
        #else:
        #    self.embeddings = torch.nn.Identity()

        if self.projector_dim is not None:
            self.projector = torch.nn.Linear(input_data_size, self.projector_dim)
        # MASKER initialization
        self.attention_mask = TabularMasker(self.missing_value)

        self.dropout = torch.nn.Dropout(self.dropout_rate)

        self.encoder = torch.nn.ModuleList([EncoderBlock(self.d_token, self.feedforward_dim, self.num_heads, bias=self.bias, activation=self.activation, dropout_rate=self.dropout_rate) for _ in range(self.num_layers)])

        self.norm = torch.nn.RMSNorm(self.d_token)

        # classifier
        if not self.extractor:
            if self.output_size[0] > 1:
                self.classifier = torch.nn.Sequential(torch.nn.Linear(self.embed_size, self.output_size[0]))
            else:
                if limit_regression_output:
                    self.classifier = torch.nn.Sequential(torch.nn.Linear(self.embed_size, self.output_size[0]), Sigmoid())
                else:
                    self.classifier = torch.nn.Sequential(torch.nn.Linear(self.embed_size, self.output_size[0]))

    def feature_embedding(self, x: Tensor):
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
            single_feature_embedding = self.embeddings[feature_idx](single_feature, feature_values)

            single_feature_embedding = torch.swapaxes(single_feature_embedding, 0, 1)
            embeddings = torch.cat([embeddings, single_feature_embedding], dim=1)
        return embeddings

    def forward(self, x, _=None):

        if self.projector_dim is not None:
            x = self.projector(x)
        embeddings = self.feature_embedding(x)

        if (self.embed_vector_fun in ("clstoken", "clstoken+mean")) or ("cls+pooling" in self.embed_vector_fun):
            cls_tokens = self.cls_token.repeat(embeddings.shape[0], 1, 1).to(embeddings.device)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

            x = torch.cat([torch.ones(x.shape[0], 1).to(x.device), x], dim=1)

        masks = self.attention_mask.mask(x)

        # transformer
        for encoder_layer in self.encoder:
            embeddings = encoder_layer(embeddings, mask=masks)

        embeddings = self.norm(embeddings)

        if self.embed_vector_fun == "mean":
            emb_masks = masks[:, 0] != 0
            embeddings[emb_masks] = torch.nan
            embeddings = embeddings.nanmean(dim=1)
            embeddings = torch.nan_to_num(embeddings, nan=0)
        elif self.embed_vector_fun == "clstoken":
            embeddings = embeddings[:, 0]
        elif self.embed_vector_fun == "clstoken+mean":
            cls_embeddings = embeddings[:, 0]
            emb_masks = masks[:, 0] != 0
            embeddings[emb_masks] = torch.nan
            mean_embeddings = embeddings.nanmean(dim=1)
            mean_embeddings = torch.nan_to_num(mean_embeddings, nan=0)
            embeddings = torch.cat([cls_embeddings, mean_embeddings], dim=1)
        elif "cls+pooling" in self.embed_vector_fun:
            pooling_size = int(self.embed_vector_fun.split(":")[-1])
            cls_tokens = embeddings[:, 0].unsqueeze(1)
            other_tokens = embeddings[:, 1:, :]
            pooled_tokens = nanmean_pool1d(other_tokens, kernel_size=pooling_size)
            embeddings = torch.cat([cls_tokens, pooled_tokens], dim=1)
            embeddings = embeddings.view(embeddings.shape[0], -1)
        else:
            embeddings = embeddings.view(embeddings.shape[0], -1)

        if self.extractor:
            return embeddings

        # classifier
        logits = self.classifier(embeddings)

        return logits


class Embedder(torch.nn.Module):
    """
    NAIM model for tabular data.
    """

    def __init__(
            self,
            input_size,
            output_size,
            cat_idxs: list,
            cat_dims: list,
            d_token: int,
            embedder_initialization: str,
            bias: bool,
            **kwargs):

        super(Embedder, self).__init__()
        self.input_size = input_size
        self.output_size = (input_size * d_token,)
        self.cat_idxs = cat_idxs if cat_idxs else [-1]
        self.cat_dims = cat_dims if cat_dims else [-1]
        self.d_token = d_token
        self.embedder_initialization = embedder_initialization
        self.bias = bias

        # EMBEDDERS initializations
        #if self.embed_input:
        j = 0
        self.embeddings = torch.nn.ModuleList()
        common_params = dict(d_token=self.d_token, bias=self.bias, initialization=self.embedder_initialization)
        for i in range(input_size):
            is_categorical_feature = i in self.cat_idxs
            feature_type_params = {True : dict(cardinalities=[self.cat_dims[j] + 1], padding_idx=self.cat_dims[j]),
                                   False: dict(cardinalities=[2], padding_idx=1)}

            j = j + (is_categorical_feature * (i != self.cat_idxs[-1]))
            embedding = CategoricalFeatureTokenizer(**common_params, **feature_type_params[is_categorical_feature])

            self.embeddings.append(embedding)


    def feature_embedding(self, x: Tensor):
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
            single_feature_embedding = self.embeddings[feature_idx](single_feature, feature_values)

            single_feature_embedding = torch.swapaxes(single_feature_embedding, 0, 1)
            embeddings = torch.cat([embeddings, single_feature_embedding], dim=1)
        return embeddings

    def forward(self, x, _=None):
        embeddings = self.feature_embedding(x)
        return embeddings


class ConcatModelODST(torch.nn.Module):
    def __init__(self, dropout: int= 0.0, odst: bool=False,
                 input_size: int= None, feedforward_dim: int= None, **kwargs
                 ):
        super(ConcatModelODST, self).__init__()
        # self.modality_tokens = torch.nn.Embedding(self.n_modalities + 1, self.d_token, padding_idx=-1)
        self.input_size = input_size

        self.mlp = torch.nn.Sequential(
            ODST(input_size, feedforward_dim) if odst else torch.nn.Linear(input_size, feedforward_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
        )
        self.output_size = feedforward_dim


    def forward(self, x, _=None):
        batch_size = x.shape[0]
        embeddings = x.view(batch_size, -1)
        embeddings = self.mlp(embeddings)
        return embeddings



class CrossAttentionMissingModality(nn.Module):
    def __init__(
        self,
        input_size: int,
        feedforward_dim: int,
        n_heads: int = 4,
        dropout: float = 0.2,
        features_per_modality: int = 400,
        **kwargs
    ):
        """
        Cross-attention multimodal fusion module that skips missing modalities (all-NaN feature blocks).

        Args:
            input_size (int): Total input dimensionality (e.g., 1200 = 3 modalities * 400).
            feedforward_dim (int): Dimensionality of the fused output and attention hidden space.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            features_per_modality (int): Feature dimensionality per modality.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = feedforward_dim
        self.n_modalities = self.input_size // features_per_modality
        self.hidden_dim = feedforward_dim
        self.features_per_modality = features_per_modality
        self.masking = kwargs.get("masking", False)

        # Linear projection from modality features → hidden space
        self.input_proj = nn.Linear(self.features_per_modality, self.hidden_dim)

        # Query token (shared modality embedding)
        self.modality_embeddings = nn.Parameter(torch.randn(1, self.hidden_dim))

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Normalization and output projection
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, feedforward_dim)
    def _build_mask_from_original_x(self, original_x):
        """
        Builds a (B, n_modalities) boolean mask.
        True = modality missing for that sample.
        """
        masks = []
        for modality_tensor in original_x:
            # (B, Fm)
            missing = torch.isnan(modality_tensor).all(dim=1) | (modality_tensor.abs().sum(dim=1) == 0)
            masks.append(missing)
        mask = torch.stack(masks, dim=1)  # (B, n_modalities)
        return mask
    def forward(self, x: torch.Tensor, original_x=None) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, input_size), possibly with NaNs.
            original_x (Tensor, optional): Original tensor before preprocessing, used for mask inference.

        Returns:
            Tensor of shape (B, feedforward_dim): fused multimodal embedding.
        """
        B = x.size(0)

        # Reshape into modality blocks
        x = x.view(B, self.n_modalities, self.features_per_modality)

        # Detect missing modalities (all-NaN or all-zero feature blocks)
        if self.masking:
            missing_mask = self._build_mask_from_original_x(original_x)
        else:
            missing_mask = torch.zeros(B, self.n_modalities, dtype=torch.bool, device=x.device)

        # Project features into attention space
        x_proj = self.input_proj(x)  # (B, n_modalities, hidden_dim)

        # Prepare query token
        if self.masking:
            query = self.modality_embeddings.expand(B, 1, -1)  # (B, 1, hidden_dim)
        else:
            query = self.modality_embeddings.expand(B, 1, -1)  # (B, 1, hidden_dim)

        # Cross-attention (mask tells which K/V tokens to ignore)
        attn_output, _ = self.cross_attn(
            query=query,
            key=x_proj,
            value=x_proj,
            key_padding_mask=missing_mask  # shape (B, n_modalities)
        )

        # Post-attention normalization + projection
        fused = self.norm(attn_output)
        fused = self.output_layer(fused.squeeze(1))  # (B, feedforward_dim)

        return fused
#

# class CrossAttentionMissingModality(nn.Module):
#     def __init__(self, input_size: int, feedforward_dim: int, n_heads: int = 4, dropout: float = 0.2, features_per_modality: int = 400, **kwargs):
#         """
#         Cross-attention multimodal fusion module that skips missing modalities (all-NaN feature blocks).
#
#         Args:
#             input_size (int): Total input dimensionality (e.g. 1200 = 3 modalities * 400).
#             hidden_dim (int): Dimensionality of the trainable modality embeddings and attention hidden space.
#             n_modalities (int): Number of modalities.
#             n_heads (int): Number of attention heads.
#             dropout (float): Dropout probability.
#         """
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = feedforward_dim
#         self.n_modalities = self.input_size // features_per_modality
#         self.hidden_dim = feedforward_dim
#         self.features_per_modality = features_per_modality
#
#
#         # Trainable modality embeddings (query tokens)
#         self.modality_embeddings = nn.Parameter(torch.randn(1, self.hidden_dim))
#
#         # Linear projection for inputs before attention
#
#         # Multi-head cross-attention
#         self.cross_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
#
#         self.input_proj = nn.Linear(self.features_per_modality, self.hidden_dim)
#
#         # Post-attention normalization and feed-forward
#         self.norm = nn.LayerNorm(self.hidden_dim)
#
#         # output layer
#         self.output_layer = nn.Linear(self.hidden_dim, feedforward_dim)
#
#
#     def forward(self, x: torch.Tensor, _) -> torch.Tensor:
#         """
#         Args:
#             x (Tensor): Input tensor of shape (batch_size, input_size),
#                         containing possibly NaNs for missing modalities.
#
#         Returns:
#             Tensor of shape (batch_size, hidden_dim): fused multimodal embedding.
#         """
#         B = x.shape[0]
#
#         # Split the input into modality-specific feature blocks
#         x = x.view(B, self.n_modalities, self.features_per_modality)
#
#         # Project features to hidden_dim
#         x_proj = self.input_proj(x)  # (B, n_modalities, hidden_dim)
#
#         # Prepare queries (modality embeddings)
#         queries = self.modality_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, n_modalities, hidden_dim)
#
#         # Cross attention: query=modality embeddings, key/value=projected features
#         attn_output, _ = self.cross_attn(query=queries, key=x_proj, value=x_proj)
#
#
#         # Normalization + feed-forward refinement
#         fused = self.norm(attn_output)
#
#         return self.output_layer(fused.squeeze(1))
#
#
#
#


class MARIA(NAIM):
    def __init__(self, *args, n_modalities: int, ntokens_per_modality: List[int], **kwargs):  # ntokens_per_modality: List[int],
        super(MARIA, self).__init__(*args, **kwargs)
        self.n_modalities = n_modalities
        self.ntokens_per_modality = ntokens_per_modality

        self.masking_options = {False: self.attention_mask.modality_mask, True: self.attention_mask.mask}
        # self.modality_tokens = torch.nn.Embedding(self.n_modalities + 1, self.d_token, padding_idx=-1)
        self.modality_tokens = CategoricalFeatureTokenizer(cardinalities=[self.n_modalities + 1], d_token=self.d_token, bias=self.bias, initialization=self.embedder_initialization, padding_idx=-1)

        # if self.extractor and 'linear' in self.embed_vector_fun:
        #     self.down_trasform = torch.nn.Linear(self.output_size[0], kwargs['feedforward_dim'])
        #     self.output_size = self.down_trasform.out_features


    def forward(self, x, original_x=None):
        batch_size = x.shape[0]
        embeddings = x.view(batch_size, -1, self.d_token)
        modalities_matrix = []
        for i in range(self.n_modalities):
            modalities_matrix.append(self.modality_tokens(torch.tensor([i] * self.ntokens_per_modality[i], dtype=torch.int64).to(embeddings.device)))
        modalities_matrix = torch.cat(modalities_matrix, dim=1).repeat(batch_size, 1, 1)

        embeddings_by_modality = torch.split(embeddings[:, :, 0], tuple(self.ntokens_per_modality), dim=1)

        embeddings = torch.nan_to_num(embeddings, nan=1)
        embeddings = modalities_matrix * embeddings

        # elif self.embed_vector_fun == "cat":
        original_tensor = []
        for j, d in enumerate(embeddings_by_modality):

            if len(d.shape) != 2:
                tnsr = torch.empty((batch_size, self.ntokens_per_modality[j]))
                for i, s in enumerate(d):
                    if torch.isnan(s).all():
                        tnsr[i, :] = torch.nan
                    else:
                        tnsr[i, :] = 0
                original_tensor.append(tnsr.to(d.device))
            else:
                if d.shape[1] > self.ntokens_per_modality[j]:
                    d = d[:, :self.ntokens_per_modality[j]]
                original_tensor.append(d)
        # original_x = torch.cat([d if len(d.shape) == 2 else torch.zeros((batch_size, 1)) for d in original_x], dim=1)
        original_tensor = torch.cat(original_tensor, dim=1)

        # use_cls_token = self.embed_vector_fun == "clstoken"
        if "clstoken" in self.embed_vector_fun or 'cls+pooling' in self.embed_vector_fun:
            cls_tokens = self.cls_token.repeat(embeddings.shape[0], 1, 1).to(embeddings.device)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)
            original_tensor = torch.cat([torch.ones(x.shape[0], 1).to(x.device), original_tensor], dim=1)

        # masks = self.masking_options[use_cls_token](embeddings=embeddings, data=original_x, use_cls_token=use_cls_token)
        masks = self.attention_mask.mask(data=original_tensor)
        # transfor
        for encoder_layer in self.encoder:
            embeddings = encoder_layer(embeddings, mask=masks)

        #embeddings = self.norm(embeddings)

        if self.embed_vector_fun == "mean":
            emb_masks = masks[:, 0] != 0
            embeddings[emb_masks] = torch.nan
            embeddings = embeddings.nanmean(dim=1)
            embeddings = torch.nan_to_num(embeddings, nan=0)
        elif self.embed_vector_fun == "clstoken":
            embeddings = embeddings[:, 0]
        elif self.embed_vector_fun == "clstoken+mean":
            cls_embeddings = embeddings[:, 0]
            emb_masks = masks[:, 0] != 0
            embeddings[emb_masks] = torch.nan
            mean_embeddings = embeddings.nanmean(dim=1)
            mean_embeddings = torch.nan_to_num(mean_embeddings, nan=0)
            embeddings = torch.cat([cls_embeddings, mean_embeddings], dim=1)
        elif "cls+pooling" in self.embed_vector_fun:

            pooling_size = int(self.embed_vector_fun.split(":")[-1])
            cls_tokens = embeddings[:, 0].unsqueeze(1)
            other_tokens = embeddings[:, 1:, :]
            pooled_tokens = nanmean_pool1d(other_tokens, kernel_size=pooling_size)
            embeddings = torch.cat([cls_tokens, pooled_tokens], dim=1)
            embeddings = embeddings.view(embeddings.shape[0], -1)
            #print('Embeddings shape after pooling:', embeddings.shape)
        else:
            embeddings = embeddings.view(embeddings.shape[0], -1)
        # if self.extractor and 'linear' in self.embed_vector_fun:
        #     # num_ftrs = embeddings.shape[1]
        #     embeddings = self.down_trasform(embeddings)

        if self.extractor:
            return embeddings

        # classifier
        logits = self.classifier(embeddings)

        return logits


def get_naim_pretrained(checkpoint_path, output_size, load_weights: bool = True, freeze_model_params: bool = False, extractor: bool = False, **kwargs):
    model = NAIM(output_size=output_size, extractor=False, **kwargs)
    model_state_dict = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = remove_DataParallel_module(model_state_dict)
    model.load_state_dict(model_state_dict)

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if output_size and not extractor:
        num_ftrs = model.classifier[0].in_features
        model.classifier[0] = torch.nn.Linear(in_features=num_ftrs, out_features=output_size)

    if extractor:
        num_ftrs = model.classifier[0].in_features
        model.classifier = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


if __name__ == "__main__":
    pass
