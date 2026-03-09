import torch
from torch import Tensor

__all__ = ["NAIMTabularMasker"]


class NAIMTabularMasker:
    def __init__(self, missing_value: str = "-inf"):
        missing_value_options = {"-inf": -torch.inf, "~inf": -1e9}
        self.missing_value = missing_value_options[missing_value]

    @staticmethod
    def _tabular_sample_mask(sample: Tensor):
        mask = torch.clone(sample)
        mask[~torch.isnan(sample)] = 0
        mask[torch.isnan(sample)] = 1
        return mask

    def mask(self, data: Tensor):
        masks = Tensor().to(data.device)

        for sample in data:
            sample_mask = self._tabular_sample_mask(sample).to(torch.bool)

            sample_mask = sample_mask.repeat(sample_mask.shape[0], 1)

            masks = torch.cat([masks, sample_mask.unsqueeze(dim=0)], dim=0)

        masks = torch.masked_fill(masks, masks.to(torch.bool), self.missing_value)

        return masks

    def modality_mask(self, embeddings: Tensor, data: Tensor, use_cls_token: bool = False):
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

        return masks


if __name__ == "__main__":
    pass
