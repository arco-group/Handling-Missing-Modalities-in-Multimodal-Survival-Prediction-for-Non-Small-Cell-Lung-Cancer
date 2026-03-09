import torch
import numpy as np

__all__ = ["multimodal_missing_augmentation"]


def multimodal_missing_augmentation(sample: list, missing_augmentation_probability: float = 0.3) -> list:
    """
    Randomly mask a fraction of the sample's modalities
    Parameters
    ----------
    sample : list
    missing_augmentation_probability : float

    Returns
    -------
    list
    """
    not_missing_idx = [i for i in range(len(sample)) if not torch.isnan(sample[i]).all()]
    to_mask = np.random.choice([False, True], p=[1 - missing_augmentation_probability, missing_augmentation_probability])
    if to_mask and len(not_missing_idx) > 1:
        idx_to_mask = np.random.choice(not_missing_idx, size=np.random.choice(np.arange(1, len(not_missing_idx))), replace=False)
        for i in idx_to_mask:
            sample[i] = torch.full(sample[i].shape, torch.nan).to(sample[i].device)
    return sample


if __name__ == "__main__":
    pass
