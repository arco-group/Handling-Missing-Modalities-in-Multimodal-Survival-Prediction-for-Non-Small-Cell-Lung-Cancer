import torch
import numpy as np

__all__ = ["missing_augmentation"]

def missing_augmentation(sample: torch.tensor, missing_augmentation_probability: float = 0.5) -> torch.tensor:
    """
    Randomly mask a fraction of the sample
    Parameters
    ----------
    sample : np.array
    missing_augmentation_probability : float

    Returns
    -------
    np.array
    """
    not_missing_idx = torch.where( ~torch.isnan(sample) )[0]
    to_mask = np.random.choice([False, True], p=[1-missing_augmentation_probability, missing_augmentation_probability])
    if to_mask and len(not_missing_idx) > 2:
        idx_to_mask = np.random.choice( not_missing_idx, size=np.random.choice( np.arange(1, len(not_missing_idx ) -1) ), replace=False)
        sample[ idx_to_mask ] = torch.nan
    return sample


if __name__ == "__main__":
    pass
