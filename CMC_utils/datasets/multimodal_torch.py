import torch
import numpy as np
from torch.utils.data import Dataset

__all__ = ["MultimodalJointFusionTabularDatasetTorch"]  # , "MultimodalEarlyFusionTabularDatasetTorch"]

from CMC_utils.data_augmentation import multimodal_missing_augmentation


class MultimodalJointFusionTabularDatasetTorch(Dataset):
    """
    Class for creating a dataset from multiple datasets. The datasets must have the same length.
    """
    def __init__(self, *args, augmentation: bool = False, missing_augmentation_probability: float = 0.5):
        args = self.drop_samples_with_all_modalities_missing(args)
        self.datasets = args

        self.input_sizes = [dataset.input_size for dataset in self.datasets]
        self.input_size = self.input_sizes  # sum(self.input_sizes)
        self.output_sizes = [dataset.output_size for dataset in self.datasets]
        self.output_size = self.output_sizes[0]
        self.tot_columns = [dataset.columns for dataset in self.datasets]
        self.__columns = [col for columns in self.tot_columns for col in columns]

        self.set_name = self.datasets[0].set_name
        self.classes = self.datasets[0].classes
        self.label_type = self.datasets[0].label_type
        self.task = self.datasets[0].task
        self.augmentation = augmentation
        self.missing_augmentation_probability = missing_augmentation_probability
        self.weighted_missing_augmentation_probabilities = self.datasets[0].weighted_missing_augmentation_probabilities
        if self.task in  ("regression", "survival_analysis"):
            self.decimals = self.datasets[0].decimals

    def drop_samples_with_all_modalities_missing(self, args):
        samples_missing_masking_all_modalities = torch.Tensor(np.concatenate([~dataset.missing_samples_mask[:,None] for dataset in args], axis=1)).all(axis=1)
        if any(samples_missing_masking_all_modalities):
            for dataset in args:
                dataset.filter_data(~samples_missing_masking_all_modalities.numpy())
        return args

    @property
    def ID(self):
        return self.datasets[0].ID

    @property
    def data(self):
        return [dataset.data for dataset in self.datasets]

    @property
    def labels(self):
        return self.datasets[0].labels

    @property
    def cv_labels(self):
        return self.datasets[0].cv_labels

    @property
    def columns(self):
        return self.__columns

    def set_augmentation(self, augmentation: bool):
        self.augmentation = augmentation

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        samples = list()
        label = None
        id = None
        for dataset in self.datasets:
            sample, label, id = dataset.__getitem__(index)
            samples.append(sample)

        if self.augmentation:
            if self.weighted_missing_augmentation_probabilities is not None:
                probability = min(np.sum(self.weighted_missing_augmentation_probabilities[self.cv_labels[index]]), 0.9)
            else:
                probability = self.missing_augmentation_probability
            samples = multimodal_missing_augmentation(samples, probability)
        if all([torch.isnan(s).all() for s in samples]):
            pass
        assert not all([torch.isnan(s).all() for s in samples]), f"Sample {id} is all NaN"

        return *samples, label, id

    def get_data(self):
        IDs = self.ID
        data = self.data
        labels = np.squeeze(self.labels)

        return data, labels, IDs


# class MultimodalEarlyFusionTabularDatasetTorch(MultimodalJointFusionTabularDatasetTorch):
#    """
#    Class for creating a dataset from multiple datasets. The datasets must have the same length.
#    """
#    def __init__(self, *args):
#        super().__init__(*args)

#    @property
#    def data(self):
#        return np.hstack([dataset.data for dataset in self.datasets])

#    def __getitem__(self, index):
#        samples = list()
#        label = None
#        id = None
#        for dataset in self.datasets:
#            sample, label, id = dataset.__getitem__(index)
#            samples.append(sample)

#        samples = np.hstack(samples)

#        return samples, label, id


if __name__ == "__main__":
    pass
