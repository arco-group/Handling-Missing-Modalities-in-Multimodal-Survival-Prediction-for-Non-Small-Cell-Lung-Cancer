import os
import torch
import logging
import torchvision
import numpy as np
import pandas as pd
from typing import List
from hydra.utils import instantiate
from torch.utils.data import Dataset
from torchvision.transforms import v2
from CMC_utils.save_load import load_image
from CMC_utils.data_augmentation import missing_augmentation, RandomOrderApply, compute_weighted_probability
from CMC_utils.preprocessing import set_fold_preprocessing, apply_preprocessing, set_target_preprocessing, apply_target_preprocessing

log = logging.getLogger(__name__)

__all__ = ["SupervisedTabularDatasetTorch", "SupervisedImagingDatasetTorch", "SupervisedFeaturesDatasetTorch"]


class SupervisedTabularDatasetTorch(Dataset):
    """
    Supervised tabular dataset for PyTorch
    """
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame, db_task: str, set_name: str, preprocessing_params: dict, preprocessing_paths: dict, cv_info: pd.DataFrame = None, test_fold: int = 0, val_fold: int = 0, augmentation: bool = False, missing_augmentation_probability: float = 0.5, weight_missing_augmentation_probability: bool = False, normalize_target: bool = False, dropna_samples_training: bool = False, decimals: int = 2, **kwargs):
        if set_name == "train":
            set_fold_preprocessing(data, test_fold, val_fold, preprocessing_paths, preprocessing_params, **kwargs)
            if normalize_target and db_task == "regression":
                set_target_preprocessing(labels, test_fold, val_fold, preprocessing_paths, **kwargs)

        [data] = apply_preprocessing(data, preprocessing_paths=preprocessing_paths, preprocessing_params=preprocessing_params, test_fold=test_fold, val_fold=val_fold, **kwargs)
        if normalize_target and db_task == "regression":
            labels = apply_target_preprocessing(labels, test_fold, val_fold, preprocessing_paths, **kwargs)

        self.__ID = data.index.values  # .to_list()
        self.__data = data.values
        self.__labels = labels.values
        self.__columns = data.columns
        self.dropna_samples_training = dropna_samples_training
        self.missing_samples_mask = ~pd.isna(self.__data).all(axis=1)

        self.set_name = set_name
        self.task = db_task
        if self.task == "classification":
            self.classes = preprocessing_params["classes"]
        elif self.task == "regression":
            self.classes = None
            self.decimals = decimals
        elif preprocessing_params["label_type"] =="single_risk_survival_regression":
            self.classes = preprocessing_params["classes"]
            self.decimals = decimals
        else:
            self.classes = preprocessing_params["classes"]

        self.label_type = preprocessing_params["label_type"]

        self.input_size = self.__data.shape[1]
        self.output_size = self.__labels.shape[1]

        self.augmentation = augmentation
        self.missing_augmentation_probability = missing_augmentation_probability
        if cv_info is not None and weight_missing_augmentation_probability:
            cv_info, class_weighted_probabilities = compute_weighted_probability(missing_augmentation_probability, cv_info, self.label_type)
            self.__cv_labels = cv_info
            self.weighted_missing_augmentation_probabilities = class_weighted_probabilities
        else:
            self.__cv_labels = None
            self.weighted_missing_augmentation_probabilities = None

        self.set_dropna_samples_training(dropna_samples_training)

    @property
    def ID(self):
        if self.dropna_samples_training:
            return self.__ID_not_missing
        return self.__ID

    @property
    def data(self):
        if self.dropna_samples_training:
            return self.__data_not_missing
        return self.__data


    def filter_data(self, mask):
        self.__ID = self.__ID[mask]
        self.__data = self.__data[mask]
        self.__labels = self.__labels[mask]
        if self.__cv_labels is not None:
            self.__cv_labels = self.__cv_labels[mask]
        self.missing_samples_mask = self.missing_samples_mask[mask]




    @property
    def labels(self):
        if self.dropna_samples_training:
            return self.__labels_not_missing
        return self.__labels

    @property
    def cv_labels(self):
        if self.__cv_labels is None:
            return None
        if self.dropna_samples_training:
            return self.__cv_labels_not_missing
        return self.__cv_labels

    @property
    def columns(self):
        return self.__columns


    def set_augmentation(self, augmentation: bool):
        self.augmentation = augmentation

    def set_dropna_samples_training(self, dropna_samples_training: bool):
        self.dropna_samples_training = dropna_samples_training
        if self.dropna_samples_training:
            self.__ID_not_missing = self.__ID[self.missing_samples_mask]
            self.__data_not_missing = self.__data[self.missing_samples_mask]
            self.__labels_not_missing = self.__labels[self.missing_samples_mask]
            if self.__cv_labels is not None:
                self.__cv_labels_not_missing = self.__cv_labels[self.missing_samples_mask]
            else:
                self.__cv_labels_not_missing = None
        else:
            self.__ID_not_missing = None
            self.__data_not_missing = None
            self.__labels_not_missing = None
            self.__cv_labels_not_missing = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        ID = self.ID[index]
        sample = self.data[index]
        label = self.labels[index]

        sample = torch.tensor(np.array( sample, dtype=float ))

        if self.augmentation:
            if self.weighted_missing_augmentation_probabilities is not None:
                probability = min(np.sum(self.weighted_missing_augmentation_probabilities[self.cv_labels[index]]), 0.9)
            else:
                probability = self.missing_augmentation_probability
            sample = missing_augmentation(sample.clone(), probability)

        return sample, label, ID

    def get_data(self):
        IDs = self.ID
        data = self.data
        labels = np.squeeze(self.labels)

        return data, labels, IDs


class SupervisedImagingDatasetTorch(Dataset):
    """
    Supervised tabular dataset for PyTorch
    """
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame, db_task: str, set_name: str, preprocessing_params: dict, preprocessing_paths: dict, cv_info: pd.DataFrame = None, test_fold: int = 0, val_fold: int = 0, missing_augmentation_probability: float = 0.5, weight_missing_augmentation_probability: bool = False, normalize_target: bool = False, image_caching: bool = False, dropna_samples_training: bool = False, decimals: int = 2, original_images_dims: List[int] = None, **kwargs):
        if set_name == "train":
            if normalize_target and db_task == "regression":
                set_target_preprocessing(labels, test_fold, val_fold, preprocessing_paths, **kwargs)

        if normalize_target and db_task == "regression":
            labels = apply_target_preprocessing(labels, test_fold, val_fold, preprocessing_paths, **kwargs)

        self.__ID = data.index.values
        self.__data = data.values
        self.__images = dict()
        self.__missing_image = None
        self.__paths = data.abs_path.values
        self.__labels = labels.values
        self.__columns = data.columns
        self.caching = image_caching
        self.dropna_samples_training = False
        self.missing_samples_mask = ~pd.isna(self.__paths)

        self.set_name = set_name
        self.task = db_task
        if self.task == "classification":
            self.classes = preprocessing_params["classes"]
        elif self.task == "regression":
            self.classes = None
            self.decimals = decimals
        else:
            self.classes = preprocessing_params["classes"]
        self.label_type = preprocessing_params["label_type"]

        self.input_size = original_images_dims
        preprocessing_functions = preprocessing_params.get("images", {})
        if len(preprocessing_functions) > 0:
            keys_to_eval = ("interpolation", )
            for i in range(len(preprocessing_functions)):
                if self.input_size is not None:
                    if preprocessing_functions[str(i)].get("size", None) is not None:
                        new_size = preprocessing_functions[str(i)].get("size")
                        self.input_size = new_size if len(self.input_size) == 2 else [self.input_size[0], *new_size]
                    if preprocessing_functions[str(i)]["_target_"].endswith("Grayscale") and len(self.input_size) > 2:
                        self.input_size = self.input_size[-2:]
                    if preprocessing_functions[str(i)]["_target_"].endswith("Gray2RGB"):
                        self.input_size = [3, *self.input_size[-2:]]
                for key in keys_to_eval:
                    if preprocessing_functions[str(i)].get(key, None) is not None:
                        preprocessing_functions[str(i)][key] = eval(preprocessing_functions[str(i)].get(key, None))
                preprocessing_functions[str(i)] = instantiate(preprocessing_functions[str(i)])
            self.preprocessing = v2.Compose([preprocessing_functions[str(i)] for i in range(len(preprocessing_functions))])
        else:
            self.preprocessing = v2.Identity()

        transformations = preprocessing_params.get("images_augmentation", {})
        if len(transformations) > 0:
            keys_to_eval = ("interpolation",)
            how_to_apply = {i: transformation for i, transformation in transformations.items() if transformation.get("method", None) == "random_order_apply"}  # isinstance(transformation, dict) and
            for i in range(len(transformations)):
                for key in keys_to_eval:
                    if transformations[str(i)].get(key, None) is not None:
                        transformations[str(i)][key] = eval(transformations[str(i)].get(key, None))
                # transformations[str(i)] = instantiate(transformations[str(i)])

            if how_to_apply:
                transformations_to_apply = [transformations[i] for i in transformations if int(i) < int(list(how_to_apply.keys())[0])]
                transformations_to_apply_randomly = [transformations[i] for i in transformations if int(i) > int(list(how_to_apply.keys())[0])]
                for transformation in transformations_to_apply_randomly:
                    if "p" in transformation.keys():
                        transformation["p"] = 1.0

                transformations_to_apply = [instantiate(transformation) for transformation in transformations_to_apply]
                transformations_to_apply_randomly = [instantiate(transformation) for transformation in transformations_to_apply_randomly]
                self.augmentation = v2.Compose(transformations_to_apply + [RandomOrderApply(transforms=transformations_to_apply_randomly, p=list(how_to_apply.values())[0]["p"])] )
            else:
                transformations_fn = [instantiate(transformations[str(i)]) for i in range(len(transformations))]
                self.augmentation = v2.Compose(transformations_fn)

            self.alternative_preprocessing = []
            for i, transformation in transformations.items():
                if transformation.get("_target_", None) in ("torchvision.transforms.v2.Resize", "CMC_utils.preprocessing.Padding"):
                    if self.input_size is not None:
                        if transformation.get("size", None) is not None:
                            new_size = transformation.get("size")
                            self.input_size = new_size if len(self.input_size) == 2 else [self.input_size[0], *new_size]
                        if transformation["_target_"].endswith("Grayscale") and len(self.input_size) > 2:
                            self.input_size = self.input_size[-2:]
                        if transformation["_target_"].endswith("Gray2RGB"):
                            self.input_size = [3, *self.input_size[-2:]]
                    self.alternative_preprocessing.append(instantiate(transformation))

            if len(self.alternative_preprocessing) == 0:
                self.alternative_preprocessing = v2.Identity()
            else:
                self.alternative_preprocessing = v2.Compose(self.alternative_preprocessing)

        else:
            self.augmentation = v2.Identity()
            self.alternative_preprocessing = v2.Identity()

        self.missing_augmentation_probability = missing_augmentation_probability
        if cv_info is not None and weight_missing_augmentation_probability:
            cv_info, class_weighted_probabilities = compute_weighted_probability(missing_augmentation_probability, cv_info, self.label_type)
            self.__cv_labels = cv_info
            self.weighted_missing_augmentation_probabilities = class_weighted_probabilities
        else:
            self.__cv_labels = None
            self.weighted_missing_augmentation_probabilities = None

        self.set_dropna_samples_training(dropna_samples_training)

        self.do_augmentation = False
        if self.input_size is None:
            input_size = tuple(self.__getitem__(0)[0].shape)
            self.input_size = input_size
        self.output_size = self.__labels.shape[1]

        self.do_augmentation = set_name == "train" and len(transformations) > 0

    @property
    def ID(self):
        if self.dropna_samples_training:
            return self.__ID_not_missing
        return self.__ID

    @property
    def data(self):
        if self.dropna_samples_training:
            return self.__data_not_missing
        return self.__data

    @property
    def images(self):
        # if self.dropna_samples_training:
        #     return self.__images_not_missing
        return self.__images

    @property
    def paths(self):
        if self.dropna_samples_training:
            return self.__paths_not_missing
        return self.__paths

    @property
    def labels(self):
        if self.dropna_samples_training:
            return self.__labels_not_missing
        return self.__labels

    @property
    def cv_labels(self):
        if self.__cv_labels is None:
            return None
        if self.dropna_samples_training:
            return self.__cv_labels_not_missing
        return self.__cv_labels

    @property
    def columns(self):
        return self.__columns

    def load_sample(self, index):
        if index not in self.__images.keys():

            if pd.isna(self.paths[index]) or not os.path.isfile(self.paths[index]):
                # log.error(f"Path not found: {self.__paths[index]}")
                if self.__missing_image is None:
                    # img = torch.tensor(load_image(np.random.choice(self.paths[pd.notna(self.paths)])))
                    # img = self.preprocessing(img)
                    # img = torch.full(img.shape, torch.nan)
                    img = torch.full(self.input_size, torch.nan)
                    self.__missing_image = img
                else:
                    img = self.__missing_image
            else:
                img = torch.tensor(load_image(self.paths[index]))
                img = self.preprocessing(img)
                if self.caching:
                    self.__images[index] = img
        else:
            img = self.__images[index]
        return img.clone()

    def set_augmentation(self, do_augmentation: bool):
        self.do_augmentation = do_augmentation

    def set_dropna_samples_training(self, dropna_samples_training: bool):
        if self.dropna_samples_training != dropna_samples_training:
            self.__images = dict()
            self.dropna_samples_training = dropna_samples_training
        if self.dropna_samples_training:
            self.__ID_not_missing = self.__ID[self.missing_samples_mask]
            self.__data_not_missing = self.__data[self.missing_samples_mask]
            #if self.__images is not None:
            #    self.__images_not_missing = self.__images[self.missing_samples_mask]
            #else:
            #    self.__images_not_missing = None
            self.__paths_not_missing = self.__paths[self.missing_samples_mask]
            self.__labels_not_missing = self.__labels[self.missing_samples_mask]
            if self.__cv_labels is not None:
                self.__cv_labels_not_missing = self.__cv_labels[self.missing_samples_mask]
            else:
                self.__cv_labels_not_missing = None
        else:
            self.__ID_not_missing = None
            self.__data_not_missing = None
            # self.__images_not_missing = None
            self.__paths_not_missing = None
            self.__labels_not_missing = None
            self.__cv_labels_not_missing = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        ID = self.ID[index]
        sample = self.load_sample(index)
        label = self.labels[index]

        if not torch.isnan(sample).all():
            if self.do_augmentation:
                sample = self.augmentation(sample)
            else:
                sample = self.alternative_preprocessing(sample)

        return sample, label, ID



class SupervisedFeaturesDatasetTorch(Dataset):
    """
    Supervised tabular dataset for PyTorch
    """
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame, db_task: str, set_name: str, preprocessing_params: dict, preprocessing_paths: dict, cv_info: pd.DataFrame = None, test_fold: int = 0, val_fold: int = 0, augmentation: bool = False, missing_augmentation_probability: float = 0.5, weight_missing_augmentation_probability: bool = False, normalize_target: bool = False, dropna_samples_training: bool = False, decimals: int = 2, **kwargs):
        if set_name == "train":
            set_fold_preprocessing(data, test_fold, val_fold, preprocessing_paths, preprocessing_params, **kwargs)
            if normalize_target and db_task == "regression":
                set_target_preprocessing(labels, test_fold, val_fold, preprocessing_paths, **kwargs)

        [data] = apply_preprocessing(data, preprocessing_paths=preprocessing_paths, preprocessing_params=preprocessing_params, test_fold=test_fold, val_fold=val_fold, **kwargs)
        if normalize_target and db_task == "regression":
            labels = apply_target_preprocessing(labels, test_fold, val_fold, preprocessing_paths, **kwargs)
        return_torch = preprocessing_params.get("return_torch",    False)



        self.__ID = data.index.values  # .to_list()
        self.__data = data.values
        self.__labels = labels.values
        self.__columns = data.columns
        self.dropna_samples_training = dropna_samples_training
        self.missing_samples_mask = ~pd.isna(self.__data).all(axis=1)

        self.set_name = set_name
        self.task = db_task
        if self.task == "classification":
            self.classes = preprocessing_params["classes"]
        elif self.task == "regression":
            self.classes = None
            self.decimals = decimals
        elif preprocessing_params["label_type"] =="single_risk_survival_regression":
            self.classes = preprocessing_params["classes"]
            self.decimals = decimals
        else:
            self.classes = preprocessing_params["classes"]

        self.label_type = preprocessing_params["label_type"]

        self.input_size = self.__data.shape[1]
        self.output_size = self.__labels.shape[1]

        self.augmentation = augmentation
        self.missing_augmentation_probability = missing_augmentation_probability
        if cv_info is not None and weight_missing_augmentation_probability:
            cv_info, class_weighted_probabilities = compute_weighted_probability(missing_augmentation_probability, cv_info, self.label_type)
            self.__cv_labels = cv_info
            self.weighted_missing_augmentation_probabilities = class_weighted_probabilities
        else:
            self.__cv_labels = None
            self.weighted_missing_augmentation_probabilities = None

        self.set_dropna_samples_training(dropna_samples_training)

    @property
    def ID(self):
        if self.dropna_samples_training:
            return self.__ID_not_missing
        return self.__ID

    @property
    def data(self):
        if self.dropna_samples_training:
            return self.__data_not_missing
        return self.__data

    @property
    def labels(self):
        if self.dropna_samples_training:
            return self.__labels_not_missing
        return self.__labels

    @property
    def cv_labels(self):
        if self.__cv_labels is None:
            return None
        if self.dropna_samples_training:
            return self.__cv_labels_not_missing
        return self.__cv_labels

    @property
    def columns(self):
        return self.__columns



    def filter_data(self, mask):
        self.__ID = self.__ID[mask]
        self.__data = self.__data[mask]
        self.__labels = self.__labels[mask]
        if self.__cv_labels is not None:
            self.__cv_labels = self.__cv_labels[mask]
        self.missing_samples_mask = self.missing_samples_mask[mask]



    def set_augmentation(self, augmentation: bool):
        self.augmentation = augmentation

    def set_dropna_samples_training(self, dropna_samples_training: bool):
        self.dropna_samples_training = dropna_samples_training
        if self.dropna_samples_training:
            self.__ID_not_missing = self.__ID[self.missing_samples_mask]
            self.__data_not_missing = self.__data[self.missing_samples_mask]
            self.__labels_not_missing = self.__labels[self.missing_samples_mask]
            if self.__cv_labels is not None:
                self.__cv_labels_not_missing = self.__cv_labels[self.missing_samples_mask]
            else:
                self.__cv_labels_not_missing = None
        else:
            self.__ID_not_missing = None
            self.__data_not_missing = None
            self.__labels_not_missing = None
            self.__cv_labels_not_missing = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        ID = self.ID[index]
        sample = self.data[index]
        label = self.labels[index]


        sample = torch.tensor(np.array( sample, dtype=float ))

        if self.augmentation:
            if self.weighted_missing_augmentation_probabilities is not None:
                probability = min(np.sum(self.weighted_missing_augmentation_probabilities[self.cv_labels[index]]), 0.9)
            else:
                probability = self.missing_augmentation_probability
            sample = missing_augmentation(sample.clone(), probability)

        return sample, label, ID

    def get_data(self):
        IDs = self.ID
        data = self.data
        labels = np.squeeze(self.labels)

        return data, labels, IDs

if __name__ == "__main__":
    pass
