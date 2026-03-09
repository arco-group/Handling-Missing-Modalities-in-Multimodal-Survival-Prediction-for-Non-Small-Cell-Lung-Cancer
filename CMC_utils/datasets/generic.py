import os
from omegaconf import DictConfig
import numpy as np
import pandas as pd

from CMC_utils.save_load import load_tabular_dataset
from CMC_utils.preprocessing import get_preprocessing_params, get_imaging_preprocessing_params
from CMC_utils.preprocessing import get_transform_labels

__all__ = ["Dataset", "DatasetSupervised"]


class Dataset:
    """
    Generic class for datasets
    """
    def __init__(self, name: str, db_type: str, task: str, **_):
        self.__name = name
        self.__db_type = db_type
        self.__task = task

    @property
    def name(self):
        return self.__name

    @property
    def db_type(self):
        return self.__db_type

    @property
    def task(self):
        return self.__task


class DatasetSupervised(Dataset):
    """
    Generic class for tabular datasets
    """
    def __init__(self, name: str, db_type: str, task: str, preprocessing_params: DictConfig, **kwargs):
        """
        This class is used to load tabular datasets
        Parameters
        ----------
        name : str
        db_type : str
        task : str
        preprocessing_params : DictConfig
        kwargs : dict
        """
        Dataset.__init__(self, name=name, db_type=db_type, task=task)

        self.__load_dataset(task=task, **kwargs)

        if db_type == "tabular" or db_type == "features":
            self.__compute_tabular_parameters(preprocessing_params=preprocessing_params)
        else:
            self.__compute_imaging_parameters(preprocessing_params=preprocessing_params)
            self.__set_paths(**kwargs)

    @property
    def data(self):
        return self.__data.loc[:, self.__features_types.index.to_list()].copy()

    @property
    def complete_idxs_data(self):
        return self.__data.copy()

    @property
    def labels(self):
        other_idxs_cols = [col for col in self.__data.columns if col not in self.__features_types.index.to_list()]
        if other_idxs_cols:
            return self.__labels.drop(other_idxs_cols, axis=1).copy()
        else:
            return self.__labels.copy()

    def set_labels(self, labels):
        self.__labels = labels

    @property
    def complete_idxs_labels(self):
        return self.__labels.copy()

    @property
    def feature_types(self):
        return self.__features_types.copy()

    def __load_dataset(self, **kwargs):
        kwargs["db_type"] = self.db_type
        data, labels, features_types = load_tabular_dataset(**kwargs)
        self.__data = data
        self.__labels = labels
        self.__features_types = features_types

    def __compute_tabular_parameters(self, preprocessing_params: DictConfig):
        self.preprocessing_params = get_preprocessing_params(self.data, self.feature_types, preprocessing_params)

    def __compute_imaging_parameters(self, preprocessing_params: DictConfig = None):
        self.preprocessing_params = get_imaging_preprocessing_params(self.feature_types, preprocessing_params)

    def __set_paths(self, data_folder: str, columns: dict, **kwargs):
        cols_info = {v: k for k, v in columns.items()}
        self.__data["abs_path"] = self.__data.apply(lambda row: os.path.join(data_folder, row[cols_info["folder"]], row[cols_info["image"]]) if pd.notna(row[cols_info["image"]]) else np.nan, axis=1)
        self.__features_types["abs_path"] = "path"


if __name__ == "__main__":
    pass
