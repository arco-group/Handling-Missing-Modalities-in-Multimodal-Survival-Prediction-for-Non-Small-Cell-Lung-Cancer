import logging
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import List, Tuple

from CMC_utils.preprocessing.labels import *
from CMC_utils.miscellaneous import do_nothing
from .generic import DatasetSupervised

__all__ = ["SupervisedTaskDataset", "ClassificationDataset", "MultiLabelClassificationDataset", "RegressionDataset", "SurvivalDataset", "SurvivalFeaturesDataset"]

log = logging.getLogger(__name__)


class SupervisedTaskDataset(DatasetSupervised):
    """
    Class for supervised datasets.
    """
    __label_type: str = None
    __model_label_types: List[str] = None

    def __init__(self, db_type: str, **kwargs):
        DatasetSupervised.__init__(self, db_type = db_type, **kwargs)

    @abstractmethod
    def __set_label_type(self, label_type, model_label_types, model_framework):
        self.__label_type = label_type

    @abstractmethod
    def __set_label_encoding_functions(self):
        self.__label_encode = lambda label: label
        self.__label_decode = lambda label: label

        self.__set_label_encode_signature()

    @abstractmethod
    def __set_label_encode_signature(self):
        input_signature = ''  # + ('n' * (self.__label_type in ("single_risk_survival", "competing_risks_survival")))
        output_signature = ''  # + ('m' * (self.__label_type not in ('binary', 'discrete')))

        self.__label_encode_signature = f"({input_signature})->({output_signature})"

    @abstractmethod
    def __compute_info_for_cv(self):
        self.__info_for_cv = self.__labels

    @abstractmethod
    def __compute_labels_for_model(self):
        self.__labels_for_model = self.batch_encode(self.labels)

    @abstractmethod
    def label_encode( self, label, *_ ):
        encoded_label = self.__label_encode( label )
        return encoded_label

    @abstractmethod
    def label_decode( self, label, *_ ):
        decoded_label = self.__label_decode( label )
        return decoded_label

    @abstractmethod
    def batch_encode(self, labels, *_):
        label_function = np.vectorize(self.__label_encode, signature=self.__label_encode_signature)
        return label_function( labels )


class ClassificationDataset(SupervisedTaskDataset):
    """
    Class for classification datasets.
    """
    def __init__(self, label_type: str, model_label_types: List[str], model_framework: str, classes: Tuple[str], **kwargs):
        SupervisedTaskDataset.__init__(self, **kwargs)

        self.__classes = classes
        self.__set_label_type(label_type, model_label_types, model_framework)
        self.__set_label_encoding_functions()
        self.__compute_info_for_cv()
        self.__compute_labels_for_model()

        self.preprocessing_params["label_type"] = self.label_type
        self.preprocessing_params["classes"] = self.classes
        log.info("Dataset ready")

    @property
    def classes(self):
        return list(self.__classes)

    @property
    def label_type(self):
        return self.__label_type

    @property
    def info_for_cv(self):
        return self.__info_for_cv

    @property
    def labels_for_model(self):
        return self.__labels_for_model

    def __set_label_type(self, label_type, model_label_types, model_framework):
        if label_type == "multiclass":
            label_type = "categorical"
            label_type = label_type if label_type in model_label_types else "discrete"
        self.__label_type = label_type

    def __set_label_encoding_functions(self):
        if self.__label_type in ("binary", "discrete"):
            self.__label_encode = lambda label: label_to_discrete(label, self.__classes)
            self.__label_decode = lambda label: discrete_to_label(label, self.__classes)
        elif self.__label_type == "categorical":
            self.__label_encode = lambda label: label_to_categorical(label, self.__classes)
            self.__label_decode = lambda label: categorical_to_label(label, self.__classes)
        else:
            raise Exception("Sorry, desired label encoding does not exist.")
        self.__set_label_encode_signature()

    def __set_label_encode_signature(self):
        input_signature = ''
        output_signature = '' + ('m' * (self.__label_type == "categorical"))

        self.__label_encode_signature = f"({input_signature})->({output_signature})"

    def __compute_info_for_cv(self):
        self.__info_for_cv = self.labels.rename({"label": "target"}, axis=1)

    def __compute_labels_for_model(self):

        labels_for_model = self.batch_encode(self.labels)  # .drop("ID", axis=1))

        if self.__label_type == "categorical":
            labels_for_model = np.squeeze(labels_for_model)

        self.__labels_for_model = pd.DataFrame(labels_for_model, index=self.labels.index)

    def label_encode( self, label, *_ ):
        encoded_label = self.__label_encode( label )
        return encoded_label

    def label_decode( self, label, *_ ):
        decoded_label = self.__label_decode( label )
        return decoded_label

    def batch_encode(self, labels, *_ ):
        label_function = np.vectorize(self.__label_encode, signature=self.__label_encode_signature)
        return label_function( labels )


class MultiLabelClassificationDataset(SupervisedTaskDataset):
    """
    Class for multilabel classification datasets.
    """
    def __init__(self, label_type: str, model_label_types: List[str], model_framework: str, classes: Tuple[str], **kwargs):
        SupervisedTaskDataset.__init__(self, **kwargs)

        if kwargs.get("replace_classes", {}) is not None:
            classes_to_replace = {float(k): v if v is not None else np.nan for k, v in kwargs.get("replace_classes", {}).items()}
            self.set_labels(self.labels.applymap(lambda x: x if x not in classes_to_replace.keys() else classes_to_replace[x]))
        self.__classes = classes
        self.__set_label_type(label_type, model_label_types, model_framework)
        self.__set_label_encoding_functions()
        self.__compute_info_for_cv()
        self.__compute_labels_for_model()

        self.preprocessing_params["label_type"] = self.label_type
        self.preprocessing_params["classes"] = self.classes
        log.info("Dataset ready")

    @property
    def classes(self):
        return list(self.__classes)

    @property
    def label_type(self):
        return self.__label_type

    @property
    def info_for_cv(self):
        return self.__info_for_cv

    @property
    def labels_for_model(self):
        return self.__labels_for_model

    def __set_label_type(self, label_type, model_label_types, model_framework):
        self.__label_type = label_type

    def __set_label_encoding_functions(self):
        if self.__label_type == "multilabel":
            self.__label_encode = lambda label: do_nothing(label)
            self.__label_decode = lambda label: do_nothing(label)

        self.__set_label_encode_signature()

    def __set_label_encode_signature(self):
        input_signature = ''
        output_signature = ''

        self.__label_encode_signature = f"({input_signature})->({output_signature})"

    def __compute_info_for_cv(self):
        target_names = self.labels.columns.tolist()
        new_idxs = pd.MultiIndex.from_product([["target"], target_names])
        targets_for_cv = self.labels
        targets_for_cv.columns = new_idxs
        targets_for_cv = targets_for_cv.fillna(-100)
        self.__info_for_cv = targets_for_cv

    def __compute_labels_for_model(self):
        labels_for_model = pd.DataFrame(self.labels, index=self.labels.index)
        labels_for_model = labels_for_model.fillna(-100)
        self.__labels_for_model = labels_for_model


class RegressionDataset(SupervisedTaskDataset):
    """
    Class for classification datasets.
    """
    def __init__(self, label_type: str, model_label_types: List[str], model_framework: str, ngroups_for_cv: int = 4, **kwargs):
        SupervisedTaskDataset.__init__(self, **kwargs)

        self.__ngroups_for_cv = ngroups_for_cv
        self.__normalize_target = kwargs.get("normalize_target", False)
        self.__set_label_type(label_type, model_label_types, model_framework)
        self.__set_label_encoding_functions()
        self.__compute_info_for_cv()
        self.__compute_labels_for_model()

        self.preprocessing_params["label_type"] = self.label_type
        log.info("Dataset ready")

    @property
    def classes(self):
        return []

    @property
    def label_type(self):
        return self.__label_type

    @property
    def info_for_cv(self):
        return self.__info_for_cv

    @property
    def labels_for_model(self):
        return self.__labels_for_model

    def __set_label_type(self, label_type, *_):
        self.__label_type = label_type

    def __set_label_encoding_functions(self):
        if self.__label_type == "continuous" and self.__normalize_target:
            self.__label_encode = lambda label, y_range: target_normalization(label, *y_range)
            self.__label_decode = lambda label, y_range: target_denormalization(label, *y_range)
        elif self.__label_type == "continuous" and not self.__normalize_target:
            self.__label_encode = lambda label, y_range: do_nothing(label, *y_range, return_first=True)
            self.__label_decode = lambda label, y_range: do_nothing(label, *y_range, return_first=True)
        else:
            raise Exception("Sorry, desired label encoding does not exist.")
        self.__set_label_encode_signature()

    def __set_label_encode_signature(self):
        input_signature = '1,n'
        output_signature = ''

        self.__label_encode_signature = f"({input_signature})->({output_signature})"

    def __compute_info_for_cv(self):
        info_for_cv = pd.qcut(self.labels.label, q=self.__ngroups_for_cv, labels=False, duplicates="drop").rename("target")

        self.__info_for_cv = info_for_cv.to_frame()

        log.info(self.__info_for_cv["target"].value_counts())

    def __compute_labels_for_model(self):
        # labels_for_model = self.batch_encode(self.labels)
        # self.__labels_for_model = pd.DataFrame(labels_for_model, index=self.labels.index)
        self.__labels_for_model = self.labels

    def label_encode( self, label, *y_range ):
        encoded_label = self.__label_encode( label, y_range )
        return encoded_label

    def label_decode( self, label, *y_range ):
        decoded_label = self.__label_decode( label, y_range )
        return decoded_label

    def batch_encode(self, labels, *y_range: float):
        label_function = np.vectorize(self.__label_encode, signature=self.__label_encode_signature)
        return label_function( labels, y_range )


class SurvivalDataset(SupervisedTaskDataset):
    """
    Class for survival datasets.
    """
    # def __init__(self, name: str, db_type: str, task: str, dataset_class: Union[dict, DictConfig], path: str, preprocessing_params: DictConfig, label_type: str, model_label_types: List[str], model_framework: str, classes: Tuple[str], max_time: int, id_column: str = "ID", data_filename: str = "data.csv", labels_filename: str = "labels.csv", types_filename: str = "dtypes.csv", **kwargs):
        # SupervisedTaskDataset.__init__(self, name = name, db_type = db_type, task = task, dataset_class = dataset_class, path = path, preprocessing_params=preprocessing_params, id_column=id_column, data_filename=data_filename, labels_filename=labels_filename, types_filename=types_filename, **kwargs)
    def __init__(self, label_type: str, model_label_types: List[str], model_framework: str, classes: Tuple[str], max_time: int, time_divisor: int = 1, ngroups_for_cv: int = 4, use_bins: bool = True, **kwargs):
        SupervisedTaskDataset.__init__(self, **kwargs)

        self.__classes = classes
        self.__ngroups_for_cv = ngroups_for_cv
        self.__max_time = max_time//time_divisor
        self.__time_divisor = time_divisor
        self.use_bins = use_bins
        tmp_labels = self.labels
        tmp_labels.event_time = tmp_labels.event_time / time_divisor
        self.set_labels( tmp_labels )

        self.__set_label_type(label_type, model_label_types, model_framework)
        self.__set_label_encoding_functions()
        self.__compute_info_for_cv()
        self.__compute_labels_for_model()

        self.preprocessing_params["max_time"] = self.__max_time
        self.preprocessing_params["num_events"] = self.num_events

        self.preprocessing_params["label_type"] = self.label_type
        self.preprocessing_params["classes"] = self.classes

        log.info("Dataset ready")

    @property
    def max_time(self):
        return self.__max_time

    @property
    def classes(self):
        return list(self.__classes[1:])

    @property
    def num_events(self):
        return len(self.classes)

    @property
    def label_type(self):
        return self.__label_type

    @property
    def info_for_cv(self):
        return self.__info_for_cv

    @property
    def labels_for_model(self):
        return self.__labels_for_model

    def __set_label_type(self, label_type, model_label_types, model_framework):
        if label_type == "single_risk_survival" and model_framework == "xgboost":  # and label_type not in model_label_types:
            label_type = "single_risk_survival_interval"
        self.__label_type = label_type

    def __set_label_encoding_functions(self):
        if self.__label_type in ("single_risk_survival", "competing_risks_survival", "single_risk_survival_regression"):
            self.__label_encode = lambda label: label_to_survival(label, self.__classes, self.__max_time)
            self.__label_decode = lambda label: survival_to_label(label, self.__classes, self.__max_time)
        elif self.__label_type == "single_risk_survival_interval":
            self.__label_encode = lambda label: label  # TODO
            self.__label_decode = lambda label: label  # TODO
        else:
            raise Exception("Sorry, desired label encoding does not exist.")
        self.__set_label_encode_signature()

    def __set_label_encode_signature(self):
        input_signature = 'n'
        output_signature = 'm'

        self.__label_encode_signature = f"({input_signature})->({output_signature})"

    def __compute_info_for_cv(self):
        if self.use_bins:
            time_slots = pd.qcut(self.labels.event_time, q=self.__ngroups_for_cv, labels=False, duplicates="drop")
        else:
            time_slots = self.labels.event_time
        log.info(time_slots.value_counts())

        info_for_cv = self.labels.assign(time_slot=time_slots)
        info_for_cv["target"] = info_for_cv.agg(lambda x: f"{x['label']}_{x['time_slot']}", axis=1)
        self.__info_for_cv = info_for_cv[["target"]]  # "ID",

        log.info(self.__info_for_cv["target"].value_counts())

    def __compute_labels_for_model(self):
        labels_for_model = self.batch_encode(self.labels)  # .drop("ID", axis=1))
        self.__labels_for_model = pd.DataFrame(labels_for_model, index=self.labels.index)

    def label_encode( self, label, *_ ):
        encoded_label = self.__label_encode( label )
        return encoded_label

    def label_decode( self, label, *_ ):
        decoded_label = self.__label_decode( label )
        return decoded_label

    def batch_encode(self, labels, *_ ):


        label_function = np.vectorize(self.__label_encode, signature=self.__label_encode_signature)
        labels = label_function(labels)
        if self.task == 'survival_analysis':

            labels = get_transform_labels(labels[:,1] , labels[:,0], self.preprocessing_params["label"])
            labels = np.stack(labels, axis=1)
        return labels


class SurvivalFeaturesDataset(SupervisedTaskDataset):
    """
    Class for survival datasets.
    """
    # def __init__(self, name: str, db_type: str, task: str, dataset_class: Union[dict, DictConfig], path: str, preprocessing_params: DictConfig, label_type: str, model_label_types: List[str], model_framework: str, classes: Tuple[str], max_time: int, id_column: str = "ID", data_filename: str = "data.csv", labels_filename: str = "labels.csv", types_filename: str = "dtypes.csv", **kwargs):
        # SupervisedTaskDataset.__init__(self, name = name, db_type = db_type, task = task, dataset_class = dataset_class, path = path, preprocessing_params=preprocessing_params, id_column=id_column, data_filename=data_filename, labels_filename=labels_filename, types_filename=types_filename, **kwargs)
    def __init__(self, label_type: str, model_label_types: List[str], model_framework: str, classes: Tuple[str], max_time: int, time_divisor: int = 1, ngroups_for_cv: int = 4, use_bins: bool = True, **kwargs):
        SupervisedTaskDataset.__init__(self, **kwargs)

        self.__classes = classes
        self.__ngroups_for_cv = ngroups_for_cv
        self.__max_time = max_time//time_divisor
        self.__time_divisor = time_divisor
        self.use_bins = use_bins
        tmp_labels = self.labels
        tmp_labels.event_time = tmp_labels.event_time / time_divisor
        self.set_labels( tmp_labels )

        self.__set_label_type(label_type, model_label_types, model_framework)
        self.__set_label_encoding_functions()
        self.__compute_info_for_cv()
        self.__compute_labels_for_model()

        self.preprocessing_params["max_time"] = self.__max_time
        self.preprocessing_params["num_events"] = self.num_events

        self.preprocessing_params["label_type"] = self.label_type
        self.preprocessing_params["classes"] = self.classes

        log.info("Dataset ready")

    @property
    def max_time(self):
        return self.__max_time

    @property
    def classes(self):
        return list(self.__classes[1:])

    @property
    def num_events(self):
        return len(self.classes)

    @property
    def label_type(self):
        return self.__label_type

    @property
    def info_for_cv(self):
        return self.__info_for_cv

    @property
    def labels_for_model(self):
        return self.__labels_for_model

    def __set_label_type(self, label_type, model_label_types, model_framework):
        if label_type == "single_risk_survival" and model_framework == "xgboost":  # and label_type not in model_label_types:
            label_type = "single_risk_survival_interval"
        self.__label_type = label_type

    def __set_label_encoding_functions(self):
        if self.__label_type in ("single_risk_survival", "competing_risks_survival", "single_risk_survival_regression"):
            self.__label_encode = lambda label: label_to_survival(label, self.__classes, self.__max_time)
            self.__label_decode = lambda label: survival_to_label(label, self.__classes, self.__max_time)
        elif self.__label_type == "single_risk_survival_interval":
            self.__label_encode = lambda label: label  # TODO
            self.__label_decode = lambda label: label  # TODO
        else:
            raise Exception("Sorry, desired label encoding does not exist.")
        self.__set_label_encode_signature()

    def __set_label_encode_signature(self):
        input_signature = 'n'
        output_signature = 'm'

        self.__label_encode_signature = f"({input_signature})->({output_signature})"

    def __compute_info_for_cv(self):
        if self.use_bins:
            time_slots = pd.qcut(self.labels.event_time, q=self.__ngroups_for_cv, labels=False, duplicates="drop")
        else:
            time_slots = self.labels.event_time
        log.info(time_slots.value_counts())

        info_for_cv = self.labels.assign(time_slot=time_slots)
        info_for_cv["target"] = info_for_cv.agg(lambda x: f"{x['label']}_{x['time_slot']}", axis=1)
  # "ID",

        mask_all_nan = self.data.isna().all(axis=1)

        info_for_cv.loc[:,'to_use'] = (~mask_all_nan).astype(int).values
        self.__info_for_cv = info_for_cv[["target", "to_use"]]

        log.info(self.__info_for_cv["target"].value_counts())

    def __compute_labels_for_model(self):
        labels_for_model = self.batch_encode(self.labels)  # .drop("ID", axis=1))
        self.__labels_for_model = pd.DataFrame(labels_for_model, index=self.labels.index)

    def label_encode( self, label, *_ ):
        encoded_label = self.__label_encode( label )
        return encoded_label

    def label_decode( self, label, *_ ):
        decoded_label = self.__label_decode( label )
        return decoded_label

    def batch_encode(self, labels, *_ ):
        label_function = np.vectorize(self.__label_encode, signature=self.__label_encode_signature)
        # apply transform to time and event
        labels = label_function(labels)
        if self.task == 'survival_analysis':
            labels = get_transform_labels(labels[:,1] , labels[:,0], self.preprocessing_params["label"])
            labels = np.stack(labels, axis=1)
        return labels


if __name__ == "__main__":
    pass
