import os
import logging
import numpy as np
from hydra.utils import call
from CMC_utils.save_load import create_directory
from CMC_utils.miscellaneous import do_nothing
from CMC_utils.datasets import SupervisedTabularDatasetTorch

log = logging.getLogger(__name__)


__all__ = ["train_sklearn_model"]


def process_survival_labels(labels: np.array) -> np.array:
    """
    Convert survival labels to a structured array
    Parameters
    ----------
    labels : np.array

    Returns
    -------
    np.array
    """
    labels = np.array( list(map(tuple, labels)), dtype=[('label', 'bool'), ('event_time', '<f8')])
    return labels


def train_sklearn_model(model, train_set: SupervisedTabularDatasetTorch, model_params: dict, model_path: str, test_fold: int = 0, val_fold: int = 0, **kwargs) -> None:
    """
    Train a sklearn model
    Parameters
    ----------
    model : sklearn model
    train_set : SupervisedTabularDatasetTorch
    model_params : dict
    model_path : str
    test_fold : int
    val_fold : int
    kwargs : dict

    Returns
    -------
    None
    """
    # model_path = os.path.join(model_path, model_params["name"])
    create_directory(model_path)
    filename = f"{test_fold}_{val_fold}"

    if os.path.exists(os.path.join(model_path, filename + f".{model_params['file_extension']}")):
        return

    data, labels, _ = train_set.get_data()

    labels_options = dict(binary=do_nothing, discrete=do_nothing, continuous=do_nothing, single_risk_survival=process_survival_labels, single_risk_survival_regression=process_survival_labels, competing_risks_survival=process_survival_labels)
    labels = labels_options[train_set.label_type](labels)

    if model_params["name"] == "TabNet" and train_set.label_type == "continuous":
        labels = np.expand_dims(labels, 1)

    model.fit(data.astype("float32"), labels, **model_params["fit_params"])

    call(model_params["save_function"], model, filename, model_path, model_params=model_params["init_params"], extension=model_params["file_extension"], _recursive_=False)

    log.info("Model trained")


if __name__ == "__main__":
    pass
