import os
import logging
import numpy as np
import pandas as pd
from hydra.utils import call
from CMC_utils.miscellaneous import do_nothing
from CMC_utils.preprocessing import survival_to_label, discrete_to_label
from CMC_utils.save_load import save_table
from CMC_utils.datasets import SupervisedTabularDatasetTorch

log = logging.getLogger(__name__)

__all__ = ["test_sklearn_model"]


def process_survival_outputs(outputs, model, max_time, **kwargs) -> np.ndarray:
    """
    Process the outputs of a survival model to obtain a matrix of probabilities for each time step
    Parameters
    ----------
    outputs : np.ndarray
    model : sklearn model
    max_time : int
    kwargs : dict

    Returns
    -------
    np.ndarray
    """
    outputs = pd.DataFrame(outputs, columns=model.unique_times_)

    complete_output = pd.DataFrame(columns=np.arange(max_time))
    complete_output = pd.concat([complete_output, outputs], axis=0).ffill(axis=1).bfill(axis=1)  # .fillna(0)
    return complete_output.values


def infer_classification_model(model) -> callable:
    """
    Infer the predict function of a sklearn model
    Parameters
    ----------
    model : sklearn model

    Returns
    -------
    function
    """
    return model.predict_proba


def infer_regression_model(model) -> callable:
    """
    Infer the predict function of a sklearn model
    Parameters
    ----------
    model : sklearn model

    Returns
    -------
    function
    """
    return model.predict


def infer_survival_model(model) -> callable:
    """
    Infer the predict function of a sklearn model
    Parameters
    ----------
    model : sklearn model

    Returns
    -------
    function
    """
    return model.predict_cumulative_hazard_function


def test_sklearn_model(*sets: SupervisedTabularDatasetTorch, model_params: dict, model_path: str, prediction_path: str, test_fold: int = 0, val_fold: int = 0, **kwargs) -> None:
    """
    Test a sklearn model
    Parameters
    ----------
    sets : SupervisedTabularDatasetTorch
    model_params : dict
    model_path : str
    prediction_path : str
    test_fold : int
    val_fold : int
    kwargs : dict

    Returns
    -------
    None
    """
    model_path = os.path.join( model_path, f"{test_fold}_{val_fold}.{model_params['file_extension']}")
    model = call(model_params["load_function"], model_path)

    model_options = dict(binary=infer_classification_model, discrete=infer_classification_model, continuous=infer_regression_model, single_risk_survival= infer_survival_model, single_risk_survival_regression= infer_regression_model, competing_risks_survival= infer_survival_model)
    model_params_options = dict(binary={}, discrete={}, continuous={}, single_risk_survival=dict(return_array=True), single_risk_survival_regression={}, competing_risks_survival=dict(return_array=True))

    output_options = dict(binary=do_nothing, discrete=do_nothing, continuous=do_nothing, single_risk_survival=process_survival_outputs, single_risk_survival_regression=do_nothing, competing_risks_survival=process_survival_outputs)
    output_params_options = dict(binary={}, discrete={}, continuous={}, single_risk_survival=dict(max_time=model_params.get("max_time")), single_risk_survival_regression={}, competing_risks_survival=dict(max_time=model_params.get("max_time")))

    labels_options = dict(binary=discrete_to_label, discrete=discrete_to_label, continuous=do_nothing, single_risk_survival=survival_to_label, single_risk_survival_regression=do_nothing, competing_risks_survival=survival_to_label)

    labels_params_options = dict(binary={"classes": sets[0].classes}, discrete={"classes": sets[0].classes}, continuous={}, single_risk_survival={"classes": sets[0].classes, "max_time": model_params.get("max_time")}, single_risk_survival_regression={}, competing_risks_survival={"classes": sets[0].classes, "max_time": model_params.get("max_time")})

    for fset in sets:

        filename = f"{test_fold}_{val_fold}_{fset.set_name}.csv"
        if os.path.exists(os.path.join(prediction_path, filename)):
            continue

        data, labels, ID = fset.get_data()

        missing_samples_mask = pd.isna(data).all(axis=1)
        data[missing_samples_mask] = np.nan_to_num(data[missing_samples_mask], nan=0)

        # probs = model_options[sets[0].label_type](model)(data.astype(float), **model_params_options[sets[0].label_type])
        data = data.astype("float32")
        probs = model_options[fset.label_type](model)(data, **model_params_options[fset.label_type])

        # probs = output_options[sets[0].label_type](probs, model=model, **output_params_options[sets[0].label_type])
        probs = output_options[fset.label_type](probs, model=model, **output_params_options[fset.label_type])

        # preds = list(map( lambda label: labels_options[sets[0].label_type](label, **labels_params_options[sets[0].label_type]), probs))
        preds = list(map( lambda label: labels_options[fset.label_type](label, **labels_params_options[fset.label_type]), probs))

        if fset.label_type == "binary":
            probs = probs[:, 1]
        elif fset.label_type == "continuous":
            preds = list(np.round(preds, fset.decimals))

        # if sets[0].label_type:
        if fset.label_type == "continuous":
            labels = list(labels)
        elif fset.label_type in ("binary", "discrete"):
            # labels = list(map( lambda label: labels_options[sets[0].label_type](label, **labels_params_options[sets[0].label_type]), labels))
            labels = list(map( lambda label: labels_options[fset.label_type](label, **labels_params_options[fset.label_type]), labels))
        else:
            labels = labels.astype(int).tolist()

        if np.any(missing_samples_mask):
            probs[missing_samples_mask] = np.full(probs[missing_samples_mask].shape, np.nan)
            if fset.label_type != "continuous":
                # preds = pd.array(preds, dtype='Int64')
                preds = pd.DataFrame(preds, dtype='Int64')
            else:
                # preds = np.array(preds)
                preds = pd.DataFrame(preds)
            preds[missing_samples_mask] = None
            # preds = list(preds)
            preds = preds.values.tolist()

        results = pd.DataFrame( dict( ID=ID, label=labels, prediction=preds, probability=probs.tolist() ))
        save_table( results, filename, prediction_path )
        log.info("Inference done")


if __name__ == "__main__":
    pass
