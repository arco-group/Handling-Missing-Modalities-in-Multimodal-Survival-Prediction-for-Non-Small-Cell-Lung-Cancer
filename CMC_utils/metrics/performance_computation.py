import logging
import numpy as np
import pandas as pd
from hydra.utils import call
from omegaconf import OmegaConf
from CMC_utils.miscellaneous import recursive_cfg_substitute, recursive_cfg_search
from sklearn.utils.class_weight import compute_sample_weight

log = logging.getLogger(__name__)

__all__ = ["metrics_computation", "metrics_computation_df"]


def metrics_computation_df(data, **kwargs) -> pd.DataFrame:
    """
    Compute the metrics from a dataframe
    Parameters
    ----------
    data : pd.DataFrame
    kwargs : dict

    Returns
    -------
    pd.DataFrame
        metrics
    """
    labels = np.vstack(data.label.values)
    predictions = np.vstack(data.prediction.values)
    probabilities = np.vstack(data.probability.values)
    return metrics_computation(labels, predictions, probabilities, **kwargs)

def metrics_computation(label, prediction, probability, metrics: dict, task: str, round_number=4, verbose: bool = True, use_weights: bool = False, **kwargs) -> pd.DataFrame:
    """
    Compute the metrics
    Parameters
    ----------
    label : np.ndarray
    prediction : np.ndarray
    probability : np.ndarray
    metrics : dict
    round_number : int
    verbose : bool
    use_weights : bool
    kwargs : dict

    Returns
    -------
    pd.DataFrame
    """
    classes_map = {}
    classes_map_inverted = {}
    if task in ("classification", "survival_analysis"):
        classes = kwargs.get("classes", np.sort(np.unique(label)))
        classes_map = {int(v): str(k) for v, k in enumerate(classes)}
        classes_map_inverted = {v: k for k, v in classes_map.items()}
    elif task == "multilabelclassification":
        classes = kwargs.get("classes", np.arange(label.shape[1]))
        # classes = [(cl, st) for cl in classes for st in ["negative", "positive"]]
        classes_map = {int(v): str(k) for v, k in enumerate(["negative", "positive"])}
        classes_map_inverted = {v: k for k, v in classes_map.items()}

    if label.dtype not in (int, float, "float32"):
        vectorized_func = np.vectorize(lambda x: classes_map_inverted.get(str(x), None))
        label = vectorized_func(label)
        prediction = vectorized_func(prediction)

    if use_weights and task in ("classification",):
        sample_weight = compute_sample_weight(class_weight='balanced', y=label)
        metrics = OmegaConf.create(metrics) if type(metrics) == dict else metrics
        metrics = OmegaConf.to_object(metrics)
        metrics = recursive_cfg_substitute(metrics, dict(sample_weight=sample_weight))
    if use_weights and task in ("multilabelclassification",):
        sample_weight = []
        for i in range(label.shape[1]):
            cl_i_map = label[:, i] != -100
            scores_i = np.full(cl_i_map.shape, np.nan)
            scores_i[cl_i_map] = compute_sample_weight(class_weight="balanced", y=label[cl_i_map, i])
            sample_weight.append(scores_i)
        sample_weight = np.vstack(sample_weight).T
        # sample_weight = np.nanprod(sample_weight, axis=1)

    average = recursive_cfg_search(metrics, "average")[0]
    if task == "multilabelclassification" and average == "micro":
        if use_weights:
            sample_weight = np.reshape(sample_weight, (-1, 1))
        #     sample_weight = np.vstack([sample_weight] * label.shape[1]).T
        #     sample_weight = np.reshape(sample_weight, (-1, 1))

        label = np.reshape(label, (-1, 1))
        prediction = np.reshape(prediction, (-1, 1))
        probability = np.reshape(probability, (-1, 1))

    if average in ("micro", "macro", "weighted"):
        classes = [f"{average}_average"]

    is_binary = len(probability[0]) == 2 and np.round(sum(probability[0]), 0) == 1

    performance = dict()
    for metric in metrics.values():
        if is_binary and metric["name"] == "auc":
            prob = probability[:, 1]
        else:
            prob = probability
        if task != "multilabelclassification":
            perf = call( metric["init"], y_true=label, y_pred=prediction, y_score=prob, labels=list(classes_map.keys()))
            performance[metric["name"]] = np.round(perf, round_number)
        else:
            perf = []

            if metric["name"] not in ("auc", "mcc"):
                metric = OmegaConf.create(metric) if type(metric) == dict else metric
                metric = recursive_cfg_substitute(metric, dict(average="binary", pos_label=1))

            for i in range(label.shape[1]):
                mask = label[:, i] != -100
                label_i = label[mask, i]
                prediction_i = prediction[mask, i]
                probability_i = prob[mask, i]
                if use_weights:
                    metric_i = OmegaConf.create(metric) if type(metric) == dict else metric
                    metric_i = OmegaConf.to_object(metric_i)
                    # metric_i = recursive_cfg_substitute(metric_i, dict(sample_weight=sample_weight[mask].squeeze()))
                    # if average != "micro":
                    #     sample_weight_i = compute_sample_weight(class_weight="balanced", y=label_i)
                    #     metric_i = recursive_cfg_substitute(metric_i, dict(sample_weight=sample_weight_i.squeeze()))
                    # else:
                    metric_i = recursive_cfg_substitute(metric_i, dict(sample_weight=sample_weight[mask, i].squeeze()))

                else:
                    metric_i = metric

                if len(np.unique(label_i)) == 1:
                    perf_i = np.nan
                else:
                    perf_i = call(metric_i["init"], y_true=label_i, y_pred=prediction_i, y_score=probability_i)  # , labels=list(classes_map.keys()))

                if type(perf_i) not in (float, int):
                    perf_i = np.squeeze(perf_i)
                perf.append(perf_i)
            performance[metric["name"]] = np.round(np.array(perf), round_number)
    if task in ("classification", "multilabelclassification", "survival_analysis"):
        performance = pd.DataFrame(performance, index=classes).rename_axis("class", axis=0)
    else:
        performance = pd.DataFrame(performance, index=[0])

    if verbose:
        log.info("Performance computed")

    return performance


if __name__ == "__main__":
    pass
