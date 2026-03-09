import os
import re
import copy
import logging
import numpy as np
import pandas as pd
from typing import List, Union
from omegaconf import DictConfig, ListConfig
from sklearn.utils.class_weight import compute_sample_weight
from CMC_utils import save_load
from CMC_utils.miscellaneous import recursive_cfg_substitute
from CMC_utils.paths import get_files_with_extension
from .performance_computation import *
from .late_fusion_computation import *


log = logging.getLogger(__name__)

__all__ = ["compute_performance", "compute_missing_performance", "compute_late_fusion_performance", "compute_late_fusion_missing_performance"]


def compute_late_fusion_missing_performance(classes: List[str], prediction_path: str, results_path: str, task: str, metrics: Union[dict, DictConfig], missing_percentages: List[float], test_missing_percentages: List[float], datasets_names: List[str], late_fusion_approaches: List[str], decimals: int = 2) -> None:
    missing_percentages = [int(miss_perc * 100) for miss_perc in missing_percentages]

    if isinstance(test_missing_percentages[0], (list, ListConfig)):
        test_missing_percentages = [ "-".join([str(int(modality_percentage * 100)) for modality_percentage in modalities_missing_percentages]) for modalities_missing_percentages in test_missing_percentages]
    else:
        test_missing_percentages = [int(miss_perc * 100) for miss_perc in test_missing_percentages]

    for train_missing_percentage in missing_percentages:
        preds_path = os.path.join( prediction_path, str(train_missing_percentage) )
        res_path = os.path.join( results_path, str(train_missing_percentage) )

        compute_late_fusion_performance(classes=classes, prediction_path=preds_path, results_path=res_path, metrics=metrics, datasets_names=datasets_names, late_fusion_approaches=late_fusion_approaches, task=task, decimals=decimals)

        for test_missing_percentage in test_missing_percentages:
            test_preds_path = os.path.join( preds_path, str(test_missing_percentage) )
            test_res_path = os.path.join(res_path, str(test_missing_percentage))

            compute_late_fusion_performance(classes=classes, prediction_path=test_preds_path, results_path=test_res_path, metrics=metrics, datasets_names=datasets_names, late_fusion_approaches=late_fusion_approaches, task=task, decimals=decimals)


def compute_late_fusion_performance(classes: List[str], prediction_path: str, results_path: str, task: str, metrics: Union[dict, DictConfig], datasets_names: List[str], late_fusion_approaches: List[str], decimals: int = 2) -> None:
    datasets_folders = [os.path.join(prediction_path, dataset_name) for dataset_name in datasets_names]

    datasets_prediction_files = []
    for dataset_folder in datasets_folders:
        prediction_files = get_files_with_extension(dataset_folder, "csv")
        prediction_files = sorted(prediction_files, key=lambda file_path: int(re.findall(r"\d+_(\d+)_\w+\.csv\Z", file_path)[0]))
        prediction_files = sorted(prediction_files, key=lambda file_path: int(re.findall(r"(\d+)_\d+_\w+\.csv\Z", file_path)[0]))
        datasets_prediction_files.append(prediction_files)

    micro_metrics = recursive_cfg_substitute(copy.deepcopy(metrics), {"average": "micro"})
    macro_metrics = recursive_cfg_substitute(copy.deepcopy(metrics), {"average": "macro"})
    weighted_metrics = recursive_cfg_substitute(copy.deepcopy(metrics), {"average": "weighted"})

    for fset in ["train", "val", "test"]:
        set_files = [[prediction_file for prediction_file in dataset_prediction_files if prediction_file.endswith(f"{fset}.csv")] for dataset_prediction_files in datasets_prediction_files]

        performance, performance_balanced = {late_fusion_approach: pd.DataFrame() for late_fusion_approach in late_fusion_approaches}, {late_fusion_approach: pd.DataFrame() for late_fusion_approach in late_fusion_approaches}
        micro_performance, micro_performance_balanced = {late_fusion_approach: pd.DataFrame() for late_fusion_approach in late_fusion_approaches}, {late_fusion_approach: pd.DataFrame() for late_fusion_approach in late_fusion_approaches}
        macro_performance, macro_performance_balanced = {late_fusion_approach: pd.DataFrame() for late_fusion_approach in late_fusion_approaches}, {late_fusion_approach: pd.DataFrame() for late_fusion_approach in late_fusion_approaches}
        weighted_performance, weighted_performance_balanced = {late_fusion_approach: pd.DataFrame() for late_fusion_approach in late_fusion_approaches}, {late_fusion_approach: pd.DataFrame() for late_fusion_approach in late_fusion_approaches}
        for files in zip(*set_files):
            assert all(os.path.basename(file) == os.path.basename(files[0]) for file in files), "Files are not aligned"

            test_fold, val_fold = [int(fold_num) for fold_num in re.findall(r"(\d+)_(\d+)_\w+\.csv\Z", files[0])[0]]

            decision_profile = get_decision_profile(files)
            for late_fusion_approach in late_fusion_approaches:
                fold_preds = apply_late_fusion_approach(decision_profile, late_fusion_approach=late_fusion_approach, task=task, classes=classes)

                if task in ("classification",):
                    fold_performance = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=False, verbose=0).mul(100).round(2).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    micro_fold_performance = metrics_computation_df(fold_preds, task=task, metrics=micro_metrics, classes=classes, use_weights=False, verbose=0).mul(100).round(2).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    macro_fold_performance = metrics_computation_df(fold_preds, task=task, metrics=macro_metrics, classes=classes, use_weights=False, verbose=0).mul(100).round(2).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    weighted_fold_performance = metrics_computation_df(fold_preds, task=task, metrics=weighted_metrics, classes=classes, use_weights=False, verbose=0).mul(100).round(2).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=True, verbose=0).mul(100).round(2).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    micro_fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=micro_metrics, classes=classes, use_weights=True, verbose=0).mul(100).round(2).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    macro_fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=macro_metrics, classes=classes, use_weights=True, verbose=0).mul(100).round(2).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    weighted_fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=weighted_metrics, classes=classes, use_weights=True, verbose=0).mul(100).round(2).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                elif task in ("multilabelclassification",):
                    fold_performance = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=False, verbose=0).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    micro_fold_performance = metrics_computation_df(fold_preds, task=task, metrics=micro_metrics, classes=classes, use_weights=False, verbose=0).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    macro_fold_performance = fold_performance.set_index(["class", "set", "test_fold", "val_fold"]).mean().round(2).rename("macro_average").to_frame().T.assign(set=fset, test_fold=test_fold,val_fold=val_fold).rename_axis("class", axis=0).reset_index()

                    classes_weights = (np.vstack(fold_preds.label.values) == 1).sum(axis=0)
                    weighted_fold_performance = pd.DataFrame(
                        np.average(fold_performance.set_index(["class", "set", "test_fold", "val_fold"]).values, weights=classes_weights, axis=0), columns=["weighted_average"]).T.reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold).round(2)
                    weighted_fold_performance.columns = fold_performance.columns

                    fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=True, verbose=0).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    micro_fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=micro_metrics, classes=classes, use_weights=True, verbose=0).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    macro_fold_performance_balanced = fold_performance_balanced.set_index(["class", "set", "test_fold", "val_fold"]).mean().round(2).rename("macro_average").to_frame().T.assign(set=fset, test_fold=test_fold,val_fold=val_fold).rename_axis("class",axis=0).reset_index()
                    sample_weight = []
                    labels = np.vstack(fold_preds.label.copy().values)
                    for i in range(labels.shape[1]):
                        cl_i_map = labels[:, i] != -100
                        scores_i = np.zeros(cl_i_map.shape)
                        scores_i[cl_i_map] = compute_sample_weight(class_weight="balanced", y=labels[cl_i_map, i])
                        sample_weight.append(scores_i)
                    sample_weight = np.vstack(sample_weight).T
                    labels = (labels == 1).astype(float)
                    classes_weights = np.sum(np.multiply(labels, sample_weight), axis=0)
                    weighted_fold_performance_balanced = pd.DataFrame(np.average(fold_performance_balanced.set_index(["class", "set", "test_fold", "val_fold"]).values, weights=classes_weights, axis=0), columns=["weighted_average"]).T.reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold).round(2)
                    weighted_fold_performance_balanced.columns = fold_performance_balanced.columns
                else:
                    fold_performance = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=False, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    micro_fold_performance = pd.DataFrame()  # metrics_computation_df(fold_preds, task=task, metrics=micro_metrics, classes=classes, use_weights=False, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    macro_fold_performance = pd.DataFrame()  # metrics_computation_df(fold_preds, task=task, metrics=macro_metrics, classes=classes, use_weights=False, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    weighted_fold_performance = pd.DataFrame()  # metrics_computation_df(fold_preds, task=task, metrics=weighted_metrics, classes=classes, use_weights=False, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=True, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    micro_fold_performance_balanced = pd.DataFrame()  # metrics_computation_df(fold_preds, task=task, metrics=micro_metrics, classes=classes, use_weights=True, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    macro_fold_performance_balanced = pd.DataFrame()  # metrics_computation_df(fold_preds, task=task, metrics=macro_metrics, classes=classes, use_weights=True, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    weighted_fold_performance_balanced = pd.DataFrame()  # metrics_computation_df(fold_preds, task=task, metrics=weighted_metrics, classes=classes, use_weights=True, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)

                performance[late_fusion_approach] = pd.concat([performance[late_fusion_approach], fold_performance], axis=0, ignore_index=True)
                micro_performance[late_fusion_approach] = pd.concat([micro_performance[late_fusion_approach], micro_fold_performance], axis=0, ignore_index=True)
                macro_performance[late_fusion_approach] = pd.concat([macro_performance[late_fusion_approach], macro_fold_performance], axis=0, ignore_index=True)
                weighted_performance[late_fusion_approach] = pd.concat([weighted_performance[late_fusion_approach], weighted_fold_performance], axis=0, ignore_index=True)
                performance_balanced[late_fusion_approach] = pd.concat([performance_balanced[late_fusion_approach], fold_performance_balanced], axis=0, ignore_index=True)
                micro_performance_balanced[late_fusion_approach] = pd.concat([micro_performance_balanced[late_fusion_approach], micro_fold_performance_balanced], axis=0, ignore_index=True)
                macro_performance_balanced[late_fusion_approach] = pd.concat([macro_performance_balanced[late_fusion_approach], macro_fold_performance_balanced], axis=0, ignore_index=True)
                weighted_performance_balanced[late_fusion_approach] = pd.concat([weighted_performance_balanced[late_fusion_approach], weighted_fold_performance_balanced], axis=0, ignore_index=True)

        for late_fusion_approach in late_fusion_approaches:
            perf = performance[late_fusion_approach]
            if not perf.empty:
                unbalanced_path = os.path.join(results_path, "unbalanced", fset, late_fusion_approach)
                if not os.path.exists(unbalanced_path):
                    os.makedirs(unbalanced_path)
                perf.loc[ :,'class'] = 'uncensored'
                compute_performance_statistics(perf, unbalanced_path, task, decimals=decimals)
            micro_perf = micro_performance[late_fusion_approach]
            macro_perf = macro_performance[late_fusion_approach]
            weighted_perf = weighted_performance[late_fusion_approach]
            if not micro_perf.empty or not macro_perf.empty or not weighted_perf.empty:
                unbalanced_path = os.path.join(results_path, "unbalanced", fset, late_fusion_approach)
                if not os.path.exists(unbalanced_path):
                    os.makedirs(unbalanced_path)
                average_performance = pd.concat([micro_perf, macro_perf, weighted_perf], axis=0, ignore_index=True)
                compute_performance_statistics(average_performance, unbalanced_path, task, decimals=decimals, averages=True)

            perf_bal = performance_balanced[late_fusion_approach]
            if not perf_bal.empty and task in ("classification", "multilabelclassification"):
                balanced_path = os.path.join(results_path, "balanced", fset, late_fusion_approach)
                if not os.path.exists(balanced_path):
                    os.makedirs(balanced_path)
                compute_performance_statistics(perf_bal, balanced_path, task, decimals=decimals)
            micro_perf_bal = micro_performance_balanced[late_fusion_approach]
            macro_perf_bal = macro_performance_balanced[late_fusion_approach]
            weighted_perf_bal = weighted_performance_balanced[late_fusion_approach]
            if (not micro_perf_bal.empty or not macro_perf_bal.empty or not weighted_perf_bal.empty) and task in ("classification", "multilabelclassification"):
                balanced_path = os.path.join(results_path, "balanced", fset, late_fusion_approach)
                if not os.path.exists(balanced_path):
                    os.makedirs(balanced_path)
                average_performance_balanced = pd.concat([micro_perf_bal, macro_perf_bal, weighted_perf_bal], axis=0,ignore_index=True)
                compute_performance_statistics(average_performance_balanced, balanced_path, task, decimals=decimals, averages=True)


def compute_missing_performance(classes: List[str], prediction_path: str, results_path: str, task: str, metrics: Union[dict, DictConfig], missing_percentages: List[float], test_missing_percentages: List[float], **kwargs) -> None:
    """
    Compute the performance for different missing percentages
    Parameters
    ----------
    classes : List[str]
    prediction_path : str
    results_path : str
    metrics : Union[dict, DictConfig]
    missing_percentages : List[float]

    Returns
    -------
    None
    """
    missing_percentages = [int(miss_perc * 100) for miss_perc in missing_percentages]

    if isinstance(test_missing_percentages[0], (list, ListConfig)):
        test_missing_percentages = [ "-".join([str(int(modality_percentage * 100)) for modality_percentage in modalities_missing_percentages]) for modalities_missing_percentages in test_missing_percentages]
    else:
        test_missing_percentages = [int(miss_perc * 100) for miss_perc in test_missing_percentages]

    for train_missing_percentage in missing_percentages:
        preds_path = os.path.join( prediction_path, str(train_missing_percentage) )
        res_path = os.path.join( results_path, str(train_missing_percentage) )

        compute_performance(classes=classes, prediction_path=preds_path, results_path=res_path, task=task, metrics=metrics, **kwargs)

        for test_missing_percentage in test_missing_percentages:
            test_preds_path = os.path.join( preds_path, str(test_missing_percentage) )
            test_res_path = os.path.join(res_path, str(test_missing_percentage))
            compute_performance(classes=classes, prediction_path=test_preds_path, results_path=test_res_path, task=task, metrics=metrics, **kwargs)


def compute_performance(classes: List[str], prediction_path: str, results_path: str, task: str, metrics: Union[dict, DictConfig], decimals: int = 2, **_) -> None:
    """
    Compute the performance for a given set of predictions
    Parameters
    ----------
    classes : List[str]
    prediction_path : str
    results_path : str
    metrics : Union[dict, DictConfig]

    Returns
    -------
    None
    """
    prediction_files = get_files_with_extension(prediction_path, "csv")
    print("PREDICTION PATH: ", prediction_path, prediction_files)
    prediction_files = get_files_with_extension(prediction_path, "csv")
    prediction_files = [p for p in prediction_files if 'pred_aggregated_score' not in p]

    prediction_files = sorted(prediction_files, key=lambda file_path: int(re.findall(r"\d+_(\d+)_\w+\.csv\Z", file_path)[0]))
    prediction_files = sorted(prediction_files, key=lambda file_path: int(re.findall(r"(\d+)_\d+_\w+\.csv\Z", file_path)[0]))


    micro_metrics = recursive_cfg_substitute(copy.deepcopy(metrics), {"average": "micro"})
    macro_metrics = recursive_cfg_substitute(copy.deepcopy(metrics), {"average": "macro"})
    weighted_metrics = recursive_cfg_substitute(copy.deepcopy(metrics), {"average": "weighted"})

    for fset in ["train", "val", "test"]:
        set_files = [file for file in prediction_files if file.endswith(f"{fset}.csv")]

        performance, performance_balanced = pd.DataFrame(), pd.DataFrame()
        micro_performance, micro_performance_balanced = pd.DataFrame(), pd.DataFrame()
        macro_performance, macro_performance_balanced = pd.DataFrame(), pd.DataFrame()
        weighted_performance, weighted_performance_balanced = pd.DataFrame(), pd.DataFrame()

        for file in set_files:
            test_fold, val_fold = [ int(fold_num) for fold_num in re.findall(r"(\d+)_(\d+)_\w+\.csv\Z", file)[0] ]

            fold_preds = save_load.load_params_table(file).set_index("ID")
            if task in ("classification", "survival_analysis"):
                fold_performance = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=False, verbose=0, prediction_path=prediction_path).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                micro_fold_performance = metrics_computation_df(fold_preds, task=task, metrics=micro_metrics, classes=classes, use_weights=False, verbose=0, prediction_path=prediction_path).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                macro_fold_performance = metrics_computation_df(fold_preds, task=task, metrics=macro_metrics, classes=classes, use_weights=False, verbose=0, prediction_path=prediction_path).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                weighted_fold_performance = metrics_computation_df(fold_preds, task=task, metrics=weighted_metrics, classes=classes, use_weights=False, verbose=0, prediction_path=prediction_path).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=True, verbose=0, prediction_path=prediction_path).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                micro_fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=micro_metrics, classes=classes, use_weights=True, verbose=0, prediction_path=prediction_path).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                macro_fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=macro_metrics, classes=classes, use_weights=True, verbose=0, prediction_path=prediction_path).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                weighted_fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=weighted_metrics, classes=classes, use_weights=True, verbose=0, prediction_path=prediction_path).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)

                print(weighted_fold_performance_balanced)
                print(weighted_fold_performance)


            elif task in ("multilabelclassification",):
                fold_performance = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=False, verbose=0).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                micro_fold_performance = metrics_computation_df(fold_preds, task=task, metrics=micro_metrics, classes=classes, use_weights=False, verbose=0).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                macro_fold_performance = fold_performance.set_index(["class", "set", "test_fold", "val_fold"]).mean().round(2).rename("macro_average").to_frame().T.assign(set=fset, test_fold=test_fold, val_fold=val_fold).rename_axis("class", axis=0).reset_index()

                classes_weights = (np.vstack(fold_preds.label.values) == 1).sum(axis=0)
                weighted_fold_performance = pd.DataFrame(np.average(fold_performance.set_index(["class", "set", "test_fold", "val_fold"]).values, weights=classes_weights, axis=0), columns=["weighted_average"]).T.reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold).round(2)
                weighted_fold_performance.columns = fold_performance.columns

                fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=True, verbose=0).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                micro_fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=micro_metrics, classes=classes, use_weights=True, verbose=0).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                macro_fold_performance_balanced = fold_performance_balanced.set_index(["class", "set", "test_fold", "val_fold"]).mean().round(2).rename("macro_average").to_frame().T.assign(set=fset, test_fold=test_fold, val_fold=val_fold).rename_axis("class", axis=0).reset_index()
                sample_weight = []
                labels = np.vstack(fold_preds.label.copy().values)
                for i in range(labels.shape[1]):
                    cl_i_map = labels[:, i] != -100
                    scores_i = np.zeros(cl_i_map.shape)
                    scores_i[cl_i_map] = compute_sample_weight(class_weight="balanced", y=labels[cl_i_map, i])
                    sample_weight.append(scores_i)
                sample_weight = np.vstack(sample_weight).T
                labels = (labels == 1).astype(float)
                classes_weights = np.sum(np.multiply(labels, sample_weight), axis=0)
                weighted_fold_performance_balanced = pd.DataFrame(np.average(fold_performance_balanced.set_index(["class", "set", "test_fold", "val_fold"]).values, weights=classes_weights, axis=0), columns=["weighted_average"]).T.reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold).round(2)
                weighted_fold_performance_balanced.columns = fold_performance_balanced.columns

            else:
                fold_performance = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=False, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                micro_fold_performance = pd.DataFrame()  # metrics_computation_df(fold_preds, task=task, metrics=micro_metrics, classes=classes, use_weights=False, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                macro_fold_performance = pd.DataFrame()  # metrics_computation_df(fold_preds, task=task, metrics=macro_metrics, classes=classes, use_weights=False, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                weighted_fold_performance = pd.DataFrame()  # metrics_computation_df(fold_preds, task=task, metrics=weighted_metrics, classes=classes, use_weights=False, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=True, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                micro_fold_performance_balanced = pd.DataFrame()  # metrics_computation_df(fold_preds, task=task, metrics=micro_metrics, classes=classes, use_weights=True, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                macro_fold_performance_balanced = pd.DataFrame()  # metrics_computation_df(fold_preds, task=task, metrics=macro_metrics, classes=classes, use_weights=True, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                weighted_fold_performance_balanced = pd.DataFrame()  # metrics_computation_df(fold_preds, task=task, metrics=weighted_metrics, classes=classes, use_weights=True, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)



            performance = pd.concat([ performance, fold_performance ], axis=0, ignore_index=True )
            performance_balanced = pd.concat([ performance_balanced, fold_performance_balanced ], axis=0, ignore_index=True )
            micro_performance = pd.concat([micro_performance, micro_fold_performance], axis=0, ignore_index=True)
            micro_performance_balanced = pd.concat([micro_performance_balanced, micro_fold_performance_balanced], axis=0, ignore_index=True)
            macro_performance = pd.concat([macro_performance, macro_fold_performance], axis=0, ignore_index=True)
            macro_performance_balanced = pd.concat([macro_performance_balanced, macro_fold_performance_balanced], axis=0, ignore_index=True)
            weighted_performance = pd.concat([weighted_performance, weighted_fold_performance], axis=0, ignore_index=True)
            weighted_performance_balanced = pd.concat([weighted_performance_balanced, weighted_fold_performance_balanced], axis=0, ignore_index=True)

        print(performance)

        if not performance.empty:
            unbalanced_path = os.path.join(results_path, "unbalanced", fset)
            if not os.path.exists(unbalanced_path):
                os.makedirs(unbalanced_path)
            compute_performance_statistics(performance, unbalanced_path, task, decimals=decimals)
        if not micro_performance.empty or not macro_performance.empty or not weighted_performance.empty:
            unbalanced_path = os.path.join(results_path, "unbalanced", fset)
            if not os.path.exists(unbalanced_path):
                os.makedirs(unbalanced_path)
            average_performance = pd.concat([micro_performance, macro_performance, weighted_performance], axis=0, ignore_index=True)
            compute_performance_statistics(average_performance, unbalanced_path, task, decimals=decimals, averages=True)

        if not performance_balanced.empty and task in ("classification", "multilabelclassification"):
            balanced_path = os.path.join(results_path, "balanced", fset)
            if not os.path.exists(balanced_path):
                os.makedirs(balanced_path)
            compute_performance_statistics(performance_balanced, balanced_path, task, decimals=decimals)
        if (not micro_performance_balanced.empty or not macro_performance_balanced.empty or not weighted_performance_balanced.empty) and task in ("classification", "multilabelclassification"):
            balanced_path = os.path.join(results_path, "balanced", fset)
            if not os.path.exists(balanced_path):
                os.makedirs(balanced_path)
            average_performance_balanced = pd.concat([micro_performance_balanced, macro_performance_balanced, weighted_performance_balanced], axis=0, ignore_index=True)
            compute_performance_statistics(average_performance_balanced, balanced_path, task, decimals=decimals, averages=True)


def compute_performance_statistics(performance, path, task, decimals: int = 2, averages=False) -> None:
    """
    Compute the performance statistics
    Parameters
    ----------
    performance : pd.DataFrame
    path : str

    Returns
    -------
   None
    """
    average_text = "averages_" if averages else ""
    save_load.save_table(performance, f"{average_text}all_test_performance.csv", path, index=False)
    if task in ("classification", "multilabelclassification", "survival_analysis"):
        mean_performance = performance.drop(["test_fold", "val_fold"], axis=1).groupby( by=["set", "class"] ).agg( [ "mean", "std", "min", "max" ] ).round(decimals)
        save_load.save_table( mean_performance.reset_index(), f"{average_text}classes_average_performance.csv", path, index=False )

    if not averages:
        average_performance_drop = ["test_fold", "val_fold", "class"] if task in ("classification", "multilabelclassification", "survival_analysis") else ["test_fold", "val_fold"]
        average_performance = performance.drop(average_performance_drop, axis=1).groupby(by=["set"]).agg(["mean", "std", "min", "max"]).round(decimals)
        save_load.save_table( average_performance.reset_index(), f"{average_text}set_average_performance.csv", path, index=False )
    log.info("Average performance computed")


if __name__ == "__main__":
    pass
