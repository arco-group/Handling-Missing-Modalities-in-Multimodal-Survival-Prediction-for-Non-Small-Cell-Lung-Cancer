import os
import pandas as pd
from CMC_utils import save_load
from tqdm import tqdm

__all__ = ["load_missing_results", "load_experiments_results"]

from CMC_utils.miscellaneous import do_nothing


def load_experiments_results(results_path: str, task: str = "classification", separate_experiments: bool = True, multimodal: bool = False) -> pd.DataFrame:
    if multimodal:
        all_results = pd.DataFrame()
        for folder in os.listdir(results_path):
            if os.path.isdir(os.path.join(results_path, folder)):
                r_path = os.path.join(results_path, folder, "stratifiedkfold_holdout")
                exp_results = load_experiments_results(r_path, task=task, separate_experiments=separate_experiments)
                all_results = pd.concat([all_results, exp_results], axis=0)
    else:
        if task == "classification":
            classes_results = load_missing_results(results_path, result_type="classes", separate_experiments=separate_experiments)  # .reset_index()
            average_results = load_missing_results(results_path, result_type="set", separate_experiments=separate_experiments)  # .reset_index()
            average_results["class"] = "all"
            all_results = pd.concat([average_results, classes_results], axis=0)
        elif task in ("regression", "time2event"):
            # classes_results = None
            average_results = load_missing_results(results_path, result_type="set", balanced=False, separate_experiments=separate_experiments)  # .reset_index()
            average_results["class"] = "all" if task == "regression" else "uncensored"
            all_results = average_results
        else:
            raise ValueError("Task must be either 'classification', 'regression', or 'time2event'")
    return all_results


def load_missing_results(results_path: str, balanced: bool = True, result_type: str = "set", separate_experiments: bool = True, verbose: bool = False) -> pd.DataFrame:
    balance_options = {True: "balanced", False: "unbalanced"}
    result_type_options = dict(set="set_average_performance.csv", classes="classes_average_performance.csv", folds="all_test_performance.csv")
    result_type_args_options = dict(set=dict(header=[0, 1], index_col=0), classes=dict(header=[0, 1], index_col=[0, 1]), folds=dict(header=[0]))
    index_cols_names = dict(set=["set"], classes=["set", "class"], folds=["set", "class", "test_fold", "val_fold"])
    all_results = pd.DataFrame()
    if separate_experiments:
        exps_path = [os.path.join(results_path, folder) for folder in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, folder))]
    else:
        res_path = os.path.join(results_path, "results")
        exps_path = [os.path.join(res_path, folder) for folder in os.listdir(res_path) if os.path.isdir(os.path.join(res_path, folder))]
    print_options = {False: lambda x: do_nothing(x, return_first=True), True: tqdm}
    for exp_path in print_options[verbose](exps_path):
        if separate_experiments:
            cfg = save_load.load_yaml(os.path.join(exp_path, "config.yaml"))
            experiment_name = cfg["experiment_name"]
        else:
            cfg_path = os.path.join(results_path, "configs", os.path.basename(exp_path))
            cfg = save_load.load_yaml(os.path.join(cfg_path, "config.yaml"))
            experiment_name = cfg["experiment_name"] + "_" + cfg["experiment_subname"]  #

        # experiment_name = experiment_name.replace("_classification_with_missing_generation", "").replace("_regression_with_missing_generation", "").replace("_multimodal", "").replace("_fusion", "").replace("multimodal_learner", "naim").replace("no_imputation", "noimputation").replace("cls_token", "clstoken")  # "noimputation")
        experiment_name = experiment_name.replace("_classification_with_missing_generation", "").replace("_regression_with_missing_generation", "").replace("_time2event_with_missing_generation", "")
        experiment_name = experiment_name.replace("_multimodal", "").replace("_fusion", "")
        experiment_name = experiment_name.replace("cls_token", "clstoken").replace("_MCAR_global", "")
        experiment_name = experiment_name.replace("_all", "").replace("_modalities", "").replace("_features", "")
        experiment_name = experiment_name.replace("_normalize", "").replace("_onehotencode", "").replace("_categoricalencode", "")
        experiment_name = experiment_name.replace(cfg["db_name"], "").replace(f"_{cfg['seed']}_", "")

        late_fusion_approaches = cfg.get("late_fusion_approaches", [""])
        if separate_experiments:
            result_path = os.path.join( exp_path, "results" )
        else:
            result_path = exp_path
        train_percentages = [folder for folder in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, folder))]
        for train_percentage in train_percentages:
            train_perc_path = os.path.join( result_path, train_percentage )
            test_percentages = [folder for folder in os.listdir(train_perc_path) if os.path.isdir(os.path.join(train_perc_path, folder)) and folder not in ("balanced", "unbalanced")]

            for test_percentage in test_percentages:

                for approach in late_fusion_approaches:

                    if approach:
                        test_perc_path = os.path.join( train_perc_path, test_percentage, balance_options[balanced], "test", approach, result_type_options[result_type] )
                        # exp_name = "_".join(experiment_name.split('_')[:-1]) + f"{approach.replace('_', '')}" + f"_{experiment_name.split('_')[-1]}"
                        exp_name = experiment_name + f"{approach.replace('_', '')}"
                    else:
                        test_perc_path = os.path.join( train_perc_path, test_percentage, balance_options[balanced], "test", result_type_options[result_type] )
                        exp_name = experiment_name

                    results = save_load.load_table(test_perc_path, **result_type_args_options[result_type])
                    if result_type != "folds":
                        joined_cols = ["_".join(col) for col in results.columns.to_list()]
                        results.columns = joined_cols
                        results = results.rename_axis(index_cols_names[result_type], axis=0).reset_index()

                    results = results.assign(experiment=exp_name, train_percentage=int(train_percentage), test_percentage=int(test_percentage))
                    results = results.assign(db=cfg["db_name"], model=cfg["model_name"], imputer=cfg["preprocessing"]["imputer"]["method"])
                    results = results.assign(missing_strategy=cfg["multimodal_missing_generation"])
                    fusion_strategies = {"multimodal_early_fusion": "early", "multimodal_late_fusion": "late", "multimodal_joint_fusion": "joint", "missing": "-"}
                    results = results.assign(fusion_strategy=fusion_strategies[cfg["pipeline"]] + approach.replace('_', ''))

                    all_results = pd.concat([all_results, results])
    all_results = all_results.drop("set", axis=1).set_index(["experiment", "db", "model", "imputer", "missing_strategy", "fusion_strategy", "train_percentage", "test_percentage"] + index_cols_names[result_type][2:]).sort_index()
    return all_results


if __name__ == "__main__":
    pass
