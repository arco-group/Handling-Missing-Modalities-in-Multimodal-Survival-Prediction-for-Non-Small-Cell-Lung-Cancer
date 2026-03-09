import os
import logging

from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf

from .routines import initialize_experiment, check_fold_to_do
from CMC_utils import miscellaneous, save_load, preprocessing, metrics, cross_validation as cv

log = logging.getLogger(__name__)

__all__ = ["supervised_tabular_missing_main"]


def supervised_tabular_missing_main(cfg: DictConfig) -> None:
    log.info("Supervised tabular missing main started")

    initialize_experiment(cfg)

    dataset = instantiate(cfg.db, model_label_types=cfg.model.label_types, model_framework=cfg.model.framework, preprocessing_params=cfg.preprocessing, _recursive_=False)

    cv.set_cross_validation(dataset.info_for_cv, cfg.paths.cv, test_params=cfg.test_cv, val_params=cfg.val_cv)

    for test_fold, val_fold, train, val, test, last_val_fold in cv.get_cross_validation(cfg.paths.cv, "train", "val", "test"):
        if not check_fold_to_do(test_fold, cfg.get("fold_to_do", None)):
            continue

        train_data, train_labels, train_cv_info, val_data, val_labels, _ = cv.get_sets_with_idx(dataset.data, train, val, labels=dataset.labels_for_model, cv_info=dataset.info_for_cv)

        for train_missing_fraction in cfg.missing_percentages:
            train_missing_percentage = int(100 * train_missing_fraction)

            preprocessing_paths = {key: os.path.join(value, str(train_missing_percentage)) for key, value in cfg.paths.preprocessing.items()}
            save_load.create_directories(**preprocessing_paths)

            train_data_missing, _ = preprocessing.generate_missing( train_data, dataset_name=dataset.name, test_fold=test_fold, val_fold=val_fold, fset="train", save_path=cfg.paths.missing_masks, **miscellaneous.recursive_cfg_substitute(cfg.missing_generation.train, {"missing_fraction": train_missing_fraction}) )
            val_data_missing, _ = preprocessing.generate_missing( val_data, dataset_name=dataset.name, test_fold=test_fold, val_fold=val_fold, fset="val", save_path=cfg.paths.missing_masks, **miscellaneous.recursive_cfg_substitute(cfg.missing_generation.val, {"missing_fraction": train_missing_fraction}) )

            train_set = instantiate(cfg.db.dataset_class, train_data_missing, train_labels, cfg.db.task, "train", cv_info=train_cv_info, preprocessing_params=dataset.preprocessing_params, preprocessing_paths=preprocessing_paths, test_fold=test_fold, val_fold=val_fold, augmentation=cfg.model_name.startswith(("naim", "survival_naim")), missing_augmentation_probability=cfg.get("missing_augmentation_probability", 0.5), normalize_target=cfg.db.get("normalize_target", False), decimals=cfg.db.get("decimals", 2), image_caching=cfg.get("image_caching", True), _recursive_=False)
            val_set = instantiate(cfg.db.dataset_class, val_data_missing, val_labels, cfg.db.task, "val", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=preprocessing_paths, test_fold=test_fold, val_fold=val_fold, normalize_target=cfg.db.get("normalize_target", False), decimals=cfg.db.get("decimals", 2), image_caching=cfg.get("image_caching", True), _recursive_=False)

            model_params = call(cfg.model.set_params_function, OmegaConf.to_object(cfg.model), preprocessing_params=dataset.preprocessing_params, train_set=train_set, val_set=val_set, _recursive_=False)
            model = instantiate(model_params["init_params"], _recursive_=False)

            model_path = os.path.join(cfg.paths.model, str(train_missing_percentage))
            train_params = OmegaConf.to_object(cfg.train)
            train_params["set_metrics"] = metrics.set_metrics_params(train_params.get("set_metrics", {}), preprocessing_params=dataset.preprocessing_params)
            call(model_params["train_function"], model, train_set, model_params, model_path, val_set=val_set, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

            prediction_path = os.path.join(cfg.paths.predictions, str(train_missing_percentage))
            save_load.create_directory(prediction_path)
            train_set.set_augmentation(False)
            call(model_params["test_function"], train_set, val_set, model_params=model_params, model_path=model_path, prediction_path=prediction_path, classes=dataset.classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

            test_data, test_labels = cv.get_sets_with_idx(dataset.data, test, labels=dataset.labels_for_model)

            for test_missing_fraction in cfg.get("test_missing_percentages", cfg.missing_percentages):
                test_missing_percentage = int(100 * test_missing_fraction)
                log.info(f"Test missing percentage: {test_missing_percentage}")

                test_data_missing, _ = preprocessing.generate_missing( test_data, dataset_name=dataset.name, test_fold=test_fold, val_fold=val_fold, fset="test", save_path=cfg.paths.missing_masks, **miscellaneous.recursive_cfg_substitute(cfg.missing_generation.test, {"missing_fraction": test_missing_fraction}) )
                test_set = instantiate(cfg.db.dataset_class, test_data_missing, test_labels, cfg.db.task, "test", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=preprocessing_paths, test_fold=test_fold, val_fold=val_fold, normalize_target=cfg.db.get("normalize_target", False), decimals=cfg.db.get("decimals", 2), _recursive_=False)

                test_prediction_path = os.path.join(prediction_path, str(test_missing_percentage))
                save_load.create_directory(test_prediction_path)
                call(model_params["test_function"], test_set, model_params=model_params, model_path=model_path, prediction_path=test_prediction_path, classes=dataset.classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

                del test_set, test_data_missing

            del train_set, val_set, train_data_missing, val_data_missing, test_data, test_labels, model_params, train_params

        del train_data, train_labels, val_data, val_labels

    performance_metrics = metrics.set_metrics_params(cfg.performance_metrics, preprocessing_params=dataset.preprocessing_params)
    metrics.compute_missing_performance(dataset.classes, cfg.paths.predictions, cfg.paths.results, cfg.db.task, performance_metrics, cfg.missing_percentages, cfg.get("test_missing_percentages", cfg.missing_percentages), decimals=max(cfg.db.get("decimals", 2), 2))

    log.info("Job finished")


if __name__ == "__main__":
    pass
