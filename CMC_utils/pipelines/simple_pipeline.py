import logging
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf
from .routines import initialize_experiment, check_fold_to_do
from CMC_utils import metrics
from CMC_utils import cross_validation as cv

log = logging.getLogger(__name__)

__all__ = ["supervised_learning_main"]


def supervised_learning_main(cfg: DictConfig) -> None:
    log.info(f"Supervised main started")

    initialize_experiment(cfg)

    dataset = instantiate(cfg.db, model_label_types=cfg.model.label_types, model_framework=cfg.model.framework, preprocessing_params=cfg.preprocessing, _recursive_=False)

    cv.set_cross_validation(dataset.info_for_cv, cfg.paths.cv, test_params=cfg.test_cv, val_params=cfg.val_cv)

    performance_metrics = metrics.set_metrics_params(cfg.performance_metrics, preprocessing_params=dataset.preprocessing_params)

    for test_fold, val_fold, train, val, test, last_val_fold in cv.get_cross_validation(cfg.paths.cv, "train", "val", "test"):
        if not check_fold_to_do(test_fold, cfg.get("fold_to_do", None)):
            continue

        train_data, train_labels, train_cv_info, val_data, val_labels, _ = cv.get_sets_with_idx(dataset.data, train, val, labels=dataset.labels_for_model, cv_info=dataset.info_for_cv)

        train_set = instantiate(cfg.db.dataset_class, train_data, train_labels, cfg.db.task, "train", cv_info=train_cv_info, preprocessing_params=dataset.preprocessing_params, preprocessing_paths=cfg.paths.preprocessing, test_fold=test_fold, val_fold=val_fold, augmentation=(cfg.model_name.startswith(("naim", "survival_naim")) and not ( 'wsi' in cfg.model_name or 'ct' in cfg.model_name)), missing_augmentation_probability=cfg.get("missing_augmentation_probability", 0.5), normalize_target=cfg.db.get("normalize_target", False), decimals=cfg.db.get("decimals", 2), image_caching=cfg.get("image_caching", True), _recursive_=False)
        val_set = instantiate(cfg.db.dataset_class, val_data, val_labels, cfg.db.task, "val", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=cfg.paths.preprocessing, test_fold=test_fold, val_fold=val_fold, normalize_target=cfg.db.get("normalize_target", False), decimals=cfg.db.get("decimals", 2), image_caching=cfg.get("image_caching", True), _recursive_=False)

        model_params = call(cfg.model.set_params_function, OmegaConf.to_object(cfg.model), preprocessing_params=dataset.preprocessing_params, train_set=train_set, val_set=val_set, _recursive_=False)
        model = instantiate(model_params["init_params"], _recursive_=False)

        train_params = OmegaConf.to_object(cfg.train)
        train_params["set_metrics"] = metrics.set_metrics_params(train_params.get("set_metrics", {}), preprocessing_params=dataset.preprocessing_params)
        call(model_params["train_function"], model, train_set, model_params, cfg.paths.model, val_set=val_set, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

        train_set.set_augmentation(False)
        call(model_params["test_function"], train_set, val_set, model_params=model_params, model_path=cfg.paths.model, prediction_path=cfg.paths.predictions, classes=dataset.classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

        test_data, test_labels = cv.get_sets_with_idx(dataset.data, test, labels=dataset.labels_for_model)
        test_set = instantiate(cfg.db.dataset_class, test_data, test_labels, cfg.db.task, "test", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=cfg.paths.preprocessing, test_fold=test_fold, val_fold=val_fold, normalize_target=cfg.db.get("normalize_target", False), decimals=cfg.db.get("decimals", 2), _recursive_=False)

        call(model_params["test_function"], test_set, model_params=model_params, model_path=cfg.paths.model, prediction_path=cfg.paths.predictions, classes=dataset.classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)


        metrics.compute_performance(dataset.classes, cfg.paths.predictions, cfg.paths.results, cfg.db.task, performance_metrics, decimals=max(cfg.db.get("decimals", 2), 2))

        del train_data, train_labels, val_data, val_labels, test_data, test_labels, train_set, val_set, test_set, model_params, train_params

    #performance_metrics = metrics.set_metrics_params(cfg.performance_metrics, preprocessing_params=dataset.preprocessing_params)
    metrics.compute_performance(dataset.classes, cfg.paths.predictions, cfg.paths.results, cfg.db.task, performance_metrics, decimals=max(cfg.db.get("decimals", 2), 2))

    log.info(f"Job finished")


if __name__ == "__main__":
    pass
