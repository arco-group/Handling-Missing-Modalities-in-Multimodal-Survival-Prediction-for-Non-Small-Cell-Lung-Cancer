import os
import logging
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf, ListConfig
from .routines import initialize_experiment, check_fold_to_do
from CMC_utils import cross_validation as cv
from CMC_utils.datasets import MultimodalJointFusionTabularDatasetTorch
from CMC_utils import metrics, miscellaneous, preprocessing, save_load

log = logging.getLogger(__name__)

__all__ = ["multimodal_joint_fusion_supervised_learning_main"]


def multimodal_joint_fusion_supervised_learning_main(cfg: DictConfig) -> None:
    log.info(f"Multimodal Joint Fusion Supervised main started")

    initialize_experiment(cfg)

    datasets = [instantiate(cfg.dbs[i], model_label_types=cfg.model.label_types, model_framework=cfg.model.framework, preprocessing_params=cfg.preprocessing, _recursive_=False) for i in cfg.dbs.keys()]
    datasets_names = [dataset.name for dataset in datasets]

    cv.set_cross_validation(datasets[0].info_for_cv, cfg.paths.cv, test_params=cfg.test_cv, val_params=cfg.val_cv)
    performance_metrics = metrics.set_metrics_params(cfg.performance_metrics, preprocessing_params=datasets[0].preprocessing_params)

    for test_fold, val_fold, train, val, test, last_val_fold in cv.get_cross_validation(cfg.paths.cv, "train", "val", "test"):
        if not check_fold_to_do(test_fold, cfg.get("fold_to_do", None)):
            continue

        for train_missing_fraction in cfg.missing_percentages:
            train_missing_percentage = int(100 * train_missing_fraction)

            preprocessing_paths = {key: os.path.join(value, str(train_missing_percentage)) for key, value in cfg.paths.preprocessing.items()}
            save_load.create_directories(**preprocessing_paths)

            train_data, train_labels, train_cv_info, val_data, val_labels, _ = cv.get_sets_with_idx([dataset.data for dataset in datasets], train, val, labels=[dataset.labels_for_model for dataset in datasets], cv_info=[dataset.info_for_cv for dataset in datasets])

            train_data_missing, _ = preprocessing.generate_multimodal_missing(train_data, datasets_names=datasets_names, test_fold=test_fold, val_fold=val_fold, fset="train", save_path=cfg.paths.missing_masks, **miscellaneous.recursive_cfg_substitute(cfg.missing_generation.train, {"missing_fraction": train_missing_fraction}))
            val_data_missing, _ = preprocessing.generate_multimodal_missing(val_data, datasets_names=datasets_names, test_fold=test_fold, val_fold=val_fold, fset="val", save_path=cfg.paths.missing_masks, **miscellaneous.recursive_cfg_substitute(cfg.missing_generation.val, {"missing_fraction": train_missing_fraction}))
            train_set = []
            val_set = []
            for dataset, db_id in zip(datasets, cfg.dbs.keys()):
                train_set.append(instantiate(cfg.dbs[db_id].dataset_class, train_data_missing[int(db_id)], train_labels[int(db_id)], cfg.dbs[db_id].task, "train", cv_info=train_cv_info[int(db_id)], preprocessing_params=dataset.preprocessing_params, preprocessing_paths=preprocessing_paths, test_fold=test_fold, val_fold=val_fold, augmentation=cfg.ms_models[db_id].name.startswith(("naim", "survival_naim")), missing_augmentation_probability=cfg.get("missing_augmentation_probability", 0.5), dataset_name=dataset.name, normalize_target=cfg.dbs[db_id].get("normalize_target", False), decimals=cfg.dbs[db_id].get("decimals", 2), image_caching=cfg.get("image_caching", True), _recursive_=False))
                val_set.append(instantiate(cfg.dbs[db_id].dataset_class, val_data_missing[int(db_id)], val_labels[int(db_id)], cfg.dbs[db_id].task, "val", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=preprocessing_paths, test_fold=test_fold, val_fold=val_fold, dataset_name=dataset.name, normalize_target=cfg.dbs[db_id].get("normalize_target", False), decimals=cfg.dbs[db_id].get("decimals", 2), image_caching=cfg.get("image_caching", True), _recursive_=False))

            train_set, val_set = MultimodalJointFusionTabularDatasetTorch(*train_set, augmentation=cfg.shared_net.name.lower().startswith(("naim", "survival_naim", "maria")), missing_augmentation_probability=cfg.get("missing_augmentation_probability", 0.3)), MultimodalJointFusionTabularDatasetTorch(*val_set)

            preprocessing_params = [dataset.preprocessing_params for dataset in datasets]
            checkpoint_path = os.path.join("./checkpoints", str(train_missing_percentage))
            model_params = call(cfg.model.set_params_function, OmegaConf.to_object(cfg.model), preprocessing_params=preprocessing_params, train_set=train_set, checkpoint_path=checkpoint_path, test_fold=test_fold, val_fold=val_fold, _recursive_=False)
            model = instantiate(model_params["init_params"], _recursive_=False)

            model_path = os.path.join(cfg.paths.model, str(train_missing_percentage))
            train_params = OmegaConf.to_object(cfg.train)
            train_params["set_metrics"] = metrics.set_metrics_params(train_params.get("set_metrics", {}), preprocessing_params=preprocessing_params[0])
            call(model_params["train_function"], model, train_set, model_params, model_path, val_set=val_set, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False, pretrained_model_parts = dict(cfg.get("load_model_pretrained_parts", {})))

            prediction_path = os.path.join(cfg.paths.predictions, str(train_missing_percentage))
            save_load.create_directory(prediction_path)

            for tr_ds in train_set.datasets:
                tr_ds.set_augmentation(False)
            train_set.set_augmentation(False)

            call(model_params["test_function"], train_set, val_set, model_params=model_params, model_path=model_path, prediction_path=prediction_path, classes=datasets[0].classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)
            del train_data, train_labels, train_set, train_data_missing, val_data, val_labels, val_set, val_data_missing

            for test_missing_fraction in cfg.get("test_missing_percentages", cfg.missing_percentages):
                if isinstance(test_missing_fraction, (list, ListConfig)):
                    test_missing_percentage = "-".join([f"{int(100 * test_missing_fraction[i])}" for i in range(len(test_missing_fraction))])
                else:
                    test_missing_percentage = int(100 * test_missing_fraction)
                log.info(f"Test missing percentage: {test_missing_percentage}")

                test_prediction_path = os.path.join(prediction_path, str(test_missing_percentage))
                save_load.create_directory(test_prediction_path)

                test_data, test_labels = cv.get_sets_with_idx([dataset.data for dataset in datasets], test, labels=[dataset.labels_for_model for dataset in datasets])
                test_data_missing, _ = preprocessing.generate_multimodal_missing(test_data, datasets_names=datasets_names, test_fold=test_fold, val_fold=val_fold, fset="test", save_path=cfg.paths.missing_masks, **miscellaneous.recursive_cfg_substitute( cfg.missing_generation.test, {"missing_fraction": test_missing_fraction}))
                test_set = []
                for dataset, db_id in zip(datasets, cfg.dbs.keys()):
                    test_set.append(instantiate(cfg.dbs[db_id].dataset_class, test_data_missing[int(db_id)], test_labels[int(db_id)], cfg.dbs[db_id].task, "test", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=preprocessing_paths, test_fold=test_fold, val_fold=val_fold, dataset_name=dataset.name, normalize_target=cfg.dbs[db_id].get("normalize_target", False), decimals=cfg.dbs[db_id].get("decimals", 2), _recursive_=False))

                test_set = MultimodalJointFusionTabularDatasetTorch(*test_set)

                call(model_params["test_function"], test_set, model_params=model_params, model_path=model_path, prediction_path=test_prediction_path, classes=datasets[0].classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

                del test_data, test_labels, test_set, test_data_missing
            # del train_data, train_labels, train_set, train_data_missing, val_data, val_labels, val_set, val_data_missing, model_params, train_params

            metrics.compute_missing_performance(datasets[0].classes, cfg.paths.predictions, cfg.paths.results, next(iter(cfg.dbs.values())).task, performance_metrics, cfg.missing_percentages, cfg.get("test_missing_percentages", cfg.missing_percentages), decimals=max(next(iter(cfg.dbs.values())).get("decimals", 2), 2))

            del model_params, train_params


    metrics.compute_missing_performance(datasets[0].classes, cfg.paths.predictions, cfg.paths.results, next(iter(cfg.dbs.values())).task, performance_metrics, cfg.missing_percentages, cfg.get("test_missing_percentages", cfg.missing_percentages), decimals=max(next(iter(cfg.dbs.values())).get("decimals", 2), 2))

    log.info(f"Job finished")


if __name__ == "__main__":
    pass
