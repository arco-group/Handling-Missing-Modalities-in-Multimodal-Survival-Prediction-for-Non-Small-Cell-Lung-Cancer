import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from omegaconf import DictConfig
from CMC_utils import miscellaneous, save_load

log = logging.getLogger(__name__)

__all__ = ["initialize_experiment", "check_fold_to_do"]


def initialize_experiment(cfg: DictConfig) -> None:
    """
    Initialize experiment by setting seed and creating directories
    Parameters
    ----------
    cfg : DictConfig

    Returns
    -------
    None
    """
    miscellaneous.seed_all(cfg.seed)

    paths_to_create = {key: value for key, value in cfg.paths.items() if key not in ("experiment", "cv_base_path") and value is not None}
    save_load.create_directories(allow_existing=cfg.continue_experiment, **paths_to_create)

    setup_new_log_file(cfg.paths.logs, "logFile")

    defaults_paths_cfg = save_load.load_yaml("./confs/experiment/paths/default.yaml")

    complete_cfg = miscellaneous.join_dictionaries(cfg, defaults_paths_cfg)

    save_load.save_yaml(complete_cfg, "config", cfg.paths.configs, "yaml")


def setup_new_log_file(log_directory, log_file_name):

    # Define the full log path
    full_log_path = os.path.join(log_directory, log_file_name + ".log")

    # Get the root logger
    logger = logging.getLogger()
    # Remove existing file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, (logging.FileHandler, RotatingFileHandler)):
            try:
                os.remove(handler.baseFilename)
            except OSError as _:
                print("File path can not be removed")
            logger.removeHandler(handler)

    # Create a new file handler and add it to the logger
    file_handler = logging.FileHandler(full_log_path)
    logger.addHandler(file_handler)

    # Optionally, set the level and formatter for the new handler
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s - [%(name)s]')
    file_handler.setFormatter(formatter)


def check_fold_to_do(current_fold, assigned_fold):
    if assigned_fold is None:
        return True
    if current_fold < assigned_fold:
        log.info(f"Skipping fold {current_fold}.")
        return False
    if current_fold > assigned_fold:
        log.info(f"Fold {assigned_fold} done. Quitting.")
        sys.exit()
    return True


if __name__ == "__main__":
    pass
