import sys
sys.path.append("CMC_utils")

import os
# import torch
import hydra
import logging
from omegaconf import DictConfig
from hydra.utils import instantiate
from CMC_utils.pipelines import supervised_learning_main, supervised_tabular_missing_main, multimodal_early_fusion_supervised_learning_main, multimodal_joint_fusion_supervised_learning_main, multimodal_late_fusion_supervised_learning_main

log = logging.getLogger(__name__)


@hydra.main(version_base="v1.3", config_path="confs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.paths = instantiate(cfg.paths)
    # torch.autograd.set_detect_anomaly(True)
    os.environ["HYDRA_FULL_ERROR"] = "1"
    pipelines_options = dict(simple=supervised_learning_main,
                             missing=supervised_tabular_missing_main,
                             multimodal_early_fusion=multimodal_early_fusion_supervised_learning_main,
                             multimodal_joint_fusion=multimodal_joint_fusion_supervised_learning_main,
                             multimodal_late_fusion=multimodal_late_fusion_supervised_learning_main)
    import time
    start = time.time()
    pipelines_options[cfg.pipeline](cfg)
    elapsed_time = time.time() - start
    log.info(f"Elapsed time: {elapsed_time}")


if __name__ == '__main__':
    main()

