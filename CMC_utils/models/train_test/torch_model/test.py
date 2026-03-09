import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import instantiate, call
from torch.utils.data import DataLoader
from CMC_utils.save_load import save_table
from CMC_utils.models import set_device, remove_DataParallel_module
from CMC_utils.miscellaneous import do_nothing
from CMC_utils.preprocessing import discrete_to_label
from CMC_utils.datasets import SupervisedTabularDatasetTorch

from .outputs_functions import *

log = logging.getLogger(__name__)

__all__ = ["test_torch_model"]



def test_torch_model(*sets: SupervisedTabularDatasetTorch, model_params: dict, model_path: str, prediction_path: str, train_params: Union[dict, DictConfig], test_fold: int = 0, val_fold: int = 0, **kwargs) -> None:
    """
    Test a torch model
    Parameters
    ----------
    sets : SupervisedTabularDatasetTorch
    model_params : dict
    model_path : str
    prediction_path : str
    train_params : Union[dict, DictConfig]
    test_fold : int
    val_fold : int
    kwargs : dict

    Returns
    -------
    None
    """
    sets = [ DataLoader(fset, batch_size=train_params.dl_params.batch_size, shuffle=False, drop_last=False, num_workers=4) for fset in sets ]

    model_path = os.path.join(model_path, f"{test_fold}_{val_fold}.{model_params['file_extension']}")

    model = instantiate(model_params["init_params"], _recursive_=False)

    model_state_dict = call(model_params["load_function"], model_path)
    if not isinstance(model, torch.nn.DataParallel):
        model_state_dict = remove_DataParallel_module(model_state_dict)
    model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    device = set_device(train_params.dl_params.device)

    output_to_pred_options = {"binary": surpass_threshold, "categorical": max_index, "single_risk_survival": survival_label,"single_risk_survival_regression" :do_nothing,  "competing_risks_survival": survival_label, "continuous": do_nothing, "multilabel": surpass_threshold}

    model = model.to(device)

    model.eval()


    for fset in sets:

        filename = f"{test_fold}_{val_fold}_{fset.dataset.set_name}.csv"
        if os.path.exists(os.path.join(prediction_path, filename)):
            continue

        pbar = tqdm(fset, leave=False, disable=(not train_params.dl_params.verbose_batch))

        total_results = pd.DataFrame()

        with torch.set_grad_enabled(False):
            for *input_list, labels, idxs in pbar:
                input_list = [inputs.float().to(device) for inputs in input_list]
                labels = labels.float().to(device)

                missing_samples_mask = torch.isnan(torch.cat([inputs.view(inputs.shape[0], -1) for inputs in input_list], dim=1)).all(dim=1).to(device)
                for inputs in input_list:
                    inputs[missing_samples_mask] = torch.nan_to_num(inputs[missing_samples_mask], nan=0)
                missing_samples_mask = missing_samples_mask.cpu().detach().numpy()

                outputs = model(*[inputs.float() for inputs in input_list])

                if (fset.dataset.label_type == "binary" and outputs.shape[-1] == 2) or fset.dataset.label_type == "categorical":
                    outputs = F.softmax(outputs, dim=-1)
                elif fset.dataset.label_type == "multilabel":
                    outputs = F.sigmoid(outputs)

                if fset.dataset.label_type in ("single_risk_survival", "competing_risks_survival"):
                    preds = output_to_pred_options[fset.dataset.label_type](outputs, return_first=True, num_events=model.num_events, max_time=model.max_time, **kwargs)
                else:
                    preds = output_to_pred_options[fset.dataset.label_type](outputs, return_first=True, **kwargs)

                if fset.dataset.label_type in ("single_risk_survival", "competing_risks_survival"):
                    outputs = torch.cumsum(outputs.view(-1, model.num_events, model.max_time), dim=-1).view(-1, model.num_events * model.max_time)
                elif labels.shape != preds.shape:
                    labels = output_to_pred_options[fset.dataset.label_type](labels, return_first=True, **kwargs)

                idxs = np.array(idxs)
                labels = np.squeeze(labels.cpu().detach().numpy())
                preds = np.squeeze(preds.cpu().detach().numpy())
                if fset.dataset.label_type == "continuous":
                    preds = list(np.round(preds, fset.dataset.decimals))
                else:
                    labels = labels.astype(int).tolist()
                    preds = preds.astype(int).tolist()
                    if isinstance(labels, int):
                        labels = [labels]
                    if isinstance(preds, int):
                        preds = [preds]
                outputs = outputs.cpu().detach().numpy().astype(float)

                if fset.dataset.label_type != "categorical":
                    outputs = np.round(outputs, getattr(fset.dataset, "decimals", 4))

                if fset.dataset.label_type == "binary":
                    if outputs.shape[1] == 2:
                        outputs = outputs[:, 1]
                    outputs = np.squeeze(outputs)

                if fset.dataset.label_type in ("binary", "categorical"):
                    preds = list(map( lambda label: discrete_to_label(label, classes=fset.dataset.classes), preds))
                    labels = list(map( lambda label: discrete_to_label(label, classes=fset.dataset.classes), labels))

                if np.any(missing_samples_mask):
                    outputs[missing_samples_mask] = np.full(outputs[missing_samples_mask].shape, np.nan)
                    if fset.dataset.label_type != "continuous":
                        # preds = pd.array(preds, dtype='Int64')
                        preds = pd.DataFrame(preds, dtype='Int64')
                    else:
                        # preds = np.array(preds)
                        preds = pd.DataFrame(preds)
                    preds[missing_samples_mask] = None
                    # preds = list(preds)
                    preds = preds.values.tolist()

                running_results = pd.DataFrame(dict( ID=idxs, label=labels, prediction=preds, probability=outputs.tolist()))
                total_results = pd.concat( [total_results, running_results], axis=0, ignore_index=True)

        save_table(total_results, filename, prediction_path)
        log.info("Inference done")


if __name__ == "__main__":
    pass
