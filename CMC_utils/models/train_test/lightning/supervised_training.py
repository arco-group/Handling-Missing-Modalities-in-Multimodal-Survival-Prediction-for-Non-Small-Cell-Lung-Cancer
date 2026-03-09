import torch
from CMC_utils.datasets import SupervisedTabularDatasetTorch
from typing import Union, Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate, call
import torchmetrics
import pytorch_lightning as pl


class SupervisedLearning(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, train_set: SupervisedTabularDatasetTorch, model_params: dict, model_path: str, val_set: SupervisedTabularDatasetTorch, train_params: Union[dict, DictConfig], test_fold: int = 0, val_fold: int = 0, use_weights: bool = True):
        super().__init__()
        self.save_hyperparameters(dict(model_params=model_params, train_params=train_params, use_weights=use_weights))

        self.model = model

        task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'

        self.acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

        self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)

        losses_params = {loss_name: call(loss_params.set_params_function, loss_params, model_params=model_params, train_set=train_set, _recursive_=False) for loss_name, loss_params in train_params.loss.items()}
        self.criterions = [instantiate(loss_params["init_params"]) for loss_params in losses_params.values()]

        self.best_val_score = 0

        print(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        return y_hat

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        x, y = batch
        y_hat = self.forward(x)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]

        self.acc_test(y_hat, y)
        self.auc_test(y_hat, y)

    def test_epoch_end(self, _) -> None:
        """
        Test epoch end
        """
        test_acc = self.acc_test.compute()
        test_auc = self.auc_test.compute()

        self.log('test.acc', test_acc)
        self.log('test.auc', test_auc)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """
        Train and log.
        """
        x, y = batch

        y_hat = self.forward(x)
        loss = 0
        for criterion in self.criterions:
            loss += criterion(y_hat, y)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]

        self.acc_train(y_hat, y)
        self.auc_train(y_hat, y)

        self.log('eval.train.loss', loss, on_epoch=True, on_step=False)

        return loss

    def training_epoch_end(self, _) -> None:
        """
        Compute training epoch metrics and check for new best values
        """
        self.log('eval.train.acc', self.acc_train, on_epoch=True, on_step=False, metric_attribute=self.acc_train)
        self.log('eval.train.auc', self.auc_train, on_epoch=True, on_step=False, metric_attribute=self.auc_train)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """
        Validate and log
        """
        x, y = batch

        y_hat = self.forward(x)
        loss = 0
        for criterion in self.criterions:
            loss += criterion(y_hat, y)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]

        self.acc_val(y_hat, y)
        self.auc_val(y_hat, y)

        self.log('eval.val.loss', loss, on_epoch=True, on_step=False)

    def validation_epoch_end(self, _) -> None:
        """
        Compute validation epoch metrics and check for new best values
        """
        if self.trainer.sanity_checking:
            return

        epoch_acc_val = self.acc_val.compute()
        epoch_auc_val = self.auc_val.compute()

        self.log('eval.val.acc', epoch_acc_val, on_epoch=True, on_step=False, metric_attribute=self.acc_val)
        self.log('eval.val.auc', epoch_auc_val, on_epoch=True, on_step=False, metric_attribute=self.auc_val)

        if self.hparams.target == 'dvm':
            self.best_val_score = max(self.best_val_score, epoch_acc_val)
        else:
            self.best_val_score = max(self.best_val_score, epoch_auc_val)

        self.acc_val.reset()
        self.auc_val.reset()

    def configure_optimizers(self):
        """
        Sets optimizer and scheduler.
        Must use strict equal to false because if check_val_n_epochs is > 1
        because val metrics not defined when scheduler is queried
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr_eval,
                                     weight_decay=self.hparams.weight_decay_eval)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=int(10 / self.hparams.check_val_every_n_epoch),
                                                               min_lr=self.hparams.lr * 0.0001)
        return optimizer

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'eval.val.loss',
                    "strict": False
                }
            }
        )
