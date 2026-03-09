import numpy as np
import torch
from typing import Union
from omegaconf import ListConfig
from CMC_utils.miscellaneous import do_nothing
import lifelines
import warnings
import pandas as pd
import torch.nn.functional as F
import functools

__all__ = ["SurvivalLogLikelihoodLoss", "SurvivalLogLikelihoodLoss_singlevalue", "SurvivalRankingLoss", "RegressionSurvivalLoss", "CoxLoss"]


def transform_survival_probability(time, event):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times

    From https://www.kaggle.com/code/cdeotte/gpu-lightgbm-baseline-cv-681-lb-685
    """
    kmf = lifelines.KaplanMeierFitter()
    kmf.fit(time, event)
    y = kmf.survival_function_at_times(time).values
    return torch.tensor(y).unsqueeze(1).float()


def transform_partial_hazard(time, event):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times

    From https://www.kaggle.com/code/andreasbis/cibmtr-eda-ensemble-model
    """
    data = pd.DataFrame({'efs_time': time, 'efs': event, 'time': time, 'event': event})
    cph = lifelines.CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(data, duration_col='time', event_col='event')
    return cph.predict_partial_hazard(data)


class SurvivalLogLikelihoodLoss(torch.nn.Module):
    """
    Loss function for survival analysis based on the log-likelihood of the survival function.
    """

    def __init__(self, num_events: int, max_time: int, eps: float = 1e-08):
        super(SurvivalLogLikelihoodLoss, self).__init__()
        self.num_events = num_events
        self.max_time = max_time
        self.eps = eps

    def get_uncensored_mask(self, labels):
        """
        Get the mask for the uncensored samples.
        Parameters
        ----------
        labels : torch.Tensor

        Returns
        -------
        mask : torch.Tensor
            Mask for the uncensored samples.
        """
        batch_dim = labels.shape[0]
        mask = torch.zeros((batch_dim, self.num_events, self.max_time)).to(labels.device)

        mask[torch.arange(batch_dim), torch.clamp(labels[:, 0, 0] - 1, min=0).long(), labels[:, 0, 1].long()] = 1
        mask[labels[:, 0, 0] == 0, 0, :] = 0

        return mask

    def get_censored_mask(self, labels):
        """
        Get the mask for the censored samples.
        Parameters
        ----------
        labels : torch.Tensor

        Returns
        -------
        mask : torch.Tensor
            Mask for the censored samples.
        """
        batch_dim = labels.shape[0]

        mask = torch.zeros((batch_dim, self.max_time)).to(labels.device)

        mask[torch.arange(batch_dim), labels[:, 0, 1].long()] = 1

        mask[labels[:, 0, 0] != 0, :] = 0

        return mask

    def forward(self, outputs, labels):
        """
        Compute the loss.
        Parameters
        ----------
        outputs : torch.Tensor
        labels : torch.Tensor

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        dim_options = {2: torch.unsqueeze, 3: do_nothing}
        labels = dim_options[len(labels.shape)](labels, dim=1)

        uncensored_mask = self.get_uncensored_mask(labels).to(outputs.device)
        censored_mask = self.get_censored_mask(labels).to(outputs.device)

        CIF = torch.cumsum(outputs, dim=-1)
        censored_values = 1 - torch.sum(CIF, dim=1, keepdim=True)

        uncensored_map = torch.sign(labels[:, :, 0])

        tmp1 = torch.nansum(torch.sum(uncensored_mask * outputs, dim=2), dim=1, keepdim=True)
        tmp1 = torch.mul(torch.log(tmp1 + self.eps), uncensored_map)

        tmp2 = torch.nansum(censored_mask * censored_values, dim=1, keepdim=True)
        tmp2 = torch.mul(torch.log(tmp2 + self.eps), (1. - uncensored_map))

        L1 = tmp1 + tmp2
        loss = - torch.nansum(L1)
        return loss


class SurvivalRankingLoss(torch.nn.Module):
    """
    Loss function for survival analysis based on the ranking loss.
    """

    def __init__(self, num_events: int, max_time: int, sigma: float, alpha: Union[float, list, ListConfig] = 1.0):
        super(SurvivalRankingLoss, self).__init__()
        self.num_events = num_events
        self.max_time = max_time
        self.sigma = sigma
        if isinstance(alpha, float):
            alpha = [alpha] * self.num_events
        self.alpha = torch.Tensor(alpha)

    def get_mask(self, labels):
        """
        Get the mask for the samples.
        Parameters
        ----------
        labels : torch.Tensor

        Returns
        -------
        mask : torch.Tensor
            Mask for the samples.
        """
        batch_dim = labels.shape[0]

        tmp1 = torch.repeat_interleave(labels[:, :, 1], batch_dim, dim=1)
        tmp1 = tmp1 < tmp1.transpose(1, 0)

        tmp2 = torch.repeat_interleave(labels[:, :, 0], batch_dim, dim=1)
        tmp2 = tmp2 != 0
        # tmp2 = tmp2 == tmp2.transpose(1, 0)
        # tmp2[labels[:, 0, 0] == 0, :] = 0

        mask = torch.reshape(torch.unsqueeze(tmp1 * tmp2, dim=2), (1, batch_dim, batch_dim))
        return mask

    def forward(self, outputs, labels):
        """
        Compute the loss.
        Parameters
        ----------
        outputs : torch.Tensor
        labels : torch.Tensor

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        dim_options = {2: torch.unsqueeze, 3: do_nothing}
        labels = dim_options[len(labels.shape)](labels, dim=1)

        batch_dim = labels.shape[0]

        CIF = torch.cumsum(outputs, dim=-1).unsqueeze(1)

        sample_idx = torch.unsqueeze(torch.arange(batch_dim), dim=1).to(outputs.device)
        k_event_idx = torch.clamp(labels[:, :, 0] - 1, min=0).long()
        k_time_idx = labels[:, :, 1].long()

        tmp1_idx = torch.cat([sample_idx, k_event_idx, k_time_idx], dim=1)
        tmp1 = CIF[tmp1_idx.chunk(chunks=3, dim=1)]
        tmp1 = torch.repeat_interleave(torch.unsqueeze(tmp1, dim=2), batch_dim, dim=2)
        tmp1 = torch.transpose(tmp1, 1, 0)

        CIF_ref = torch.transpose(CIF, 2, 0)
        tmp2_idx = torch.cat([k_time_idx, k_event_idx], dim=1)
        tmp2 = CIF_ref[tmp2_idx.chunk(chunks=2, dim=1)]
        tmp2 = torch.transpose(tmp2, 1, 0)

        tmp_num = tmp1 - tmp2

        tmp = torch.exp(- tmp_num / self.sigma)
        mask = self.get_mask(labels).to(outputs.device)

        L2 = torch.nansum(mask * tmp, dim=2).squeeze()
        alpha = self.alpha.to(outputs.device)
        alpha = alpha[k_event_idx].squeeze()

        L2 = L2 * alpha
        loss = torch.nansum(L2)

        return loss


class SurvivalLogLikelihoodLoss_singlevalue(torch.nn.Module):
    """
    Loss function for survival analysis based on the log-likelihood of the survival function.
    """

    def __init__(self, num_events: int, max_time: int, sigma: int = 1.0, eps: float = 1e-08):
        super(SurvivalLogLikelihoodLoss_singlevalue, self).__init__()
        self.num_events = num_events
        self.max_time = max_time
        self.sigma = sigma
        self.eps = eps

    def get_uncensored_mask(self, labels):
        """
        Get the mask for the uncensored samples.
        Parameters
        ----------
        labels : torch.Tensor

        Returns
        -------
        mask : torch.Tensor
            Mask for the uncensored samples.
        """
        batch_dim = labels.shape[0]
        mask = torch.zeros((batch_dim, self.num_events, self.max_time)).to(labels.device)

        mask[torch.arange(batch_dim), torch.clamp(labels[:, 0, 0] - 1, min=0).long(), labels[:, 0, 1].long()] = 1
        mask[labels[:, 0, 0] == 0, 0, :] = 0

        return mask

    def get_censored_mask(self, labels):
        """
        Get the mask for the censored samples.
        Parameters
        ----------
        labels : torch.Tensor

        Returns
        -------
        mask : torch.Tensor
            Mask for the censored samples.
        """
        batch_dim = labels.shape[0]

        mask = torch.zeros((batch_dim, self.max_time)).to(labels.device)

        mask[torch.arange(batch_dim), labels[:, 0, 1].long()] = 1

        mask[labels[:, 0, 0] != 0, :] = 0

        return mask

    def forward(self, outputs, labels, **_):
        """
        Compute the loss.
        Parameters
        ----------
        outputs : torch.Tensor
        labels : torch.Tensor

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        dim_options = {2: torch.unsqueeze, 3: do_nothing}
        labels = dim_options[len(labels.shape)](labels, dim=1)

        # uncensored_mask = self.get_uncensored_mask(labels).to(outputs.device)
        # censored_mask = self.get_censored_mask(labels).to(outputs.device)

        CIF = torch.cumsum(outputs, dim=-1).sum(dim=-1, keepdim=True).div(self.max_time)  # 1 -
        # censored_values = 1 - torch.sum(CIF, dim=1, keepdim=True)

        uncensored_map = labels[:, :, 0]  # torch.sign(labels[:, :, 0])
        # targets = labels[:, :, 1] / self.max_time

        #
        targets = transform_survival_probability(labels[:, :, 1].cpu().numpy().squeeze(), labels[:, :, 0].cpu().numpy().squeeze()).to(outputs.device)  # 1 -
        #
        # tmp1 = torch.nn.MSELoss(reduction='none')(outputs, targets)
        # tmp1 = uncensored_map * tmp1
        #
        # normalized_target = labels[:, :, 1].div(self.max_time)
        tmp1 = uncensored_map * torch.abs(CIF - targets)
        tmp1 = uncensored_map * torch.log(torch.exp(tmp1 / self.sigma) + self.eps)

        # tmp1 = torch.nansum(torch.sum(uncensored_map * outputs, dim=-1), dim=-1, keepdim=True)

        # tmp2 = torch.nansum(censored_mask * censored_values, dim=-1, keepdim=True)
        # tmp2 = torch.mul( torch.log(tmp2 + self.eps), (1. - uncensored_map) )

        # L1 = tmp1 + tmp2
        loss = torch.nansum(tmp1)
        # loss = - torch.nansum(tmp1)
        return loss


@functools.lru_cache
def combinations(N):
    """
    calculates all possible 2-combinations (pairs) of a tensor of indices from 0 to N-1,
    and caches the result using functools.lru_cache for optimization
    """
    ind = torch.arange(N)
    comb = torch.combinations(ind, r=2)
    return comb.to('cuda' if torch.cuda.is_available() else 'cpu')


class MSE(torch.nn.Module):
    def __init__(self, reduction='none'):
        super(MSE, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y):
        return F.mse_loss(y_pred, y, reduction=self.reduction)


class RankingLossWithMargin(torch.nn.Module):
    def __init__(self, margin=1.0, reduction='none'):
        super(RankingLossWithMargin, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, y_pred, y, efs):
        """
        y_pred: (batch_size, 1)
        y: (batch_size, 1) with values in {0, 1}
        """
        # Following the pairwise ranking-with-margin approach. We build all pairs,
        # keep only pairs where at least one subject experienced the event (efs==1),
        # compute signed labels (+1/-1) and apply a ReLU margin on prediction differences.
        N = y.shape[0]
        comb = combinations(N)
        # keep pairs where at least one is eventful aording to efs
        #         comb = comb[(efs[comb[:, 0]] == 1) | (efs[comb[:,cc 1]] == 1)]

        pred_left = y_pred[comb[:, 0]]
        pred_right = y_pred[comb[:, 1]]
        y_left = y[comb[:, 0]]
        y_right = y[comb[:, 1]]

        # construct pairwise signed label: +1 if left>right else -1
        pair_y = 2 * (y_left > y_right).int() - 1
        margin = self.margin + pair_y * (y_left-y_right)
        # raw pairwise hinge-like loss with margin
        loss_vals = F.relu(-pair_y.float() * (pred_left - pred_right) + margin)

        # build mask compatible with other losses: try to reuse get_mask if present
        # The module doesn't have a get_mask method here, but other classes do. We'll
        # construct a simple mask: valid pairs are those where (y_left != y_right) OR
        # where at least one efs==1 (we already filtered on efs), so mask is 1 where
        # labels are not equal.
        left_outlived = y_left >= y_right
        left_1_right_0 = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
        mask2 = (left_outlived & left_1_right_0)
        right_outlived = y_right >= y_left
        right_1_left_0 = (efs[comb[:, 1]] == 1) & (efs[comb[:, 0]] == 0)
        mask2 |= (right_outlived & right_1_left_0)
        mask2 = ~mask2
        mask = mask2
        loss = (loss_vals.double() * (mask.double())).sum() / (mask.float().sum() + 1e-10)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class SurvivalLogLikelihoodLoss_singlevalue(torch.nn.Module):
    """
    Loss function for survival analysis based on the log-likelihood of the survival function.
    """
    def __init__(self, num_events: int, max_time: int, sigma: int = 1.0, eps: float = 1e-08):
        super(SurvivalLogLikelihoodLoss_singlevalue, self).__init__()
        self.num_events = num_events
        self.max_time = max_time
        self.sigma = sigma
        self.eps = eps

    def get_uncensored_mask(self, labels):
        """
        Get the mask for the uncensored samples.
        Parameters
        ----------
        labels : torch.Tensor

        Returns
        -------
        mask : torch.Tensor
            Mask for the uncensored samples.
        """
        batch_dim = labels.shape[0]
        mask = torch.zeros((batch_dim, self.num_events, self.max_time)).to(labels.device)

        mask[torch.arange(batch_dim), torch.clamp(labels[:, 0, 0] - 1, min=0).long(), labels[:, 0, 1].long()] = 1
        mask[labels[:, 0, 0] == 0, 0, : ] = 0

        return mask

    def get_censored_mask(self, labels):
        """
        Get the mask for the censored samples.
        Parameters
        ----------
        labels : torch.Tensor

        Returns
        -------
        mask : torch.Tensor
            Mask for the censored samples.
        """
        batch_dim = labels.shape[0]

        mask = torch.zeros((batch_dim, self.max_time)).to(labels.device)

        mask[torch.arange(batch_dim), labels[:, 0, 1].long()] = 1

        mask[labels[:, 0, 0] != 0, :] = 0

        return mask

    def forward(self, outputs, labels, **_):
        """
        Compute the loss.
        Parameters
        ----------
        outputs : torch.Tensor
        labels : torch.Tensor

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        dim_options = {2: torch.unsqueeze, 3: do_nothing}
        labels = dim_options[len(labels.shape)](labels, dim=1)

        # uncensored_mask = self.get_uncensored_mask(labels).to(outputs.device)
        # censored_mask = self.get_censored_mask(labels).to(outputs.device)

        CIF = torch.cumsum(outputs, dim=-1).sum(dim=-1, keepdim=True).div(self.max_time)  # 1 -
        # censored_values = 1 - torch.sum(CIF, dim=1, keepdim=True)

        uncensored_map = labels[:, :, 0]  # torch.sign(labels[:, :, 0])
        # targets = labels[:, :, 1] / self.max_time

        #
        targets = transform_survival_probability(labels[:, :, 1].cpu().numpy().squeeze(), labels[:, :, 0].cpu().numpy().squeeze()).to(outputs.device)  # 1 -
        #
        # tmp1 = torch.nn.MSELoss(reduction='none')(outputs, targets)
        # tmp1 = uncensored_map * tmp1
        #
        # normalized_target = labels[:, :, 1].div(self.max_time)
        tmp1 = uncensored_map * torch.abs(CIF - targets)
        tmp1 = uncensored_map * torch.log(torch.exp(tmp1/self.sigma) + self.eps)

        # tmp1 = torch.nansum(torch.sum(uncensored_map * outputs, dim=-1), dim=-1, keepdim=True)

        # tmp2 = torch.nansum(censored_mask * censored_values, dim=-1, keepdim=True)
        # tmp2 = torch.mul( torch.log(tmp2 + self.eps), (1. - uncensored_map) )

        # L1 = tmp1 + tmp2
        loss = torch.nansum(tmp1)
        # loss = - torch.nansum(tmp1)
        return loss


class CoxLoss(torch.nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()
    @staticmethod
    def get_R_matrix(survival_time):
        """
        Create an indicator matrix of risk sets, where T_j >= T_i.
        Input:
            survival_time: a Pytorch tensor that the number of rows is equal top the number of samples
        Output:
            indicator matrix: an indicator matrix
        """
        batch_length = survival_time.shape[0]
        R_matrix = np.zeros([batch_length, batch_length], dtype=int)
        for i in range(batch_length):
            for j in range(batch_length):
                R_matrix[i, j] = survival_time[j] >= survival_time[i]
        return R_matrix

    def neg_par_log_likelihood(self, pred, labels):
        """
        Calculate the average Cox negative partial log-likelihood
        Input:
            pred: linear predictors from trained model.
            survival_time: survival time from ground truth
            survival_event: survival event from ground truth: 1 for event and 0 for censored
        Output:
            cost: the survival cost to be minimized
        """
        survival_time = labels[:, 1]
        survival_event = labels[:, 0].reshape((-1, 1))
        pred = pred.reshape((-1, 1))
        n_observed = torch.sum(survival_event)
        # print(n_observed)
        R_matrix = self.get_R_matrix(survival_time)
        R_matrix = torch.Tensor(R_matrix).to(pred.device)

        risk_set_sum = R_matrix.mm(torch.exp(pred))
        # print("risk_set_sum", risk_set_sum)
        diff = pred - torch.log(risk_set_sum)
        # print("diff", diff)
        sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(survival_event)
        # print("sum_diff_in_observed", sum_diff_in_observed)
        loss = (- (sum_diff_in_observed) / n_observed).reshape((-1,))
        return loss

    def forward(self, y_pred,  labels, **_):
        """
        Compute the loss.
        Parameters
        ----------
        y_pred : torch.Tensor
        labels : torch.Tensor
        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        loss = self.neg_par_log_likelihood(y_pred, labels)
        return loss


class RegressionSurvivalLoss(torch.nn.Module):
    """
    Loss function for survival analysis based on the regression loss.
    """

    def __init__(self, margin: float, aux_weight: float = 1.0):
        super(RegressionSurvivalLoss, self).__init__()

        self.ranking_loss = RankingLossWithMargin(margin=margin, reduction='none')

    def forward(self, outputs, labels, **_):
        """
        Compute the loss.
        Parameters
        ----------
        outputs : torch.Tensor
        labels : torch.Tensor

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        scoring_output = outputs

        efs = labels[:, 0]
        efs_time = labels[:, 1]


        #efs_time = efs_time.float()
        ranking_loss = self.ranking_loss(scoring_output, efs_time, efs)
        total_loss =   ranking_loss #+ self.aux_weight * aux_loss
        return total_loss


if __name__ == "__main__":
    pass
