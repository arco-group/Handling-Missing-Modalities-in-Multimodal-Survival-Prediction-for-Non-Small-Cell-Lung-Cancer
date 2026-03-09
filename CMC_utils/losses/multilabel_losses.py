import torch
import torch.nn.functional as F


__all__ = ["MultilabelBCEWithLogitsLoss"]


class MultilabelBCEWithLogitsLoss(torch.nn.Module):

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, ignore_index=-100):
        super(MultilabelBCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, y_score, y_true):
        mask = y_true != self.ignore_index
        loss = F.binary_cross_entropy_with_logits(y_score, y_true.masked_fill(~mask, 0), reduction='none')
        loss = loss * mask
        loss = loss.sum(dim=0) / mask.sum(dim=0)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


if __name__ == "__main__":
    pass
