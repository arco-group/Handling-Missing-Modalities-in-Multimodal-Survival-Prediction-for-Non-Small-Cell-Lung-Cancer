from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score

__all__ = ["AUC", "accuracy", "recall", "precision", "f1", "mcc", "gmean"]


def AUC(y_true, y_score, average='macro', sample_weight=None, max_fpr=None, multi_class='ovr', labels=None, **_):
    """Compute Area Under the Curve (AUC) from prediction scores"""
    return roc_auc_score(y_true, y_score, average=average, sample_weight=sample_weight, max_fpr=max_fpr,
                         multi_class=multi_class, labels=labels)


def accuracy(y_true, y_pred, normalize=True, sample_weight=None, **_):
    """Accuracy classification score."""
    return [ accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight) ]


def recall(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn',
           **kwargs):
    """Compute the recall"""
    return recall_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average,
                        sample_weight=sample_weight, zero_division=zero_division)


def precision(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn', **_):
    """Compute the precision"""
    return precision_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average,
                           sample_weight=sample_weight, zero_division=zero_division)


def f1(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn', **_):
    """Compute the F1 score"""
    return f1_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight,
                    zero_division=zero_division)


def mcc(y_true, y_pred, sample_weight=None, **_):
    """Compute the Matthews correlation coefficient (MCC)"""
    return matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)


def gmean(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, correction=0.0, **_):
    """Compute the geometric mean"""
    return geometric_mean_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average,
                                sample_weight=sample_weight, correction=correction)


if __name__ == "__main__":
    pass
