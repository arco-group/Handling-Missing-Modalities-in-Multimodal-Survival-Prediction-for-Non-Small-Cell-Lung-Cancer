import numpy as np

__all__ = ["intersection_over_union", "dice_coefficient", "dice_loss"]


def intersection_over_union(y_true, y_pred, **kwargs):
    """
    Compute the intersection over union
    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    kwargs : dict

    Returns
    -------
    np.ndarray
    """
    y_true, y_pred = y_true.flatten().astype(np.float32), y_pred.flatten().astype(np.float32)

    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection

    iou = (intersection + 1e-15) / (union + 1e-15)
    iou = iou.astype(np.float32)

    return iou


def dice_coefficient(y_true, y_pred, **kwargs):
    """
    Compute the dice coefficient
    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    kwargs : dict

    Returns
    -------
    np.ndarray
    """
    y_true, y_pred = y_true.flatten().astype(np.float32), y_pred.flatten().astype(np.float32)

    intersection = (y_true * y_pred).sum()

    return (2. * intersection + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def dice_loss(y_true, y_pred, **kwargs):
    """
    Compute the dice loss
    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    kwargs : dict

    Returns
    -------
    np.ndarray
    """
    return 1.0 - dice_coefficient(y_true, y_pred)


if __name__ == "__main__":
    pass
