import torch

__all__ = ["surpass_threshold", "max_index", "survival_label"]


def survival_label(outputs, num_events: int, max_time: int, **_) -> torch.Tensor:
    """
    Returns the prediction for the survival model. The prediction is the time of the event and the event itself.
    Parameters
    ----------
    outputs : torch.Tensor
    num_events : int
    max_time : int

    Returns
    -------
    torch.Tensor
    """
    outputs = outputs.view(-1, num_events, max_time)
    max_probs, times = torch.max(outputs, dim=2)
    _, preds = torch.max(max_probs, dim=1)
    preds = preds.unsqueeze(dim=1)

    event_times = torch.gather(times, 1, preds)
    prediction = torch.cat((preds.add(1), event_times), dim=1)
    return prediction


def max_index(outputs, **_) -> torch.Tensor:
    """
    Returns the index of the maximum value in the output tensor.
    Parameters
    ----------
    outputs : torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    _, prediction = torch.max(outputs, 1)
    return prediction


def surpass_threshold(outputs, threshold=0.5, **_) -> torch.Tensor:
    """
    Returns the prediction for the binary classification model. The prediction is 1 if the output surpasses the threshold
    and 0 otherwise.
    Parameters
    ----------
    outputs : torch.Tensor
    threshold : float

    Returns
    -------
    torch.Tensor
    """
    preds = (outputs > threshold).long()
    return preds

if __name__ == "__main__":
    pass
