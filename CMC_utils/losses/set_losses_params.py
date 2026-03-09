
__all__ = ["set_survival_loss_params", "set_huber_mean_loss_params", "set_survival_regression_loss_params"]


def set_survival_regression_loss_params(loss_params: dict, model_params: dict, **_):
    """
    Set the number of events and the maximum time for the loss function.
    Parameters
    ----------
    loss_params : dict
    model_params : dict
    _

    Returns
    -------
    dict
        Updated loss_params.
    """
    # loss_params["init_params"]["num_events"] = model_params["init_params"]["num_events"]
    # loss_params["init_params"]["max_time"] = model_params["init_params"]["max_time"]
    return loss_params


def set_survival_loss_params(loss_params: dict, model_params: dict, **_):
    """
    Set the number of events and the maximum time for the loss function.
    Parameters
    ----------
    loss_params : dict
    model_params : dict
    _

    Returns
    -------
    dict
        Updated loss_params.
    """
    loss_params["init_params"]["num_events"] = model_params["init_params"]["num_events"]
    loss_params["init_params"]["max_time"] = model_params["init_params"]["max_time"]
    return loss_params


def set_huber_mean_loss_params(loss_params: dict, train_set: object, **_):
    """
    Set the number of events and the maximum time for the loss function.
    Parameters
    ----------
    loss_params : dict
    train_set : object
    _

    Returns
    -------
    dict
        Updated loss_params.
    """
    loss_params["init_params"]["delta"] = 10**(-((train_set.decimals//2) + 1 ))
    return loss_params


if __name__ == "__main__":
    pass
