from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

__all__ = ["RMSE", "MAE", "MSE", "R2", "EVS"]


def RMSE(y_true, y_pred, sample_weight=None, multioutput='uniform_average', **_):
    return root_mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)


def MAE(y_true, y_pred, sample_weight=None, multioutput='uniform_average', **_):
    return mean_absolute_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)


def MSE(y_true, y_pred, sample_weight=None, multioutput='uniform_average', **_):
    return mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)


def R2(y_true, y_pred, sample_weight=None, multioutput='uniform_average', **_):
    return r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)


def EVS(y_true, y_pred, sample_weight=None, multioutput='uniform_average', **_):
    return explained_variance_score(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)


if __name__ == "__main__":
    pass
