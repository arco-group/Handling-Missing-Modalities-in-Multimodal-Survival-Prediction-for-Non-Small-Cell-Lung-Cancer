import warnings

import numpy as np
from typing import Tuple
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import rankdata
from sklearn.preprocessing import quantile_transform

from .classification import label_to_discrete


def get_transform_labels(time, event, transform: str):
    if transform == "none":
        return event, time
    elif transform == "separate":
        return transform_separate(time, event)
    elif transform == "rank":
        return transform_rank_log(time, event)
    elif transform == "quantile":
        return transform_quantile(time, event)
    elif transform == "survival_probability":
        return transform_survival_probability(time, event)
    elif transform == "partial_hazard":
        return transform_partial_hazard(time, event)
    else:
        raise ValueError(f"Unknown transform: {transform}")


__all__ = ["label_to_survival", "survival_to_label",
           "transform_survival_probability", "transform_partial_hazard",
           "transform_separate", "transform_rank_log", "transform_quantile"]


def label_to_survival( label, classes: tuple, max_time: int) -> np.ndarray:
    event = label[0]  # label_to_discrete( label[0], classes )  #

    floor_time = np.floor(label[1])
    time = min(floor_time, max_time-1)



    # if alive after max time -> censored
    if event != classes[0] and time < floor_time:
        event = classes[0]

    return np.array([ label_to_discrete(event, classes), time ], dtype=int)


def survival_to_label( label, classes: tuple, max_time: int, **_ ) -> Tuple[str, int]:
    # assert len(label) == len( classes ) * max_time

    time = np.argmax( label )
    event = 0
    while time > max_time:
        time -= max_time
        event += 1
    return classes[ event ], time


def transform_survival_probability(time, event):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times

    From https://www.kaggle.com/code/cdeotte/gpu-lightgbm-baseline-cv-681-lb-685
    """
    kmf = KaplanMeierFitter()
    kmf.fit(time, event)
    y = kmf.survival_function_at_times(time).values
    return y


def transform_partial_hazard(time, event):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times

    From https://www.kaggle.com/code/andreasbis/cibmtr-eda-ensemble-model
    """
    data = pd.DataFrame({'efs_time': time, 'efs': event, 'time': time, 'event': event})
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(data, duration_col='time', event_col='event')
    return cph.predict_partial_hazard(data)


def transform_separate(time, event):
    """Transform the target by separating events from non-events

    From https://www.kaggle.com/code/mtinti/cibmtr-lofo-feature-importance-gpu-accelerated"""
    transformed = time.copy()
    mx = transformed[event == 1].max()  # last patient who dies
    mn = transformed[event == 0].min()  # first patient who survives
    transformed[event == 0] = time[event == 0] + mx - mn
    transformed = rankdata(transformed)
    transformed[event == 0] += len(transformed) // 2
    transformed = transformed / transformed.max()
    return - transformed


def transform_rank_log(time, event):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times

    From https://www.kaggle.com/code/cdeotte/nn-mlp-baseline-cv-670-lb-676"""
    transformed = time.copy()
    mx = transformed[event == 1].max()  # last patient who dies
    mn = transformed[event == 0].min()  # first patient who survives
    transformed[event == 0] = time[event == 0] + mx - mn
    transformed = rankdata(transformed)
    transformed[event == 0] += len(transformed) * 2
    transformed = transformed / transformed.max()
    transformed = np.log(transformed)
    return - transformed


def transform_quantile(time, event):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times

    From https://www.kaggle.com/code/ambrosm/esp-eda-which-makes-sense"""
    transformed = np.full(len(time), np.nan)
    transformed_dead = quantile_transform(- time[event == 1].reshape(-1, 1)).ravel()
    transformed[event == 1] = transformed_dead
    transformed[event == 0] = transformed_dead.min() - 0.3
    return transformed







if __name__ == "__main__":
    pass
