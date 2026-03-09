from .classification import *
from .regression import *
from .survival import *


def get_transform_labels(time, event, transform: dict):

    transform = transform.get("transform", "none")
    if transform == "none":
        return event, time
    elif transform == "separate":
        return event, transform_separate(time, event)
    elif transform == "rank":
        return event, transform_rank_log(time, event)
    elif transform == "quantile":
        return event, transform_quantile(time, event)
    elif transform == "survival_probability":
        return event, transform_survival_probability(time, event)
    elif transform == "partial_hazard":
        return event, transform_partial_hazard(time, event)
    else:
        raise ValueError(f"Unknown transform: {transform}")
