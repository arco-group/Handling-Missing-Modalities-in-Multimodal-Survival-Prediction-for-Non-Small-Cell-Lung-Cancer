"""
To evaluate the equitable prediction of transplant survival outcomes,
we use the concordance index (C-index) between a series of event
times and a predicted score across each race group.

It represents the global assessment of the model discrimination power:
this is the model’s ability to correctly provide a reliable ranking
of the survival times based on the individual risk scores.

The concordance index is a value between 0 and 1 where:

0.5 is the expected result from random predictions,
1.0 is perfect concordance (with no censoring, otherwise <1.0),
0.0 is perfect anti-concordance (with no censoring, otherwise >0.0)

"""

import pandas as pd
import pandas.api.types
import numpy as np
from lifelines.utils import concordance_index
import logging

log = logging.getLogger(__name__)

__all__ = ['c_index']


class ParticipantVisibleError(Exception):
    pass


def c_index(y_true, y_score, race_group, **_):
    """
    Compute the stratified concordance index
    Parameters
    ----------
    y_true : np.ndarray
    y_score : np.ndarray
    """
    # y_score_idx = np.argmax(y_score, axis=1)
    # max_mask = np.arange(y_score.shape[1]) < y_score_idx[:, None]
    # y_score = np.sum(y_score * max_mask, axis=1)
    # y_score = np.expand_dims(y_score, axis=1)
    y_score = (np.sum(y_score, axis=1, keepdims=True) / y_score.shape[1])  # 1 -
    y_pred = pd.DataFrame({"prediction": y_score.tolist()})

    y_true = pd.DataFrame({"efs": y_true[:, 0].tolist(), "efs_time": y_true[:, 1].tolist(), "race_group": race_group.tolist()})
    y_true.insert(0, "ID", range(len(y_true)))
    y_pred.insert(0, "ID", range(len(y_pred)))
    y_pred.loc[:, "prediction"] = y_pred["prediction"].explode()
    y_true.loc[:, "race_group"] = y_true["race_group"].explode()

    y_true2 = y_true.copy()
    y_pred2 = y_pred.copy()
    y_pred2.loc[:, "prediction"] = 1 - y_pred2["prediction"]

    y_true3 = y_true.copy()
    y_pred3 = y_pred.copy()
    y_pred3.loc[:, "prediction"] = - y_pred3["prediction"]

    a = score(y_true, y_pred.astype(float), "ID")
    #b = score(y_true2, y_pred2.astype(float), "ID")
    #c = score(y_true3, y_pred3.astype(float), "ID")
    # log.info(f"c_index: score {a}")
    # log.info(f"c_index: 1-score {b}")
    # log.info(f"c_index: -score {c}")
    return a  # [a, b, c]
    # return score(y_true, y_pred.astype(float), "ID")


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    > import pandas as pd
    > row_id_column_name = "id"
    > y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    > y_pred = pd.DataFrame(y_pred)
    > y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    > y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    > y_true = pd.DataFrame(y_true)
    > y_true.insert(0, row_id_column_name, range(len(y_true)))
    > score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                merged_df_race[interval_label],
                -merged_df_race[prediction_label],
                merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
