import pandas as pd
from typing import List, Tuple, Union

__all__ = ["get_sets_with_idx", "get_sets_with_ID"]


def get_sets_with_idx(data: Union[pd.DataFrame, List[pd.DataFrame]], *sets: pd.DataFrame, labels: Union[pd.DataFrame, List[pd.DataFrame]] = None, cv_info: Union[pd.DataFrame, List[pd.DataFrame]] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    This function returns the data and labels of the sets passed as arguments.
    Parameters
    ----------
    data : pd.DataFrame or list[pd.DataFrame]
    sets : pd.DataFrame
    labels : pd.DataFrame
    cv_info : pd.DataFrame

    Returns
    -------
    list
        List of data and labels of the sets passed as arguments.
    """
    sets_data = []
    for fset in sets:
        if isinstance(data, list):
            sets_data.append([dataset.iloc[fset.idx] for dataset in data])
            if labels is not None:
                sets_data.append([dataset_labels.iloc[fset.idx] for dataset_labels in labels])
            if cv_info is not None:
                sets_data.append([dataset_cv_info.iloc[fset.idx] for dataset_cv_info in cv_info])
        else:
            sets_data.append(data.iloc[fset.idx])
            if labels is not None:
                sets_data.append(labels.iloc[fset.idx])
            if cv_info is not None:
                sets_data.append(cv_info.iloc[fset.idx])

    return sets_data


def get_sets_with_ID(data: Union[pd.DataFrame, List[pd.DataFrame]], *sets: pd.DataFrame, labels: Union[pd.DataFrame, List[pd.DataFrame]] = None, cv_info: Union[pd.DataFrame, List[pd.DataFrame]] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    This function returns the data and labels of the sets passed as arguments.
    Parameters
    ----------
    data : pd.DataFrame or list[pd.DataFrame]
    sets : pd.DataFrame
    labels : pd.DataFrame
    cv_info : pd.DataFrame

    Returns
    -------
    list
        List of data and labels of the sets passed as arguments.
    """
    sets_data = []
    for fset in sets:
        if isinstance(data, list):
            sets_data.append([dataset.loc[fset.ID] for dataset in data])
            if labels is not None:
                sets_data.append([dataset_labels.loc[fset.ID] for dataset_labels in labels])
            if cv_info is not None:
                sets_data.append([dataset_cv_info.loc[fset.ID] for dataset_cv_info in cv_info])
        else:
            sets_data.append(data.loc[fset.ID])
            if labels is not None:
                sets_data.append(labels.loc[fset.ID])
            if cv_info is not None:
                sets_data.append(cv_info.loc[fset.ID])

    return sets_data


if __name__ == "__main__":
    pass
