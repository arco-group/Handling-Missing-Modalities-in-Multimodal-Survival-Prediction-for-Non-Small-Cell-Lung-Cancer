
import logging
import os

import numpy as np
import pandas as pd
from typing import Tuple

import torch
from statsmodels.stats.descriptivestats import pd_ptp

from .tables import load_table

__all__ = ["load_tabular_dataset"]

log = logging.getLogger(__name__)

def load_features_dataset(data_paths: [], columns: dict, **kwargs ) -> pd.DataFrame:
    """
    Loads a features dataset.

    Parameters:

    """
    loading_dict = {'.npy': np.load, '.pt': torch.load}
    Ids = data_paths.index.astype(str).tolist() if data_paths is not None else []
    loaded_array = [loading_dict[os.path.splitext(data_path)[1]](data_path, allow_pickle=True) if data_path == data_path else None for data_path in data_paths]
    shape = np.array([arr.shape for arr in loaded_array if arr is not None][0])[0]
    loaded_array = [ arr if arr is not None else np.full(shape, np.nan) for arr in loaded_array]




    features_loaded = np.stack(loaded_array)
    for i in range(features_loaded.shape[1]):
        columns[f'feat_{i}'] = 'float'

    features_df = pd.DataFrame(features_loaded, columns=[f'feat_{i}' for i in range(features_loaded.shape[1])], index=pd.Series(Ids, name=[column for column in columns.keys() if columns[column] == 'id'][0] if 'id' in columns.values() else 'ID'))


    return features_df, columns





def load_tabular_dataset(path: str, columns: dict, task: str, **kwargs ) -> Tuple[ pd.DataFrame, pd.DataFrame, pd.Series ]:
    """
    Loads a tabular dataset.

    Parameters:
        path: str
            Path to the folder containing the dataset.
        columns: dict
            Dictionary containing the features types.
        task: str

    Returns:

    data: pd.DataFrame.
        Pandas DataFrame containing the features.
    labels: pd.DataFrame.
        Pandas DataFrame containing the features.
    columns_info: pd.Series.
        Pandas Series containing the features types or columns info.
    """

    load_params = kwargs.get("pandas_load_kwargs", None)
    load_params = {} if load_params is None else load_params

    if load_params.get("header", None) is None and load_params.get("names", None) is None:
        columns_names = list(columns.keys())
    else:
        columns_names = load_params.get("names", None)

    data = load_table( path, names=columns_names, **load_params )

    db_type = kwargs.get("db_type", "tabular")
    id_col = [column_name for column_name, column_type in columns.items() if column_type == "id"]


    target_cols = {column_name: column_type for column_name, column_type in columns.items() if column_type.startswith("target") }
    target_name = [ column_name for column_name, column_type in target_cols.items() ]

    data = data[columns.keys()]
    columns = {column_name: columns[column_name] for column_name in data.columns}

    if "id" in list(columns.values()) and load_params.get("index_col", None) is None:
        data = data.set_index(id_col)
        for col in id_col:
            del columns[col]

    data = data.rename_axis(["ID"] + id_col[1:], axis=0)

    if db_type == "features":
        path_column = [ column_name for column_name, column_type in columns.items() if column_type == "path" ][0]


        data_paths = data[path_column].copy() if path_column in data.columns else None
        features_loaded, columns = load_features_dataset(data_paths, columns, data_path=data_paths)
        if path_column is not None and path_column in data.columns:
            columns.pop(path_column)
            data.drop(path_column, axis=1, inplace=True)
        features_loaded.index = features_loaded.index.astype(int)
        data = data.join(features_loaded, how="outer")



    if task != "multilabelclassification" and db_type != "features":
        data = data.dropna(subset=target_name, how="any")

    labels = data.loc[:, target_name].reset_index()
    data = data.drop(target_name, axis=1).reset_index()

    if task != "multilabelclassification":
        label_col = target_name if isinstance(target_name, str) else target_name[0]
        labels = labels.rename({label_col: "label"}, axis=1)

    for col, col_type in target_cols.items():
        if col_type == "target_event_time":
            labels = labels.rename({col: "event_time"}, axis=1)

    data["ID"] = data.ID.astype(str)
    labels["ID"] = labels.ID.astype(str)

    data.set_index("ID", inplace=True)
    labels.set_index("ID", inplace=True)

    columns_info = pd.Series(columns, name="dtype").drop(target_name)
    log.info("Data loaded")

    return data, labels, columns_info


if __name__ == "__main__":
    pass
