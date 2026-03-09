import logging
import numpy as np
import pandas as pd
from typing import Union, Tuple

from .miscellaneous import get_unique_values
from category_encoders.leave_one_out import LeaveOneOutEncoder

log = logging.getLogger(__name__)

__all__ = ["categorical_preprocessing"]

def one_hot_encode(table: pd.DataFrame, unique_values: Union[ dict, pd.Series ] = None,
                   nan_as_category: bool = False, fill_value: int = None,
                   return_categories: bool = False, **kwargs) -> Union[ pd.DataFrame, Tuple[ pd.DataFrame, dict ] ]:
    """
    One-hot encode a Pandas DataFrame.

    Parameters:
        table: pd.DataFrame.
            A Pandas DataFrame.
        unique_values: dict | pd.Series, optional.
            A dictionary or Pandas Series containing the unique values for each column in the DataFrame.
            If not provided, this is obtained by calling the `get_unique_values` function on `table`.
        nan_as_category: bool, default = False.
            A boolean indicating whether to include a separate column for `NaN` values in the resulting
            one-hot encoded DataFrame.
        fill_value: int or None, optional.
            The value to use for filling missing values if `nan_as_category` is `False`, if None NaN values
            are preserved.
        return_categories: bool, default = False.
            A boolean indicating whether to return the unique_values or not.
        kwargs:
            Additional keyword arguments to pass on to the 'get_unique_values' function.

    Returns:

    table_1hot: pd.DataFrame.
        A one-hot encoded version of the input DataFrame, where each categorical column is replaced by
        multiple binary columns, one for each category. If `nan_column` is `True`, it includes a separate
        column for `NaN` values in the resulting DataFrame. If `fill_missing` is `True`, it fills missing
        values with the specified `fill_value`.
    unique_values: dict, optional.
        If return_params is True, returns a tuple containing the one-hot encoded version of the input
        DataFrame and the dict containing the unique values for each column in the DataFrame.
    """

    # If `unique_values` is not provided, get it by calling the `get_unique_values` function on `table`
    if unique_values is None:
        unique_values = get_unique_values(table, nan_as_category=nan_as_category, **kwargs)

    if isinstance(unique_values, pd.Series):
        unique_values = unique_values.to_dict()

    # Make a copy of the input DataFrame
    table_1hot = table.copy()

    columns_to_drop = []
    # Iterate over each column in the DataFrame
    for column_name, categories in unique_values.items():
        if not nan_as_category and any(pd.isna(categories)):
            categories = list(filter(lambda x: not pd.isna(x), categories))  # categories[ np.logical_not(np.isnan(categories)) ]

        if len(categories) <= 2:
            for cat_idx, category in enumerate(categories):
                # Set the values in the new column to the category index if the corresponding value in the original column is equal to `category`, and 0 otherwise
                table_1hot.loc[table_1hot.loc[:, column_name] == category, column_name] = cat_idx
        else:
            columns_to_drop.append( column_name )
            # Iterate over each category in the current column
            for category in categories:
                # Create a new column with the name `column_name_category`
                new_column_name = f"{column_name}_{category}"

                new_column = pd.DataFrame({new_column_name: (table_1hot[column_name].values == category).astype(int)}, index=table_1hot.index)
                table_1hot = pd.concat( [table_1hot, new_column], axis=1 )
                # Set the values in the new column to 1 if the corresponding value in the original column is equal to `category`, and 0 otherwise
                #table_1hot[new_column_name] = table_1hot[column_name].values == category
                #table_1hot[new_column_name] = table_1hot[new_column_name].astype(int)
                # Set the values in the new column to `NaN` if the corresponding value in the original column is `NaN`
                # table_1hot[new_column_name] = np.where(np.isnan(table_1hot[column_name]), np.nan, table_1hot[new_column_name])
                table_1hot[new_column_name] = np.where(table_1hot[column_name].isna(), np.nan, table_1hot[new_column_name])

        # If `nan_as_category` is True and there are `NaN` values in the current column
        if nan_as_category and table_1hot.loc[:, column_name].isna().any(axis=None):
            # Create a new column with the name `column_name_nan`
            table_1hot[f"{column_name}_nan"] = table_1hot[column_name].isna().astype(int)

    # Drop the original columns from the DataFrame
    table_1hot = table_1hot.drop(columns_to_drop, axis=1)

    if nan_as_category:
        fill_value = 0
    if fill_value not in ( None, np.nan ):
        # Fill missing values (`NaN`) in the DataFrame with the specified `fill_value`
        table_1hot = table_1hot.fillna( fill_value ).astype(int)

    parameters = dict(method="onehotencode", unique_values=unique_values, nan_as_category=nan_as_category, fill_value=fill_value )
    output_options = {False: table_1hot, True: (table_1hot, parameters)}
    return output_options[return_categories]


def categorical_encode(table: pd.DataFrame, unique_values: Union[dict, pd.Series] = None, nan_as_category: bool = False,
                       fill_value: int = None, return_categories: bool = False, **kwargs) -> Union[ pd.DataFrame, Tuple[ pd.DataFrame, dict]]:
    """
    Categorical encodes a Pandas DataFrame.

    Parameters:
        table: pd.DataFrame.
            A Pandas DataFrame
        unique_values: dict | pd.Series, optional.
            A Pandas Series containing the unique values for each column in the DataFrame. If not provided, this is obtained by calling the `get_unique_values` function on `table`.
        nan_as_category: bool, default = False.
            A boolean indicating whether to include `NaN` values as a separate category in the resulting categorical encoded DataFrame. Default is False.
        fill_value: int, optional.
            The value to use for filling missing values if `nan_as_category` is `False`, if None 'NaN' values are preserved.
        return_categories: bool, default = False.
            A boolean indicating whether to return the unique_values or not.
        kwargs:
            Additional keyword arguments to pass on to the 'get_unique_values' function.

    Returns:

    table_encoded: pd.DataFrame.
        A categorical encoded version of the input DataFrame, where each categorical column is encoded with
        successive numbers, one for each category. If `nan_column` is `True`, it includes `NaN` values as a
        separate category in the resulting DataFrame. If `fill_missing` is `True`, it fills missing values
        with the specified `fill_value`.
    unique_values: dict, optional.
        If return_params is True, returns a tuple containing the categorical encoded version of the input
        DataFrame and the dict containing the unique values for each column in the DataFrame.
    """
    # If `unique_values` is not provided, get it by calling the `get_unique_values` function on `table`
    if unique_values is None:
        unique_values = get_unique_values(table, nan_as_category=nan_as_category, **kwargs)

    if isinstance(unique_values, pd.Series):
        unique_values = unique_values.to_dict()

    # Make a copy of the input DataFrame
    table_encoded = table.copy()

    # Iterate over each column in the DataFrame
    for column_name, categories in unique_values.items():

        if not nan_as_category and any(pd.isna(categories)):
            categories = list(filter(lambda x: not pd.isna(x), categories))

        # Iterate over each category in the current column
        for cat_idx, category in enumerate(categories):
            # Set the values in the new column to the category index if the corresponding value in the original column is equal to `category`, and 0 otherwise
            table_encoded.loc[ table[column_name] == category, column_name ] = cat_idx

        if nan_as_category:
            fill_value = (~pd.isna(categories)).sum()

        if fill_value is not None:
            # Set the values in the new column to `NaN` if the corresponding value in the original column is `NaN`
            # table_encoded[column_name] = np.where(np.isnan(table[column_name]), fill_value, table_encoded[column_name])
            # table_encoded[column_name] = table_encoded[column_name].where(table[column_name].isna(), fill_value, table_encoded[column_name])
            # table_encoded[column_name].where(table[column_name].notna(), fill_value, inplace=True)
            table_encoded[column_name] = table_encoded[column_name].mask(table[column_name].isna(), fill_value)

    parameters = dict(method= "categoricalencode", unique_values=unique_values, nan_as_category=nan_as_category, fill_value=fill_value)
    output_options = {False: table_encoded, True: (table_encoded, parameters)}
    return output_options[return_categories]


def leave_one_out_encode(table: pd.DataFrame, unique_values: Union[dict, pd.Series] = None, nan_as_category: bool = False,
                       fill_value: int = None, return_categories: bool = False,  categorical_encoder = None,
                       verbose: int = 0, drop_invariant: bool = False, return_df: bool = True, handle_unknown: str = "value",
                       random_state: int = None, sigma: float = None, labels: pd.Series = None, **kwargs) -> Union[ pd.DataFrame, Tuple[ pd.DataFrame, dict]]:
    """
    Leave one out encodes a Pandas DataFrame.

    Parameters:
        table: pd.DataFrame.
            A Pandas DataFrame
        unique_values: dict | pd.Series, optional.
            A Pandas Series containing the unique values for each column in the DataFrame. If not provided, this is obtained by calling the `get_unique_values` function on `table`.
        nan_as_category: bool, default = False.
            A boolean indicating whether to include `NaN` values as a separate category in the resulting categorical encoded DataFrame. Default is False.
        fill_value: int, optional.
            An integer indicating whether to fill missing values with the target mean (1) or leave a NaN (0).
        return_categories: bool, default = False.
            A boolean indicating whether to return the unique_values or not.
        categorical_encoder: encoder, default = None.
            The fitted categorical encoder to be used, if provided.
        verbose: int, default = 0.
            An integer indicating verbosity of the output. 0 for none.
        drop_invariant: bool, default = False.
            A boolean indicating whether to drop columns with 0 variance or not.
        return_df: bool, default True.
            A boolean indicating whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
        handle_unknown: str, default = "value"
            Options are ‘error’, ‘return_nan’ and ‘value’, defaults to ‘value’, which returns the target mean.
        random_state: int, default = None.
            An integer used to reproduce the output.
        sigma: float, default = 0.
            adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
            Sigma gives the standard deviation (spread or “width”) of the normal distribution. The optimal value is commonly between 0.05 and 0.6.
            The default is to not add noise, but that leads to significantly suboptimal results.
        labels: pd.Series, default = None.
            A pandas Series containing the target variables.

        kwargs:
            Additional keyword arguments to pass on to the 'get_unique_values' function.

    Returns:

    table_encoded: pd.DataFrame.
        A categorical encoded version of the input DataFrame, where each categorical column is encoded with
        successive numbers, one for each category. If `nan_column` is `True`, it includes `NaN` values as a
        separate category in the resulting DataFrame. If `fill_missing` is `True`, it fills missing values
        with the specified `fill_value`.
    unique_values: dict, optional.
        If return_params is True, returns a tuple containing the categorical encoded version of the input
        DataFrame and the dict containing the unique values for each column in the DataFrame.
    """

    # If `unique_values` is not provided, get it by calling the `get_unique_values` function on `table`
    if unique_values is None:
        unique_values = get_unique_values(table, nan_as_category=nan_as_category, **kwargs)

    if isinstance(unique_values, pd.Series):
        unique_values = unique_values.to_dict()

    # Make a copy of the input DataFrame
    table_encoded = table.copy()

    cols = list(unique_values.keys())
    if categorical_encoder is None:
        handle_missing = "value" if not nan_as_category and fill_value == 1 else "return_nan"
        categorical_encoder = LeaveOneOutEncoder(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                                                 handle_unknown=handle_unknown, handle_missing=handle_missing, random_state=random_state, sigma=sigma)
        categorical_encoder = categorical_encoder.fit(table_encoded, labels)

    table_encoded = categorical_encoder.transform(table_encoded)

    if nan_as_category and table_encoded.isna().any(axis=None):
        for column_name in cols:
            # Create a new column with the name `column_name_nan`
            table_encoded[f"{column_name}_nan"] = table_encoded[column_name].isna().astype(int)

    parameters = dict(method= "leaveoneoutencode", unique_values=unique_values, nan_as_category=nan_as_category,
                      fill_value=fill_value, categorical_encoder=categorical_encoder, verbose=verbose,
                      drop_invariant=drop_invariant, return_df=return_df, handle_unknown=handle_unknown, random_state=random_state, sigma=sigma)

    output_options = {False: table_encoded, True: (table_encoded, parameters)}
    return output_options[return_categories]


def categorical_preprocessing(table: pd.DataFrame, method: str = "onehotencode", **kwargs) -> Union[ pd.DataFrame, Tuple[ pd.DataFrame, dict]]:
    """
    Applies categorical preprocessing to a Pandas DataFrame.

    Parameters:
    table: pd.DataFrame.
        The DataFrame to preprocess.
    method: str, default = 'one_hot_encode'.
        The method to use for preprocessing. Must be "one_hot_encode" or "categorical_encode". Default is "one_hot_encode".
    kwargs:
        Additional keyword arguments to pass to the preprocessing function.

    Returns:

    output: pd.DataFrame.
        The preprocessed DataFrame.
    parameters: dict, optional.
        If return_params is True, returns a tuple containing the preprocessed DataFrame and the preprocessing parameters.
    """
    # Define a dictionary mapping method names to functions
    options = {"onehotencode": one_hot_encode, "categoricalencode": categorical_encode }  # , "leave_one_out_encode": leave_one_out_encode}
    # Select the appropriate function based on the method name
    output = options[method](table, **kwargs)

    log.info("Categorical preprocessing done")

    return output


if __name__ == "__main__":
    pass
