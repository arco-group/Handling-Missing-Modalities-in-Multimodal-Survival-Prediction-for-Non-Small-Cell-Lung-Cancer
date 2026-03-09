import os
import logging
import itertools
import numpy as np
import pandas as pd
from omegaconf import ListConfig
from CMC_utils.miscellaneous import do_nothing, longest_common_substring
from CMC_utils import save_load

from typing import Union, Tuple, List

__all__ = ["generate_missing", "generate_multimodal_missing", "compute_missing_features_n_modalities"]

log = logging.getLogger(__name__)


def set_to_nullable_type(data: pd.DataFrame):
    for col, dtype in zip(data.columns, data.dtypes):
        if dtype == "int16":
            data[col] = data[col].astype("Int16")
        elif dtype == "int32":
            data[col] = data[col].astype("Int32")
        elif dtype == "int64":
            data[col] = data[col].astype("Int64")
    return data


def MCAR_vector_mask(data: Union[pd.Series, np.array], missing_fraction: float, index_direction: int) -> np.array:
    """
    Generate a mask for a vector of data, where the missing values are randomly distributed according to a missing fraction
    Parameters
    ----------
    data : Union[pd.Series, np.array]
    missing_fraction : float
    index_direction : int

    Returns
    -------
    np.array
    """
    data_length = len(data)
    values_to_remove = int(np.floor(missing_fraction * data_length))
    values_already_missing = data.isna().values
    values_to_remove = np.maximum( values_to_remove - values_already_missing.sum(), 0 )

    missing_mask = np.hstack([np.ones(values_to_remove), np.zeros(data_length - values_to_remove)])  # ones to be removed, zeros to keep

    np.random.shuffle(missing_mask)

    missing_mask = missing_mask + values_already_missing
    missing_masked = np.where(missing_mask == 2)[0]
    not_masked = np.where(missing_mask == 0)[0]

    not_masked_to_mask = np.random.choice(not_masked, size=len(missing_masked), replace=False)
    missing_mask[not_masked_to_mask] = 1
    missing_mask[missing_masked] = 1

    missing_mask = np.expand_dims( missing_mask.astype(bool), axis=index_direction )

    return missing_mask


def check_uncorrectable_vectors(data: pd.DataFrame, mask: np.array, axis_to_check: int) -> np.array:
    not_missing_mask = data.notna()

    main_axis_vectors_with_one_value_map = not_missing_mask.sum(axis=axis_to_check) == 1
    main_axis_completely_masked_vectors = mask.all(axis=axis_to_check)

    vector_idxs_to_correct = np.where(main_axis_vectors_with_one_value_map & main_axis_completely_masked_vectors)[0]
    usable_idxs = np.where(~main_axis_vectors_with_one_value_map & ~main_axis_completely_masked_vectors)[0]

    if len(vector_idxs_to_correct) > 0:

        idxs_options = {0: (np.arange(data.shape[0]), vector_idxs_to_correct), 1: (vector_idxs_to_correct, np.arange(data.shape[1]))}

        second_axis_idxs_to_check = np.unique(not_missing_mask.iloc[idxs_options[axis_to_check]].sum(axis=int(not axis_to_check)).values.nonzero()[0])

        for idx in second_axis_idxs_to_check:
            not_missing_idxs_options = {0: (idx, usable_idxs), 1: (usable_idxs, idx)}
            usable_values_map = ~mask[not_missing_idxs_options[axis_to_check]]

            missing_idxs_options = {0: (idx, vector_idxs_to_correct), 1: (vector_idxs_to_correct, idx)}
            missing_values_map = data.iloc[missing_idxs_options[axis_to_check]].notna()

            values_to_correct = missing_values_map.sum() - usable_values_map.sum()

            if values_to_correct > 0:
                available_idxs_to_correct = list(missing_values_map.iloc[ missing_values_map.values.nonzero()[0] ].index)
                idxs_to_correct = np.random.choice(available_idxs_to_correct, size=values_to_correct, replace=False)
                idxs_options_to_correct = {0: (idx, idxs_to_correct), 1: (idxs_to_correct, idx)}
                mask[idxs_options_to_correct[axis_to_check]] = False

    return mask


def mask_correction(mask: np.array, axis_to_check: int, data: pd.DataFrame) -> np.array:
    """
    Correct the mask to avoid completely missing rows or columns
    Parameters
    ----------
    mask : np.array
    axis_to_check : int
    data : np.array

    Returns
    -------
    np.array
    """
    completely_missing_map = mask.all(axis=axis_to_check)

    if completely_missing_map.any():

        completely_missing_idxs = completely_missing_map.nonzero()[0]

        # selectable_second_axis_idxs = ((~mask).sum(axis=int(not axis_to_check)) > 1).nonzero()[0]

        for missing_idx in completely_missing_idxs:
            selectable_second_axis_idxs = (np.invert(mask).sum(axis=int(not axis_to_check)) > 1).nonzero()[0]

            element_corrected = False
            while not element_corrected and len(selectable_second_axis_idxs) > 0:
                selected_second_axis_idx = np.random.choice( selectable_second_axis_idxs )

                axis_second_condition_options = {0: (selected_second_axis_idx, np.arange(mask.shape[1])), 1: (np.arange(mask.shape[0]), selected_second_axis_idx)}
                if axis_to_check == 0:
                    vector = data.iloc[:, missing_idx]
                else:
                    vector = data.iloc[missing_idx, :]

                change_index = vector.isna().iloc[selected_second_axis_idx]
                if change_index:
                    selectable_second_axis_idxs = np.delete(selectable_second_axis_idxs, np.where(selectable_second_axis_idxs == selected_second_axis_idx))
                    continue

                first_condition_selectable_map = np.invert(mask).sum( axis=axis_to_check ) > 1
                second_condition_selectable_map = np.invert(mask)[axis_second_condition_options[axis_to_check]]
                third_condition_selectable_map = np.invert(data.iloc[axis_second_condition_options[axis_to_check]].isna())
                selectable_main_axis_idxs = np.vstack( [first_condition_selectable_map, second_condition_selectable_map, third_condition_selectable_map] ).all(axis=0).nonzero()[0]

                if len(selectable_main_axis_idxs) == 0:
                    selectable_second_axis_idxs = np.delete(selectable_second_axis_idxs, np.where(selectable_second_axis_idxs == selected_second_axis_idx))
                else:
                    selected_main_axis_idx = np.random.choice( selectable_main_axis_idxs )

                    to_be_deleted_options = {0: (selected_second_axis_idx, selected_main_axis_idx), 1: (selected_main_axis_idx, selected_second_axis_idx)}
                    to_be_corrected_options = {0: (selected_second_axis_idx, missing_idx), 1: (missing_idx, selected_second_axis_idx)}

                    mask[to_be_deleted_options[axis_to_check]] = True
                    mask[to_be_corrected_options[axis_to_check]] = False

                    element_corrected = True
            assert element_corrected, "There are too few not-missing values to correct the mask"

    return mask


def MCAR_feature(data: pd.DataFrame, missing_fraction: float, printing: bool = True, **_) -> Tuple[pd.DataFrame, np.array]:
    """
    Generate a mask for a vector of data, where the missing values are randomly distributed according to a missing fraction
    Parameters
    ----------
    data : Union[pd.Series, np.array]
    missing_fraction : float
    printing : bool

    Returns
    -------
    np.array
    """
    already_completely_missing_features = data.isna().all(axis=0).any() or data.shape[1] == 1
    masked_data = data.copy()

    generation_direction = 1

    missing_mask = np.array([], dtype=bool).reshape(masked_data.shape[0], 0)

    for feature_idx in range(masked_data.shape[generation_direction]):
        missing_vector_mask = MCAR_vector_mask(masked_data.iloc[:, feature_idx], missing_fraction, index_direction=generation_direction)
        missing_mask = np.hstack([missing_mask, missing_vector_mask])

    if not already_completely_missing_features:
        # samples_with_one_value = (masked_data.notna().sum(axis=generation_direction) == 1).astype(int).values.nonzero()[0]
        # respective_features_represented = (masked_data.iloc[samples_with_one_value].notna().sum(axis=int(not generation_direction)) > 0).astype(int).values.nonzero()[0]

        # enough_not_missing_values = min(np.floor((masked_data.shape[0] - len(samples_with_one_value)) * (masked_data.shape[1] - len(respective_features_represented)) * (1 - missing_fraction)), masked_data.notna().sum().sum()) >= max(masked_data.shape[0] - len(samples_with_one_value), masked_data.shape[1] - len(respective_features_represented)) + len(samples_with_one_value)

        # if enough_not_missing_values:
        #     missing_mask = check_uncorrectable_vectors(masked_data, missing_mask, generation_direction)
        missing_mask = mask_correction(missing_mask, generation_direction, masked_data)

    masked_data = masked_data.mask(missing_mask, np.nan)

    final_missing_percentage = np.mean( np.round((masked_data.isna().sum(axis=0) / masked_data.shape[0]) * 100) )
    print_options = {True: log.info, False: do_nothing}
    print_options[printing](f"{final_missing_percentage}% of column data is missing on average")

    return masked_data, final_missing_percentage


def MCAR_sample(data: pd.DataFrame, missing_fraction: float, printing: bool = True, **_) -> Tuple[pd.DataFrame, np.array]:
    """
    Generate a mask for a vector of data, where the missing values are randomly distributed according to a missing fraction
    Parameters
    ----------
    data : Union[pd.Series, np.array]
    missing_fraction : float
    printing : bool

    Returns
    -------
    np.array
    """
    already_completely_missing_samples = data.isna().all(axis=1).any() or data.shape[1] == 1
    masked_data = data.copy()

    generation_direction = 0

    missing_mask = np.array([], dtype=bool).reshape(0, masked_data.shape[1])

    for sample_idx in range(masked_data.shape[generation_direction]):
        missing_vector_mask = MCAR_vector_mask(masked_data.iloc[sample_idx, :], missing_fraction, index_direction=generation_direction)
        missing_mask = np.vstack([missing_mask, missing_vector_mask])

    if not already_completely_missing_samples:
        # features_with_one_value = (masked_data.notna().sum(axis=generation_direction) == 1).astype(int).values.nonzero()[0]
        # respective_sample_represented = (masked_data.iloc[:, features_with_one_value].notna().sum(axis=int(not generation_direction)) > 0).astype(int).values.nonzero()[0]

        # enough_not_missing_values = min(np.floor((masked_data.shape[0] - len(respective_sample_represented)) * (masked_data.shape[1] - len(features_with_one_value)) * (1 - missing_fraction)), masked_data.notna().sum().sum()) >= max(masked_data.shape[0] - len(respective_sample_represented), masked_data.shape[1] - len(features_with_one_value)) + len(features_with_one_value)

        # if enough_not_missing_values:
        #    missing_mask = check_uncorrectable_vectors(masked_data, missing_mask, generation_direction)
        missing_mask = mask_correction(missing_mask, generation_direction, masked_data)

    masked_data = masked_data.mask(missing_mask, np.nan)

    final_missing_percentage = np.mean( np.round((masked_data.isna().sum(axis=1) / masked_data.shape[1]) * 100) )
    print_options = {True: log.info, False: do_nothing}
    print_options[printing](f"{final_missing_percentage}% of row data is missing on average")

    return masked_data, final_missing_percentage


def MCAR_global(data: pd.DataFrame, missing_fraction: float, printing: bool = True, **_) -> Tuple[pd.DataFrame, np.array]:
    """
    Generate a mask for a vector of data, where the missing values are randomly distributed according to a missing fraction
    Parameters
    ----------
    data : Union[pd.Series, np.array]
    missing_fraction : float
    printing : bool

    Returns
    -------
    np.array
    """
    already_complete_missing_features = data.isna().all(axis=0).any() or data.shape[1] == 1
    already_completely_missing_samples = data.isna().all(axis=1).any() or data.shape[1] == 1

    masked_data = data.copy()

    missing_mask = MCAR_vector_mask(pd.Series(masked_data.to_numpy().flatten()), missing_fraction, index_direction=0)
    missing_mask = missing_mask.reshape(masked_data.shape)

    if not already_complete_missing_features:
        features_with_one_value = (masked_data.notna().sum(axis=0) == 1).astype(int).values.nonzero()[0]
        respective_sample_represented = (masked_data.iloc[:, features_with_one_value].notna().sum(axis=1) > 0).astype(int).values.nonzero()[0]

        enough_not_missing_values = min(np.floor( (masked_data.shape[0] - len(respective_sample_represented)) * (masked_data.shape[1] - len(features_with_one_value)) * (1-missing_fraction)), masked_data.notna().sum().sum()) >= max(masked_data.shape[0] - len(respective_sample_represented), masked_data.shape[1] - len(features_with_one_value)) + len(features_with_one_value)

        if not enough_not_missing_values:
            missing_mask = check_uncorrectable_vectors(masked_data, missing_mask, 0)
        missing_mask = mask_correction(missing_mask, 0, masked_data)

    if not already_completely_missing_samples:
        samples_with_one_value = (masked_data.notna().sum(axis=1) == 1).astype(int).values.nonzero()[0]
        respective_features_represented = (masked_data.iloc[samples_with_one_value].notna().sum(axis=0) > 0).astype(int).values.nonzero()[0]

        enough_not_missing_values = min(np.floor(( masked_data.shape[0] - len(samples_with_one_value)) * (masked_data.shape[1] - len(respective_features_represented)) * (1 - missing_fraction)), masked_data.notna().sum().sum()) >= max(masked_data.shape[0] - len(samples_with_one_value), masked_data.shape[1] - len(respective_features_represented)) + len(samples_with_one_value)

        if not enough_not_missing_values:
            missing_mask = check_uncorrectable_vectors(masked_data, missing_mask, 1)
        missing_mask = mask_correction(missing_mask, 1, masked_data)

    masked_data = masked_data.mask(missing_mask, np.nan)

    final_missing_percentage = np.round((masked_data.isna().sum().sum() / (masked_data.shape[0] * masked_data.shape[1]))*100)
    print_options = {True: log.info, False: do_nothing}
    print_options[printing]( f"{final_missing_percentage}% of data is missing" )

    return masked_data, final_missing_percentage


def no_generation(data: pd.DataFrame, printing: bool = True, **_) -> Tuple[pd.DataFrame, np.array]:
    final_missing_percentage = np.round((data.isna().sum().sum() / (data.shape[0] * data.shape[1])) * 100)
    print_options = {True: log.info, False: do_nothing}
    print_options[printing](f"{final_missing_percentage}% of data is missing")
    return data, final_missing_percentage


def compute_missing_features_n_modalities(*data: pd.DataFrame) -> np.array:
    missing_percentages = []
    for mod_id, modality in enumerate(data):
        missing_percentage = np.round((modality.isna().sum().sum() / (modality.shape[0] * modality.shape[1])) * 100)

        missing_modality_percentage = np.round(( modality.isna().all(axis=1).sum() / modality.shape[0] ) * 100)
        log.info(f"{missing_percentage}% of missing features for modality {mod_id}, {missing_modality_percentage}% of completely missing samples for modality {mod_id}")
        missing_percentages.append(tuple([missing_percentage, missing_modality_percentage]))
    return np.array(missing_percentages)


def generate_multimodal_missing(datasets: List[pd.DataFrame], datasets_names: List[str], method: str, missing_fraction: Union[float, list], apply_to: str = "features", seed: int = 42, **kwargs) -> Tuple[List[pd.DataFrame], np.array]:
    kwargs.update( {"printing": False} )
    dataset_common_name = longest_common_substring(datasets_names)
    masked_data = []
    if apply_to == "features":
        for i, (dataset_name, data) in enumerate(zip(datasets_names, datasets)):
            data, _ = generate_missing(data, dataset_name=dataset_name, method=method, missing_fraction=missing_fraction, seed=seed+i, **kwargs)
            masked_data.append(data)
    elif apply_to == "modalities":
        modalities_map = {dataset_name: dataset.isna().all(axis=1).astype(float).values for dataset_name, dataset in zip(datasets_names, datasets)}
        modalities_map = pd.DataFrame(modalities_map, index=datasets[0].index)
        modalities_map[(modalities_map == 1.0).values] = np.nan
        modalities_map, _ = generate_missing(modalities_map, dataset_name=dataset_common_name, method=method, missing_fraction=missing_fraction, seed=seed, **kwargs)
        modalities_map = modalities_map.fillna(1)
        for dataset_name, data in zip(datasets_names, datasets):
            mask = np.vstack([modalities_map[dataset_name].astype(bool).values for _ in range(data.shape[1])]).T
            masked_data.append( data.mask(mask, np.nan) )
    elif apply_to == "separate_modalities":
        modalities_map = {dataset_name: dataset.isna().all(axis=1).astype(float).values for dataset_name, dataset in zip(datasets_names, datasets)}
        modalities_map = pd.DataFrame(modalities_map, index=datasets[0].index)
        modalities_map.loc[(modalities_map == 1.0).values] = np.nan

        if not isinstance(missing_fraction, (list, ListConfig)):
            missing_fraction = [missing_fraction] * len(datasets_names)

        for dataset_name, fraction in zip(datasets_names, missing_fraction):
            modalities_map.loc[:, [dataset_name]], _ = generate_missing(modalities_map.loc[:, [dataset_name]], dataset_name=dataset_name, method=method, missing_fraction=fraction, seed=seed, **kwargs)
        modalities_map = modalities_map.fillna(1)
        for dataset_name, data in zip(datasets_names, datasets):
            mask = np.vstack([modalities_map[dataset_name].astype(bool).values for _ in range(data.shape[1])]).T
            masked_data.append(data.mask(mask, np.nan))
    elif apply_to == "separate_modalities_previously_missing":
        modalities_map = {dataset_name: dataset.isna().all(axis=1).astype(float).values for dataset_name, dataset in zip(datasets_names, datasets)}
        modalities_map = pd.DataFrame(modalities_map, index=datasets[0].index).astype(bool)
        modalities_map = (modalities_map)[~modalities_map]

        if not isinstance(missing_fraction, (list, ListConfig)):
            missing_fraction = [missing_fraction] * len(datasets_names)

        for dataset_name, fraction in zip(datasets_names, missing_fraction):
            modalities_map.loc[:, [dataset_name]], _ = generate_missing(modalities_map.loc[:, [dataset_name]], dataset_name=dataset_name, method=method, missing_fraction=fraction, seed=seed, **kwargs)
        modalities_map = modalities_map.fillna(1)
        for dataset_name, data in zip(datasets_names, datasets):
            mask = np.vstack([modalities_map[dataset_name].astype(bool).values for _ in range(data.shape[1])]).T
            masked_data.append(data.mask(mask, np.nan))
    elif apply_to == "all":
        modalities_columns = [data.columns for data in datasets]
        data = pd.concat(datasets, axis=1)
        data, _ = generate_missing(data, dataset_name=dataset_common_name, method=method, missing_fraction=missing_fraction, seed=seed, **kwargs)
        masked_data = [data[columns] for columns in modalities_columns]
    else:
        raise ValueError(f"apply_to must be one of ['features', 'modalities', 'all'], got {apply_to}")

    missing_percentages = compute_missing_features_n_modalities(*masked_data)

    return masked_data, missing_percentages

def generate_missing( data: pd.DataFrame, dataset_name: str, method: str, missing_fraction: float, fset: str, test_fold: str, val_fold: str, save_path: str, copy: bool = True, seed: int = 42, **kwargs ) -> Tuple[pd.DataFrame, np.array]:
    """
    Generate a mask for a vector of data, where the missing values are randomly distributed according to a missing fraction
    Parameters
    ----------
    data : Union[pd.Series, np.array]
    dataset_name : str
    method : str
    missing_fraction : float
    fset: str
    test_fold : str
    val_fold : str
    save_path : str
    copy : bool
    seed : int
    kwargs : dict

    Returns
    -------
    np.array
    """
    if copy:
        data = data.copy()
    missing_percentage = int(100 * missing_fraction)
    filename = f"{dataset_name}_{test_fold}_{val_fold}_{fset}_{missing_percentage}.csv"
    if os.path.exists( os.path.join( save_path, filename ) ):
        mask = save_load.load_table( os.path.join( save_path, filename ), index_col=0, header=0 ).astype(bool)

        data = set_to_nullable_type(data)
        masked_data = data.values

        masked_data[mask.values] = np.nan
        masked_data = pd.DataFrame(masked_data, columns=data.columns, index=data.index)
        return masked_data, np.round((masked_data.isna().sum().sum() / (masked_data.shape[0] * masked_data.shape[1])) * 100)

    st0 = np.random.get_state()
    np.random.seed(seed)

    options = dict( MCAR_sample = MCAR_sample, MCAR_feature = MCAR_feature, MCAR_global = MCAR_global, no_generation=no_generation )

    params = { "missing_fraction": missing_fraction, **kwargs }

    IDs = data.index.to_list()

    masked_data = data.reset_index(drop=True)
    if missing_fraction != 0:

        already_complete_missing_features = masked_data.isna().all(axis=0).any() or masked_data.shape[1] == 1
        already_completely_missing_samples = masked_data.isna().all(axis=1).any() or masked_data.shape[1] == 1

        features_with_one_value = (masked_data.notna().sum(axis=0) == 1).astype(int).values.nonzero()[0]
        respective_sample_represented = (masked_data.iloc[:, features_with_one_value].notna().sum(axis=1) > 0).astype(int).values.nonzero()[0]
        samples_with_one_value = (masked_data.notna().sum(axis=1) == 1).astype(int).values.nonzero()[0]
        respective_features_represented = (masked_data.iloc[samples_with_one_value].notna().sum(axis=0) > 0).astype(int).values.nonzero()[0]

        min_values_1 = min(np.floor((masked_data.shape[0] - len(respective_sample_represented)) * (masked_data.shape[1] - len(features_with_one_value)) * (1 - missing_fraction)), masked_data.notna().sum().sum())
        min_values_2 = min(np.floor((masked_data.shape[0] - len(samples_with_one_value)) * (masked_data.shape[1] - len(respective_features_represented)) * (1 - missing_fraction)), masked_data.notna().sum().sum())
        enough_not_missing_values = min(min_values_1, min_values_2) >= max(masked_data.shape[0] - len(respective_sample_represented) - len(samples_with_one_value), masked_data.shape[1] - len(features_with_one_value) - len(respective_features_represented)) + len(features_with_one_value) + len(samples_with_one_value)

        masked_data, final_missing_percentage = options[ method ]( masked_data, **params )
        if not already_complete_missing_features and not already_completely_missing_samples and enough_not_missing_values:
            assert not masked_data.isna().all(axis=0).any() and not masked_data.isna().all(axis=1).any(), "There are completely missing rows or columns"
    else:
        final_missing_percentage = np.round((masked_data.isna().sum().sum() / (masked_data.shape[0] * masked_data.shape[1])) * 100)
    masked_data = masked_data.assign(ID=IDs).set_index("ID")

    save_load.save_table(masked_data.isna().astype(int), filename, save_path, index=True)

    np.random.set_state(st0)

    return masked_data, final_missing_percentage


def generate_sample_missing_combinations(sample) -> pd.DataFrame:

    indices = np.where( np.invert(np.isnan(sample.to_numpy().astype(float))) )[0]

    masks = np.array([])

    # for r in np.arange(len(indices)//2, 0, -1 ):
    # for r in np.arange(min(len(indices)-1, 3), 0, -1 ):
    for r in np.arange(len(indices), 0, -1 ):

        combinations = np.array( list(itertools.combinations(indices, r)) )
        mask = np.ones( ( len(combinations), len(sample) ), dtype=bool)
        np.put_along_axis(mask, combinations, False, axis=1)
        masks = np.vstack((masks, mask)) if masks.size else mask

    if len(masks.shape) < 2:
        masks = np.expand_dims(masks, axis=0)

    sample_list = np.tile( sample.to_numpy(), (masks.shape[0], 1) )
    # sample_list = np.where(~masks, sample_list, np.nan)
    sample_list = np.where(np.invert(masks.astype(bool)), sample_list.astype(float), np.nan)

    new_index = [f"{sample.name}_{i}" for i in range(masks.shape[0])]
    new_data = pd.DataFrame( sample_list, index=new_index, columns=sample.index.tolist() ).rename_axis("ID")  # pd.concat( [new_data, pd.DataFrame( sample_list, index=new_index, columns=sample.columns.tolist() )] )  # .rename_axis("ID", axis=0)

    nan_weights = pd.Series( 1 - masks.sum(axis=1)/sample_list.shape[1], index=new_index)  # pd.concat( [nan_weights, pd.Series( 1 - masks.sum(axis=1)/sample_list.shape[1], index=new_index) ], axis=0 )  # .rename_axis("ID", axis=0)
    new_data = new_data.assign(weight=nan_weights)

    return new_data


def random_mask_sample(sample: pd.Series) -> pd.Series:
    not_missing_idx = np.where(~pd.isna(sample))[0]
    to_mask = np.random.choice([False, True], p=[0.01, 0.99])
    if to_mask and len(not_missing_idx) > 1:
        idx_to_mask = np.random.choice(not_missing_idx, size=np.random.choice(np.arange(1, len(not_missing_idx))), replace=False)
        sample[idx_to_mask] = np.nan
    return sample


def generate_k_missing_combinations( data: pd.DataFrame, labels: pd.DataFrame, k: int):
    data_w_combination = data.copy().set_index( data.index.to_series().apply(lambda x: f"{x}_0" ))
    labels_w_combination = labels.copy().set_index( labels.index.to_series().apply(lambda x: f"{x}_0" ))
    combinations_weight = pd.Series(data_w_combination.notna().sum(axis=1).div(data.shape[1]), name="weight")
    for combination_number in range(1, k):
        k_combination = data.apply(random_mask_sample, axis=1)
        k_combination = k_combination.set_index(k_combination.index.to_series().apply(lambda x: f"{x}_{combination_number}" ))
        data_w_combination = pd.concat([data_w_combination, k_combination], axis=0)
        k_combination_labels = labels.set_index(labels.index.to_series().apply(lambda x: f"{x}_{combination_number}" ))
        labels_w_combination = pd.concat([labels_w_combination, k_combination_labels], axis=0)
        k_combination_weight = k_combination.notna().sum(axis=1).div(k_combination.shape[1]).rename("weight")
        combinations_weight = pd.concat([combinations_weight, k_combination_weight])

    return data_w_combination.sort_index().reset_index(), labels_w_combination.sort_index().reset_index(), combinations_weight.sort_index().reset_index()


def generate_missing_combinations( data: pd.DataFrame, labels: pd.DataFrame ):
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=False)
    
    new_data = data.parallel_apply(generate_sample_missing_combinations, axis=1)

    new_data = pd.concat(new_data.values, axis=0)

    nan_weights = new_data.loc[:, "weight"]
    new_data = new_data.drop("weight", axis=1)

    new_labels = labels.loc[ new_data.index.to_series().str.replace(r"\_.+", "", regex=True) ].rename_axis("old_ID", axis=0).reset_index()
    new_labels_IDs = new_data.index.to_frame().reset_index(drop=True)
    new_labels = pd.concat( [ new_labels, new_labels_IDs ], axis=1 )
    new_labels = new_labels.drop("old_ID", axis=1).set_index("ID")

    new_data, new_labels, nan_weights = new_data.reset_index(), new_labels.reset_index(), nan_weights.reset_index()
    log.info("Missing data combination generated")
    return new_data, new_labels, nan_weights


if __name__ == "__main__":

    np.random.seed( 42 )

    df = pd.DataFrame( np.random.rand(100, 6), columns=["ID", "label", "A", "B", "C", "D"])

    a = generate_missing(df, "MCAR_global", 0.75)[0]

    b = a.drop(["ID", "label"], axis=1).isna()

    print( [ b[col].value_counts().div(b.shape[0]) for col in b ])
    c = b.T
    print( [c[col].value_counts().div(c.shape[0]) for col in c])

    print( np.sum(b.values, axis=None)/(b.shape[0]*b.shape[1]))
