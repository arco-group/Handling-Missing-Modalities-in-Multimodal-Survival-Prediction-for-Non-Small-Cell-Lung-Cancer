import numpy as np
import pandas as pd

__all__ = ['compute_weighted_probability']

from sklearn.utils.multiclass import class_distribution


def compute_weighted_probability(missing_augmentation_probability, cv_info,  label_type, missing_value: int = -100):
    if label_type != "multilabel":
        class_samples = cv_info.value_counts().sort_index()
        N_samples = class_samples.sum()
        class_distribution = class_samples.div(N_samples)
        norm_coeff = (missing_augmentation_probability * N_samples) / class_samples.mul(class_distribution).sum()

        class_weighted_probabilities = (norm_coeff * class_distribution).values
        cv_info = pd.get_dummies(cv_info.astype(str)).sort_index(axis=1).astype(bool).values
    else:
        class_weighted_probabilities = []
        for class_name in cv_info:
            single_class_samples = cv_info[class_name].replace(missing_value, np.nan).value_counts().sort_index()
            N_samples = single_class_samples.sum()
            single_class_distribution = single_class_samples.div(N_samples)
            norm_coeff = (missing_augmentation_probability * N_samples) / single_class_samples.mul(single_class_distribution).sum()
            single_class_weighted_probabilities = (norm_coeff * single_class_distribution).values
            class_weighted_probabilities.append(single_class_weighted_probabilities[-1])
        class_weighted_probabilities = np.array(class_weighted_probabilities)
        cv_info = cv_info.replace(missing_value, 0).astype(bool).values
    return cv_info, class_weighted_probabilities


if __name__ == "__main__":
    pass
