import numpy as np
import h5py


def generate_empty_results(n_lambda1, n_lambda3, n_folds, compute_recMSE=False):
    result = {}

    result['cms'] = np.empty(shape=(n_lambda1, n_lambda3, n_folds))
    result['sensitivity_with_bias'] = np.empty(shape=(n_lambda1, n_lambda3, n_folds, 101))
    result['specificity_with_bias'] = np.empty(shape=(n_lambda1, n_lambda3, n_folds, 101))
    result['predSSE'] = np.empty(shape=(n_lambda1, n_lambda3, n_folds))
    
    if compute_recMSE:
        result['recMSE'] = np.empty(shape=(n_lambda1, n_lambda3, n_folds))

    return result

def store_results(result_dict, path_to_storage, identifier=None, attrs=None):
    with h5py.File(
        path_to_storage+identifier, "w"
    ) as outfile:
        for key in result_dict.keys():
            outfile.create_dataset(key, result_dict[key])
        
        if not(attrs is None):
            for key in attrs.keys():
                outfile.attrs[key] = attrs[key]
