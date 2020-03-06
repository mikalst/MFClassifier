import numpy as np
from multiprocessing import Array
import h5py

class Result(dict):
    def __init__(self, N_STEPS_L1, N_STEPS_L3, N_FOLDS, N_Z, N_STEPS_BIAS, compute_recMSE=False):

        self['cms'] = -np.ones(shape=(N_STEPS_L1*N_STEPS_L3*N_FOLDS*N_Z**2))
        self['sensitivity_with_bias'] = -np.ones(shape=(N_STEPS_L1*N_STEPS_L3*N_FOLDS*N_STEPS_BIAS))
        self['specificity_with_bias'] = -np.ones(shape=(N_STEPS_L1*N_STEPS_L3*N_FOLDS*N_STEPS_BIAS))
        self['predSSE'] = -np.ones(shape=(N_STEPS_L1*N_STEPS_L3*N_FOLDS))
        
        if compute_recMSE:
            self['recMSE'] = -np.ones(shape=(N_STEPS_L1*N_STEPS_L3*N_FOLDS))

        self.attrs = {
            'N_STEPS_L1': N_STEPS_L1,
            'N_STEPS_L3': N_STEPS_L3,
            'N_FOLDS': N_FOLDS,
            'N_Z': N_Z,
            'N_STEPS_BIAS': N_STEPS_BIAS,
            'compute_recMSE': compute_recMSE
        }

class SharedMemoryResult(Result):
    def __init__(self, N_STEPS_L1, N_STEPS_L3, N_FOLDS, N_Z, N_STEPS_BIAS, compute_recMSE=False):
        super(SharedMemoryResult, self).__init__(N_STEPS_L1, N_STEPS_L3, N_FOLDS, N_Z, N_STEPS_BIAS, compute_recMSE)

        for key in self.keys():
            self[key] = Array('d', (self[key]))


def store_results(result_obj, path_to_storage, identifier):
    with h5py.File(
        path_to_storage+identifier, "w"
    ) as outfile:
        
        shape = (
            result_obj.attrs['N_STEPS_L1'],
            result_obj.attrs['N_STEPS_L3'],
            result_obj.attrs['N_FOLDS']
        )

        outfile.create_dataset(
            "cms",
            data=np.array(result_obj["cms"]).reshape(
                shape+(result_obj.attrs['N_Z'], result_obj.attrs['N_Z'])
            )
        )

        outfile.create_dataset(
            "sensitivity_with_bias",
            data=np.array(result_obj["sensitivity_with_bias"]).reshape(
                shape+(result_obj.attrs['N_STEPS_BIAS'], )
            )
        )

        outfile.create_dataset(
            "specificity_with_bias",
            data=np.array(result_obj["specificity_with_bias"]).reshape(
                shape+(result_obj.attrs['N_STEPS_BIAS'], )
            )
        )

        outfile.create_dataset(
            "predSSE",
            data=np.array(result_obj["predSSE"]).reshape(shape)
        )

        if result_obj.attrs['compute_recMSE']:
            outfile.create_dataset(
                "recMSE",
                data=np.array(result_obj["recMSE"]).reshape(shape)
            )
        
        for key in result_obj.attrs.keys():
            outfile.attrs[key] = result_obj.attrs[key]