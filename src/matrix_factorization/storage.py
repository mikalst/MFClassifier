import numpy as np
from multiprocessing import Array
import h5py


class Result(dict):
    def __init__(self, N_STEPS_L1, N_STEPS_L3, N_FOLDS, N_Z, N_STEPS_BIAS, compute_recMSE=False):

        self['cms'] = np.empty(shape=(N_STEPS_L1*N_STEPS_L3*N_FOLDS*N_Z**2))
        self['sensitivity_with_bias'] = np.empty(shape=(N_STEPS_L1*N_STEPS_L3*N_FOLDS*N_STEPS_BIAS))
        self['specificity_with_bias'] = np.empty(shape=(N_STEPS_L1*N_STEPS_L3*N_FOLDS*N_STEPS_BIAS))
        self['predSSE'] = np.empty(shape=(N_STEPS_L1*N_STEPS_L3*N_FOLDS))
        
        if compute_recMSE:
            self['recMSE'] = np.empty(shape=(N_STEPS_L1*N_STEPS_L3*N_FOLDS))

        self.attrs = {
            'N_STEPS_L1': N_STEPS_L1,
            'N_STEPS_L3': N_STEPS_L3,
            'N_FOLDS': N_FOLDS,
            'N_Z': N_Z,
            'N_STEPS_BIAS': N_STEPS_BIAS,
            'compute_recMSE': compute_recMSE
        }

    def save(self, path, identifier):
        with h5py.File(
        path+identifier, "w"
        ) as outfile:
        
            shape = (
                self.attrs['N_STEPS_L1'],
                self.attrs['N_STEPS_L3'],
                self.attrs['N_FOLDS']
            )

            outfile.create_dataset(
                "cms",
                data=np.array(self["cms"]).reshape(
                    shape+(self.attrs['N_Z'], self.attrs['N_Z'])
                )
            )

            outfile.create_dataset(
                "sensitivity_with_bias",
                data=np.array(self["sensitivity_with_bias"]).reshape(
                    shape+(self.attrs['N_STEPS_BIAS'], )
                )
            )

            outfile.create_dataset(
                "specificity_with_bias",
                data=np.array(self["specificity_with_bias"]).reshape(
                    shape+(self.attrs['N_STEPS_BIAS'], )
                )
            )

            outfile.create_dataset(
                "predSSE",
                data=np.array(self["predSSE"]).reshape(shape)
            )

            if self.attrs['compute_recMSE']:
                outfile.create_dataset(
                    "recMSE",
                    data=np.array(self["recMSE"]).reshape(shape)
                )
            
            for key in self.attrs.keys():
                outfile.attrs[key] = self.attrs[key]


class SharedMemoryResult(Result):
    def __init__(self, N_STEPS_L1, N_STEPS_L3, N_FOLDS, N_Z, N_STEPS_BIAS, compute_recMSE=False):
        super(SharedMemoryResult, self).__init__(N_STEPS_L1, N_STEPS_L3, N_FOLDS, N_Z, N_STEPS_BIAS, compute_recMSE)

        for key in self.keys():
            self[key] = Array('d', (self[key]))