import numpy as np
import multiprocessing
from multiprocessing import Array
import h5py




class Result(dict):
    def __init__(self, N_SEARCH_POINTS, N_FOLDS, N_Z, N_STEPS_BIAS, compute_recMSE=False):

        # Input parameters
        self['lambda0'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))
        self['lambda1'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))
        self['lambda2'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))
        self['lambda3'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))
        self['K'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))
        self['theta'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))

        # Scoring measures
        self['cms'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS*N_Z**2))
        self['sensitivity_with_bias'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS*N_STEPS_BIAS))
        self['specificity_with_bias'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS*N_STEPS_BIAS))
        self['predSSE'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))

        # Optional scoring measure
        if compute_recMSE:
            self['recMSE'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))

        self.attrs = {
            'N_SEARCH_POINTS': N_SEARCH_POINTS,
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
                self.attrs['N_SEARCH_POINTS'],
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
    def __init__(self, N_SEARCH_POINTS, N_FOLDS, N_Z, N_STEPS_BIAS, compute_recMSE=False):
        super(SharedMemoryResult, self).__init__(N_SEARCH_POINTS, N_FOLDS, N_Z, N_STEPS_BIAS, compute_recMSE)

        for key in self.keys():
            self[key] = Array('d', (self[key]))


def search(
    data_obj,
    model_generator,
    idc_parameter_select,
    results
):

    for idx in idc_parameter_select:
    
        model = model_generator(idx)
        model.fit(data_obj.X_train)
        model.score(data_obj, results, idx)

        print("Running idx:", idx)


def search_parallelize(
    data_obj,
    model_generator,
    idc_parameter_select,
    results,
    N_CPU
):

    idc_per_cpu = np.array_split(idc_parameter_select, N_CPU)

    workers = []
    for i_cpu in range(N_CPU):
        workers.append(multiprocessing.Process(target=search, args=
                (
                    data_obj,
                    model_generator,
                    idc_per_cpu[i_cpu],
                    results
                )
            )
        )

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()