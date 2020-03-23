import numpy as np
import multiprocessing
from multiprocessing import Array
import h5py

import sklearn.metrics


N_STEPS_BIAS = 100


class Result(dict):
    def __init__(self, N_SEARCH_POINTS, N_FOLDS, N_Z, compute_recMSE=False):
        # Input parameters
        self['lambda0'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))
        self['lambda1'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))
        self['lambda2'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))
        self['lambda3'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))
        self['K'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))
        self['theta'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))

        # Scoring measures
        self['confusion_matrix'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS*N_Z**2))
        self['accuracy_bin_with_bias'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS*N_STEPS_BIAS))
        self['sensitivity_with_bias'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS*N_STEPS_BIAS))
        self['specificity_with_bias'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS*N_STEPS_BIAS))
        self['SSE'] = np.empty(shape=(N_SEARCH_POINTS*N_FOLDS))

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
                "confusion_matrix",
                data=np.array(self["confusion_matrix"]).reshape(
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
    def __init__(self, N_SEARCH_POINTS, N_FOLDS, N_Z, compute_recMSE=False):
        super(SharedMemoryResult, self).__init__(N_SEARCH_POINTS, N_FOLDS, N_Z, compute_recMSE)

        for key in self.keys():
            self[key] = Array('d', (self[key]))



def _score(model, data_obj, result_obj, idx):

    result_obj['lambda0'][idx] = model.lambda0
    result_obj['lambda1'][idx] = model.lambda1
    result_obj['lambda2'][idx] = model.lambda2
    result_obj['lambda3'][idx] = model.lambda3
    result_obj['K'][idx] = model.K
    result_obj['theta'][idx] = model.theta

    N_Z = result_obj.attrs['N_Z']
    predicted_z = model.predict(data_obj.X_pred_regressor, data_obj.time_of_prediction)
    result_obj['confusion_matrix'][idx*N_Z**2: (idx+1)*N_Z**2] = (
        sklearn.metrics.confusion_matrix(
            data_obj.y_true,
            predicted_z,
            labels=model.domain_z
        )
    ).flatten()

    predicted_proba_binary = model.predict_proba_binary(data_obj.X_pred_regressor, data_obj.time_of_prediction)
    y_true_bin = model.z_to_binary_mapping(data_obj.y_true)
    for i_bias, bias in enumerate(np.linspace(0, 1, N_STEPS_BIAS)):
        predict_bin = predicted_proba_binary >= bias
        result_obj['sensitivity_with_bias'][N_STEPS_BIAS*idx + i_bias] = np.mean(predict_bin[y_true_bin == 1] == y_true_bin[y_true_bin == 1])
        result_obj['specificity_with_bias'][N_STEPS_BIAS*idx + i_bias] = np.mean(predict_bin[y_true_bin == 0] == y_true_bin[y_true_bin == 0])
        result_obj['accuracy_bin_with_bias'][N_STEPS_BIAS*idx + i_bias] = np.mean(predict_bin == y_true_bin)

    result_obj['SSE'][idx] = np.mean(((model.U@model.V.T - model.X_train)[model.nonzero_rows, model.nonzero_cols])**2)
    
    # Optional scoring measure
    if result_obj.attrs['compute_recMSE']:
        result_obj['recMSE'][idx] = np.mean(((model.U@model.V.T - data_obj.ground_truth_train)[~model.nonzero_rows, ~model.nonzero_cols])**2)


def search(
    data_obj,
    model_generator,
    idc_parameter_select,
    result_obj
):

    N_FOLDS = result_obj.attrs['N_FOLDS']

    for idx in idc_parameter_select:
    
        model = model_generator(idx)

        for i_fold in range(N_FOLDS):
    
            data_obj.i_fold = i_fold
            model.fit(data_obj.X_train)
            _score(model, data_obj, result_obj, N_FOLDS*idx + i_fold)


def search_parallelize(
    data_obj,
    model_generator,
    idc_parameter_select,
    result_obj,
    N_CPU
):

    idc_per_cpu = np.array_split(idc_parameter_select, N_CPU)

    print(idc_per_cpu)

    workers = []
    for i_cpu in range(N_CPU):
        workers.append(multiprocessing.Process(target=search, args=
                (
                    data_obj,
                    model_generator,
                    idc_per_cpu[i_cpu],
                    result_obj
                )
            )
        )

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()