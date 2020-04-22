import os
import sys
import time
import multiprocessing
import numpy as np

path_to_project_root = sys.path[0]+'/../'
sys.path.append(path_to_project_root)

from dgdpredict.utils.special_matrices import finite_difference_matrix
from dgdpredict.model_selection import search, search_parallelize
from dgdpredict.model_selection import Result, SharedMemoryResult
from dgdpredict.data import TemporalDatasetTrain, TemporalDatasetPredict, TemporalDatasetKFold
from dgdpredict.model import DGDClassifier

home = os.path.expanduser("~")
sys.path.append(sys.path[0]+'/../../data_generators')

import src.mask
import src.dgd_data_generator


def gridsearch_jerome_data(
    N_FOLDS=None,
    N_STEPS_L1=25,
    N_STEPS_L3=25,
    LOW_L1=1,
    HIGH_L1=1e2,
    LOW_L3=1e3,
    HIGH_L3=1e4,
    K_UPPER_RANK_EST=5,
    THETA_EST=2.5,
    PARALLELIZE=False
):

    # Prepare Jerome data
    X_jerome = np.load(path_to_project_root +
                       'data/jerome_processed/training_data.npy')

    # Create data object
    data_obj = TemporalDatasetKFold(
        X_jerome, prediction_rule='last_observed', n_splits=N_FOLDS)

    # Prepare a model generator that yields a model for every index
    def model_generator(idx):

        l1_vals = np.linspace(LOW_L1, HIGH_L1, N_STEPS_L1)
        l3_vals = np.linspace(LOW_L3, HIGH_L3, N_STEPS_L3)

        return DGDClassifier(
            lambda0=1.0,
            lambda1=l1_vals[idx // N_STEPS_L3],
            lambda2=0.25,
            lambda3=l3_vals[idx % N_STEPS_L3],
            K=5,
            domain_z=np.arange(1, 5),
            z_to_binary_mapping=lambda x: np.array(x) > 2,
            T=321,
            max_iter=2000,
            tol=1e-4
        )

    # Create indices that select a particular model
    idc_parameter_select = np.arange(0, N_STEPS_L1*N_STEPS_L3)

    if PARALLELIZE:
        # Allocated empty results object
        results = SharedMemoryResult(
            N_SEARCH_POINTS=N_STEPS_L1*N_STEPS_L3,
            N_FOLDS=N_FOLDS,
            N_Z=4,
            compute_recMSE=False
        )

        # Search in parallel over all possible models
        search_parallelize(
            data_obj,
            model_generator,
            idc_parameter_select,
            results,
            N_CPU=multiprocessing.cpu_count()
        )
    else:
        # Allocated empty results object
        results = Result(
            N_SEARCH_POINTS=N_STEPS_L1*N_STEPS_L3,
            N_FOLDS=N_FOLDS,
            N_Z=4,
            compute_recMSE=False
        )
        search(
            data_obj,
            model_generator,
            idc_parameter_select,
            results
        )

    results.save(
        path=path_to_project_root+'results/experiments_jerome_data/',
        identifier=r"run{:d}.hdf5".format(int(time.time())),
    )



if __name__ == '__main__':
    t0 = time.time()

    gridsearch_jerome_data(
        N_FOLDS=int(sys.argv[1]),
        N_STEPS_L1=int(sys.argv[2]),
        N_STEPS_L3=int(sys.argv[3]),
        LOW_L1=float(sys.argv[4]),
        HIGH_L1=float(sys.argv[5]),
        LOW_L3=float(sys.argv[6]),
        HIGH_L3=float(sys.argv[7]),
        K_UPPER_RANK_EST=int(sys.argv[8]),
        THETA_EST=float(sys.argv[9]),
        PARALLELIZE=eval(sys.argv[10])
    )

    print("Time spent: ", time.time() - t0)
