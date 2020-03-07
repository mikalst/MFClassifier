import sys
import time
import tqdm.autonotebook as tqdm
import multiprocessing
import numpy as np
import copy

path_to_project_root = '../'
sys.path.append(path_to_project_root)

import src.simulation.simulation
import src.utils.special_matrices
from src.matrix_factorization.models import MatrixFactorization
from src.matrix_factorization.data import TemporalDataKFold
from src.matrix_factorization.data import TemporalDataPrediction
from src.matrix_factorization.metrics import evaluate_all_folds
from src.matrix_factorization.storage import Result, SharedMemoryResult
from src.matrix_factorization.storage import store_results


def gridsearch(
    data_obj,
    model_generator,
    idc_parameter_select,
    results,
    X_reals_ground_truth=None,
):

    for idx in idc_parameter_select:
    
        model = model_generator(idx, data_obj)
        evaluate_all_folds(model, data_obj, results, idx, X_reals_ground_truth)

    return results


def gridsearch_parallelize(
    data_obj,
    model_generator,
    idc_parameter_select,
    results,
    X_reals_ground_truth=None,
    N_CPU=4
):

    idc_per_cpu = np.array_split(idc_parameter_select, N_CPU)

    workers = []
    for i_cpu in range(N_CPU):
        data_obj_copy = copy.deepcopy(data_obj)
        workers.append(multiprocessing.Process(target=gridsearch, args=
                (
                    data_obj_copy,
                    model_generator,
                    idc_per_cpu[i_cpu],
                    results,
                    X_reals_ground_truth
                )
            )
        )

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def gridsearch_synthetic_data(
    N_FOLDS=None,
    N_STEPS_L1=25,
    N_STEPS_L3=25,
    LOW_L1=1,
    HIGH_L1=1e2,
    LOW_L3=1e3,
    HIGH_L3=1e4,
    N_STEPS_BIAS=101,
    K_UPPER_RANK_EST=5, 
    THETA_EST=2.5,
    PARALLELIZE=True
):
    # Prepare data
    X_reals = np.load(path_to_project_root +'data/synthetic/X_train_reals.npy')

    parameters_simulate_integer = {
        'output_domain': np.arange(1, 5),
        'kernel_parameter': THETA_EST,
    }

    tp = np.array([0.05, 0.15, 0.40, 0.60, 0.20])

    parameters_simulate_mask = {
        'mask_transition_expectations': tp,
        'mask_transition_variances': 1e9*np.ones(5),
        'memory_length': 10,
        'mask_level': 0.6
    }

    X_integers = src.simulation.simulation.simulate_integer_from_float(
        X_reals,
        integer_parameters=parameters_simulate_integer,
        seed=43
    )

    mask = src.simulation.simulation.simulate_mask(
        X_integers,
        mask_parameters=parameters_simulate_mask
    )

    X_masked = X_integers*mask
    
    # Create data object
    data_obj = TemporalDataKFold(X_masked, 'last_observed', n_splits=N_FOLDS)

    # Allocated empty results object
    results = SharedMemoryResult(
        N_STEPS_L1=N_STEPS_L1,
        N_STEPS_L3=N_STEPS_L3,
        N_FOLDS=N_FOLDS,
        N_STEPS_BIAS=N_STEPS_BIAS,
        N_Z=4,
        compute_recMSE=True
    )

    # Create indices that select a particular model
    idc_parameter_select = np.arange(0, N_STEPS_L1*N_STEPS_L3)

    l1_values = np.linspace(LOW_L1, HIGH_L1, N_STEPS_L1)
    l3_values = np.linspace(LOW_L1, HIGH_L1, N_STEPS_L1)

    # Prepare a model generator that yields a model for every index
    def model_generator(idx, data_obj):

        parameters_algorithm = {
            'lambda0' : 1.0,
            'lambda1' : l1_values[idx // N_STEPS_L3],
            'lambda2' : 0.25,
            'lambda3' : l3_values[idx % N_STEPS_L3],
            'Y' : data_obj.X_train,
            'R' : src.utils.special_matrices.finite_difference_matrix(data_obj.X_train.shape),
            'J' : np.ones((data_obj.X_train.shape[1], 5)),
            'C' : np.identity(data_obj.X_train.shape[1]),
            'K' : 5,
            'domain_z': np.arange(1, 5),
            'theta_estimate': 2.5,
            'total_iterations' : 2000,
            'convergence_tol': 1e-4
        }

        return MatrixFactorization(parameters_algorithm)

    # Search in parallel over all possible models
    gridsearch_parallelize(
        data_obj,
        model_generator,
        idc_parameter_select,
        results,
        X_reals_ground_truth=X_reals,
        N_CPU=4
    )

    store_results(
        result_obj=results,
        path_to_storage=path_to_project_root+'results/experiments_synthetic_data/',
        identifier=r"run{:d}.hdf5".format(int(time.time())),
    )


def gridsearch_jerome_data(
    N_FOLDS=None,
    N_STEPS_L1=25,
    N_STEPS_L3=25,
    LOW_L1=1,
    HIGH_L1=1e2,
    LOW_L3=1e3,
    HIGH_L3=1e4,
    N_STEPS_BIAS=101,
    K_UPPER_RANK_EST=5, 
    THETA_EST=2.5,
    PARALLELIZE=True
):
    # Not yet implemented
    pass


if __name__=='__main__':
    
    gridsearch_synthetic_data(
        N_FOLDS=int(sys.argv[1]),
        N_STEPS_L1=int(sys.argv[2]),
        N_STEPS_L3=int(sys.argv[3]),
        LOW_L1=float(sys.argv[4]),
        HIGH_L1=float(sys.argv[5]),
        LOW_L3=float(sys.argv[6]),
        HIGH_L3=float(sys.argv[7]),
        N_STEPS_BIAS=int(sys.argv[8]),
        K_UPPER_RANK_EST=int(sys.argv[9]),
        THETA_EST=float(sys.argv[10])
    )
