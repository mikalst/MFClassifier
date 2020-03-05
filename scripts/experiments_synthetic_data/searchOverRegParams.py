import sys
import time
import tqdm.autonotebook as tqdm
import multiprocessing
import numpy as np

path_to_project_root = '../../'
sys.path.append(path_to_project_root)

import src.simulation.simulation
import src.utils.special_matrices
from src.matrix_factorization.models import MatrixFactorization
from src.matrix_factorization.data import TemporalDataKFold
from src.matrix_factorization.metrics import evaluate_all_folds
from src.matrix_factorization.storage import generate_empty_results, store_results


def main(
    N_FOLDS=2,
    N_STEPS_L1=25,
    N_STEPS_L3=25,
    LOW_L1=1,
    HIGH_L1=1e2,
    LOW_L3=1e3,
    HIGH_L3=1e4,
    N_STEPS_BIAS=101,
    K_UPPER_RANK_EST=5, 
    THETA_EST=2.5
):

    X_reals = np.load(path_to_project_root +'data/synthetic/X_train_reals.npy')

    N, T = X_reals.shape

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
    
    data_obj = TemporalDataKFold(X_masked, 'last_observed', n_splits=N_FOLDS)

    results = generate_empty_results(
        n_lambda1=N_STEPS_L1,
        n_lambda3=N_STEPS_L3,
        n_folds=N_FOLDS,
        compute_recMSE=True
    )

    pbar = tqdm.tqdm(total=N_FOLDS*N_STEPS_L1*N_STEPS_L3)
    for i_l1, l1 in enumerate(np.linspace(LOW_L1, HIGH_L1, N_STEPS_L1)):
        for i_l3, l3 in enumerate(np.linspace(LOW_L3, HIGH_L3, N_STEPS_L3)):
            
            parameters_algorithm = {
                'lambda0' : 1.0,
                'lambda1' : l1,
                'lambda2' : 0.25,
                'lambda3' : l3,
                'Y' : data_obj.X_train,
                'R' : src.utils.special_matrices.finite_difference_matrix(data_obj.X_train.shape),
                'J' : np.ones((T, K_UPPER_RANK_EST)),
                'C' : np.identity(T),
                'K' : K_UPPER_RANK_EST,
                'domain_z': np.arange(1, 5),
                'theta_estimate': 2.5,
                'total_iterations' : 2000,
                'convergence_tol': 1e-4
            }

            model = MatrixFactorization(parameters_algorithm)
        
            evaluate_all_folds(
                model=model,
                data_obj=data_obj,
                output_dict=results,
                idc_output_array=(i_l1, i_l3),
                X_reals_ground_truth=X_reals[data_obj.train_idc]
            )

            pbar.update()

    store_results(
        result_dict=results,
        path_to_storage=path_to_project_root+"results/",
        identifier="myrun",
        attrs=None
    )


if __name__=='__main__':
    main(
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
