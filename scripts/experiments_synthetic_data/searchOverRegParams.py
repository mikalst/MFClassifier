import sys
import time
import argparse
import numpy as np
import scipy as sp
import sklearn.metrics
import h5py
from tqdm import tqdm

path_to_project_root = '../../'
sys.path.append(path_to_project_root)

import src.utils.special_matrices
import src.simulation.simulation
from src.matrix_factorization.models import MatrixFactorization
from src.matrix_factorization.data import TemporalDataKFold

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

    recMSE = np.zeros((N_STEPS_L1, N_STEPS_L3, N_FOLDS))
    predSSE = np.empty((N_STEPS_L1, N_STEPS_L3, N_FOLDS))

    bias = np.linspace(0, 1, N_STEPS_BIAS)
    sensitivity_with_bias = np.empty((N_STEPS_L1, N_STEPS_L3, N_FOLDS, N_STEPS_BIAS))
    specificity_with_bias = np.empty((N_STEPS_L1, N_STEPS_L3, N_FOLDS, N_STEPS_BIAS))

    cms = np.empty((N_STEPS_L1, N_STEPS_L3, N_FOLDS, 4, 4))

    pbar = tqdm(total=N_FOLDS*N_STEPS_L1*N_STEPS_L3)
    for i_l1, l1 in enumerate(np.linspace(LOW_L1, HIGH_L1, N_STEPS_L1)):
        for i_l3, l3 in enumerate(np.linspace(LOW_L3, HIGH_L3, N_STEPS_L3)):
            for i_fold in range(N_FOLDS):

                X_train, X_pred_regressor, y_pred, t_of_prediction = data_obj.get_fold(i_fold)
                idc_train, idc_pred = data_obj.get_fold_idc(i_fold)

                parameters_algorithm = {
                    'lambda0' : 1.0,
                    'lambda1' : l1,
                    'lambda2' : 0.25,
                    'lambda3' : l3,
                    'Y' : X_train,
                    'R' : src.utils.special_matrices.finite_difference_matrix(X_train.shape),
                    'J' : np.ones((X_train.shape[1], K_UPPER_RANK_EST)),
                    'C' : np.identity(X_train.shape[1]),
                    'K' : K_UPPER_RANK_EST,
                    'total_iterations' : 2000,
                    'convergence_tol': 1e-4
                }

                model = MatrixFactorization(parameters_algorithm)
                model.train()

                # RECONSTRUCTION
                recMSE[i_l1, i_l3] = np.mean(
                    ((X_reals[idc_train] - model.U@(model.V.T))[X_train == 0])**2
                )
                predSSE[i_l1, i_l3] = np.mean(
                    ((X_train - model.U@model.V.T)[X_train != 0])**2
                )

                posterior_probability = model.posterior(
                    X_pred_regressor,
                    t_of_prediction,
                    domain_z=np.arange(1, 5),
                    theta_estimate=THETA_EST
                )

                # Classificiation in a boolean (Sick / Healthy) setting
                for i_bias, b in enumerate(bias):                
                    
                    predicted_e = model.predict_rulebased(
                        X_pred_regressor,
                        t_of_prediction,
                        domain_z=np.arange(1, 5),
                        theta_estimate=THETA_EST,
                        p_z_precomputed=posterior_probability,
                        rule_z_to_e= lambda x: 0 if x==1 else 1,
                        domain_e = np.arange(0, 2),
                        bias_e = np.array([1-b, b])
                    )

                    cm = sklearn.metrics.confusion_matrix(y_pred > 1, predicted_e)
                    sensitivity_with_bias[i_l1, i_l3, i_fold, i_bias] = cm[1, 1] / np.sum(cm[1, :])
                    specificity_with_bias[i_l1, i_l3, i_fold, i_bias] = cm[0, 0] / np.sum(cm[0, :])

                # Classification full
                predicted_integers = model.predict(
                    X_pred_regressor,
                    t_of_prediction,
                    domain_z=np.arange(1, 5),
                    theta_estimate=THETA_EST,
                    p_z_precomputed=posterior_probability
                )
                
                cms[i_l1, i_l3, i_fold] = sklearn.metrics.confusion_matrix(y_pred, predicted_integers)

                pbar.update()

    with h5py.File(
        path_to_project_root+"/results/experiments_synthetic_data/"+"run"+str(int(time.time() // 1))+".hdf5", "w"
    ) as outfile:
        outfile.create_dataset("sensitivity_binary", data=sensitivity_with_bias)
        outfile.create_dataset("specificity_binary", data=specificity_with_bias)
        outfile.create_dataset("confusion_matrices", data=cms)

        outfile.attrs['K_UPPER_RANK_ESTIMATE']=K_UPPER_RANK_EST
        outfile.attrs['THETA_EST']=THETA_EST
        outfile.attrs['N_FOLDS'] = 5
        outfile.attrs['N_STEPS_L1'] = N_STEPS_L1
        outfile.attrs['N_STEPS_L3'] = N_STEPS_L3
        outfile.attrs['LOW_L1'] = LOW_L1
        outfile.attrs['HIGH_L1'] = HIGH_L1
        outfile.attrs['LOW_L3'] = LOW_L3
        outfile.attrs['HIGH_L3'] = HIGH_L3
        outfile.attrs['N_STEPS_BIAS'] = N_STEPS_BIAS

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
