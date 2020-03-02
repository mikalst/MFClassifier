from tqdm import tqdm
import numpy as np
import scipy as sp
import time
import sklearn.metrics
import h5py
import sys
path_to_project_root = '../../'
sys.path.append(path_to_project_root)

import src.utils.special_matrices
from src.matrix_factorization.models import MatrixFactorization
from src.matrix_factorization.data import TemporalDataKFold

def main(
    N_FOLDS=5,
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

    data = np.load(path_to_project_root+"data/jerome_processed/training_data.npy")
    data_obj = TemporalDataKFold(data, 'last_observed', n_splits=N_FOLDS)

    bias = np.linspace(0, 1, N_STEPS_BIAS)
    sensitivity_with_bias = np.empty((N_STEPS_L1, N_STEPS_L3, N_FOLDS, N_STEPS_BIAS))
    specificity_with_bias = np.empty((N_STEPS_L1, N_STEPS_L3, N_FOLDS, N_STEPS_BIAS))

    cm_maxp = np.empty((N_STEPS_L1, N_STEPS_L3, N_FOLDS, 4, 4))
    pbar = tqdm(total=N_FOLDS*N_STEPS_L1*N_STEPS_L3)
    for i_l1, l1 in enumerate(np.linspace(LOW_L1, HIGH_L1, N_STEPS_L1)):
        for i_l3, l3 in enumerate(np.linspace(LOW_L3, HIGH_L3, N_STEPS_L3)):
            for i_fold in range(N_FOLDS):

                X_train, X_pred_regressor, y_pred, t_of_prediction = data_obj.get_fold(i_fold)

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
                
                cm_maxp[i_l1, i_l3, i_fold] = sklearn.metrics.confusion_matrix(y_pred, predicted_integers)

                pbar.update()

    with h5py.File(
        path_to_project_root+"/results/experiments_jerome_data/"+"run"+str(time.time() // 1)+".hdf5", "w"
    ) as outfile:
        outfile.create_dataset("sensitivity_binary", data=sensitivity_with_bias)
        outfile.create_dataset("specificity_binary", data=specificity_with_bias)
        outfile.create_dataset("confusion_matrices", data=cm_maxp)

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