path_to_project_root = '../../'
import sys
sys.path.append(path_to_project_root)

import numpy as np
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

import src.simulation.simulate
import src.utils.plotting
import src.utils.saving
import src.optimization
import src.utils.stats
from src.models.matrix_factorization import MatrixFactorization

def main(l1_low, l1_high, l1_res, l3_low, l3_high, l3_res):

    X_train_reals = np.load(path_to_project_root+'data/synthetic/X_train_reals.npy')
    X_pred_reals = np.load(path_to_project_root+'data/synthetic/X_pred_reals.npy')[:5000]

    N1, T = X_train_reals.shape
    N2, T = X_pred_reals.shape

    parameters_simulate_integer = {
        'output_domain': np.arange(1,5),
        'kernel_parameter': 2.5,
    }

    tp = np.array([0.05, 0.15, 0.40, 0.60, 0.20])

    parameters_simulate_mask = {
        'mask_transition_expectations': tp,
        'mask_transition_variances': 1e9*np.ones(5),
        'memory_length': 10,
        'mask_level': 0.6
    }

    X_train_masked = src.simulation.simulate.simulate_synthetic(
        X_train_reals,
        integer_parameters=parameters_simulate_integer,
        mask_parameters=parameters_simulate_mask,
        seed=42
    )

    X_pred_integers = src.simulation.simulate.simulate_integer_from_float(
        X_pred_reals,
        integer_parameters=parameters_simulate_integer,
        seed=43
    )

    mask_pred = src.simulation.simulate.simulate_mask(
        X_pred_integers,
        mask_parameters=parameters_simulate_mask
    )

    X_pred_masked = X_pred_integers * mask_pred
    start_prediction = 125
    X_pred_masked[:, start_prediction:] = 0
    pctg_nonzero_pred = np.mean(X_pred_masked != 0)

    resolution = 20
    lambda1_vals = np.linspace(1, 100, resolution)
    lambda2_vals = np.linspace(0.20, 0.50, resolution)
    lambda3_vals = np.linspace(1e3, 1.5e5, resolution)

    recMSE = np.zeros((resolution, resolution, resolution))
    predSENS = np.empty((resolution, resolution, resolution))
    predSPEC = np.empty((resolution, resolution, resolution))
    predACC = np.empty((resolution, resolution, resolution))

    k = 5
    with tqdm.tqdm_notebook(total=resolution**3) as process_bar:
        for i, l1 in enumerate(lambda1_vals):
            for j, l2 in enumerate(lambda2_vals):
                for k, l3 in enumerate(lambda3_vals):

                    parameters_algorithm = {
                        'lambda0' : 1,
                        'lambda1' : l1, 
                        'lambda2' : l2,
                        'lambda3' : l3,
                        'X' : X_train_masked,
                        'R' : (np.diag(np.pad(-np.ones(T - 1), (0, 1), 'constant')) + np.diag(np.ones(T-1), 1)),
                        'J' : np.ones((T, k)),
                        'kappa' : np.identity(T),
                        'k' : k,
                        'total_iterations' : 1000,
                        'convergence_tol': 5e-4
                    }

                    model = MatrixFactorization(parameters_algorithm)

                    ## Running the algorithm
                    model.iteration = 0

                    converged = next(model)

                    while not converged:
                        converged = next(model)

                    process_bar.update()

                    # RECONSTRUCTION
                    recMSE[i, j, k] = np.mean(((X_train_reals - (model.U)@(model.V.T))[X_train_masked == 0])**2)

                    # PREDICTION
                    pred_window = 10

                    predicted_integers = 1 + np.argmax(model.predict(X_pred_masked, start_prediction + pred_window), axis=1)
                    cm_mc = confusion_matrix(X_pred_integers[:, start_prediction+pred_window], predicted_integers)

                    predSENS[i, j, k] = np.sum(cm_mc[1:, 1:]) / np.sum(cm_mc[1:, :])
                    predSPEC[i, j, k] = np.sum(cm_mc[0, 0]) / np.sum(cm_mc[0, :])
                    predACC[i, j, k] = np.sum(np.diag(cm_mc)) / np.sum(cm_mc)


    np.save(path_to_project_root+'results/recMSE_'+str(resolution), recMSE)

    lambda_vals = np.vstack(
        (
            lambda1_vals,
            lambda2_vals,
            lambda3_vals
        )
    )

    np.save(path_to_project_root+'results/lambdaValues_'+str(resolution), lambda_vals)
    np.save(path_to_project_root+'results/predSENS_'+str(resolution), predSENS)
    np.save(path_to_project_root+'results/predSPEC_'+str(resolution), predSPEC)
    np.save(path_to_project_root+'results/predACC_'+str(resolution), predACC)

