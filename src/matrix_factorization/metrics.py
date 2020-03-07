import numpy as np
import sklearn
import copy
import tqdm.autonotebook as tqdm


def evaluate(model, data_obj, output_result_obj, idx_output_result, X_reals_ground_truth=None):

    N_STEPS_BIAS = output_result_obj.attrs['N_STEPS_BIAS']
    N_Z = output_result_obj.attrs['N_Z']

    if not(X_reals_ground_truth is None):
        output_result_obj['recMSE'][idx_output_result] = np.mean(
            ((X_reals_ground_truth[data_obj.train_idc] -
              model.U@(model.V.T))[data_obj.X_train == 0])**2
        )

    output_result_obj['predSSE'][idx_output_result] = np.mean(
        ((data_obj.X_train - model.U@model.V.T)[data_obj.X_train != 0])**2
    )

    posterior_probability = model.posterior(
        data_obj.X_pred_regressor,
        data_obj.time_of_prediction,
    )

    for i_bias, b in enumerate(np.linspace(0, 1, N_STEPS_BIAS)):

        predicted_e = model.predict_rulebased(
            data_obj.X_pred_regressor,
            data_obj.time_of_prediction,
            p_z_precomputed=posterior_probability,
            rule_z_to_e=lambda x: 0 if x == 1 else 1,
            domain_e=np.arange(0, 2),
            bias_e=np.array([1-b, b])
        )

        cm = sklearn.metrics.confusion_matrix(data_obj.y_pred > 1, predicted_e)
        output_result_obj['sensitivity_with_bias'][N_STEPS_BIAS*idx_output_result + i_bias] = cm[1, 1] / np.sum(cm[1, :])
        output_result_obj['specificity_with_bias'][N_STEPS_BIAS*idx_output_result + i_bias] = cm[0, 0] / np.sum(cm[0, :])

    # Classification full
    predicted_integers = model.predict(
        data_obj.X_pred_regressor,
        data_obj.time_of_prediction,
        p_z_precomputed=posterior_probability
    )

    output_result_obj['cms'][(N_Z**2)*idx_output_result:(N_Z**2)*idx_output_result+(N_Z**2)] = (sklearn.metrics.confusion_matrix(
        data_obj.y_pred, predicted_integers, labels=range(1, N_Z+1))).flatten()


def evaluate_all_folds(model, data_obj, output_result_obj, idx_output_result, X_reals_ground_truth=None, verbose=False):

    N_FOLDS = data_obj.n_splits

    if verbose:
        pbar = tqdm.tqdm(total=N_FOLDS)

    for i_fold in range(N_FOLDS):

        data_obj.select_fold(i_fold)
        model.Y = data_obj.X_train
        model.reset()
        model.train()

        evaluate(model, data_obj, output_result_obj, N_FOLDS*idx_output_result+i_fold, X_reals_ground_truth=X_reals_ground_truth)

        if verbose:
            pbar.update()
