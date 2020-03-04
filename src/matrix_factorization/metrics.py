import numpy as np
import sklearn
import tqdm.autonotebook as tqdm


def evaluate(model, data_obj, output_dict, idc_output_array, X_reals_ground_truth=None):

    if not(X_reals_ground_truth is None):
        output_dict['recMSE'][idc_output_array] = np.mean(
            ((X_reals_ground_truth[data_obj.train_idc] -
              model.U@(model.V.T))[data_obj.X_train == 0])**2
        )

    output_dict['predSSE'][idc_output_array] = np.mean(
        ((data_obj.X_train - model.U@model.V.T)[data_obj.X_train != 0])**2
    )

    posterior_probability = model.posterior(
        data_obj.X_pred_regressor,
        data_obj.time_of_prediction,
    )

    N_STEPS_BIAS = (output_dict['sensitivity_with_bias']
                    [idc_output_array]).shape[0]
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
        output_dict['sensitivity_with_bias'][idc_output_array +
                                             (i_bias,)] = cm[1, 1] / np.sum(cm[1, :])
        output_dict['specificity_with_bias'][idc_output_array +
                                             (i_bias,)] = cm[0, 0] / np.sum(cm[0, :])

    # Classification full
    predicted_integers = model.predict(
        data_obj.X_pred_regressor,
        data_obj.time_of_prediction,
        p_z_precomputed=posterior_probability
    )

    output_dict['cms'][idc_output_array] = sklearn.metrics.confusion_matrix(
        data_obj.y_pred, predicted_integers)


def evaluate_all_folds(model, data_obj, output_dict, idc_output_array, X_reals_ground_truth=None, verbose=False):

    if verbose:
        pbar = tqdm.tqdm(total=data_obj.n_splits)
    for i_fold in range(data_obj.n_splits):

        data_obj.select_fold(i_fold)
        model.Y = data_obj.X_train
        model.train()

        evaluate(model, data_obj, output_dict, idc_output_array +
                 (i_fold,), X_reals_ground_truth=X_reals_ground_truth)

        if verbose:
            pbar.update()
