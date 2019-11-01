import json
import numpy as np
import os


def save_ridge(
    parameters_simulate_ordinal,
    parameters_simulate_mask,
    parameters_algorithm,
    X_ordinal,
    X_float,
    X_masked,
    ridge,
    out
):
    percentage_nonzero = X_masked.count_nonzero(
    ) / (X_masked.shape[0] * X_masked.shape[1])
    nonzero_rows, nonzero_cols = X_masked.nonzero()

    cm_mc = ridge.confusion_matrix()
    cm_ff = ridge.confusion_matrix_forward_fill()

    params = {
        'ord_output_domain': ''.join([str(s) for s in parameters_simulate_ordinal['output_domain']]),
        'ord_pdf': ''.join([str(s) for s in parameters_simulate_ordinal['original_data_pdf']]),
        'ord_kernel_parameter': parameters_simulate_ordinal['kernel_parameter'],
        'ord_truncate_limits': ''.join([str(s) for s in parameters_simulate_ordinal['truncate_limits']]),

        'mask_expectations': ''.join([str(s) for s in parameters_simulate_mask['mask_transition_expectations']]),
        'mask_variances': ''.join([str(s) for s in parameters_simulate_mask['mask_transition_variances']]),
        'mask_level': parameters_simulate_mask['mask_level'],
        'mask_memory_length': parameters_simulate_mask['memory_length'],
        'mask_prctg_nonzero': percentage_nonzero,

        'alg_lambda0': parameters_algorithm['lambda0'],
        'alg_lambda1': parameters_algorithm['lambda1'],
        'alg_lambda2': parameters_algorithm['lambda2'],
        'alg_lambda3': parameters_algorithm['lambda3'],
        'alg_k': parameters_algorithm['k'],
        'alg_total_iterations': parameters_algorithm['total_iterations'],

        'predict_window': parameters_algorithm['predict_window'],

        'result_reconstruct_ordinal': np.mean(np.abs((ridge.U@ridge.V.T - X_ordinal)[~nonzero_rows, ~nonzero_cols])),
        'result_reconstruct_mean': np.mean(np.abs((ridge.U@ridge.V.T - X_float)[~nonzero_rows, ~nonzero_cols])),
        'result_specificity_mc': (cm_mc[0, 0] / np.sum(cm_mc[:, 0])),
        'result_sensitivity_mc': (np.sum(cm_mc[1:, 1:]) / np.sum(cm_mc[:, 1:])),
        'result_specificity_ff': (cm_ff[0, 0] / np.sum(cm_ff[:, 0])),
        'result_sensitivity_ff': np.sum(cm_ff[1:, 1:]) / np.sum(cm_ff[:, 1:])
    }

    params_json = json.dumps(params)

    if not os.path.exists('/'.join(out.split('/')[:-1])):
        os.makedirs('/'.join(out.split('/')[:-1]))

    out_params = open(out + 'params.json', 'w')
    out_params.write(params_json)
    out_params.close()

    np.save(out + 'cm_mc.npy', cm_mc)
    np.save(out + 'cm_ff.npy', cm_ff)

    np.save(out + 'V.npy', ridge.V)
