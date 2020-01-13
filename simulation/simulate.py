#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithms for simulating ordinal from continuous data and for simulating a
missing data mask from an ordinal dataset.
"""

import numpy as np


def simulate_mask(
    X,
    parameters
):
    """Simulation of a missing data mask.

    Parameters
    ----------
    X : The unmasked data.
    f_of_t : The time distribution of the mask
    expectation_of_priors : ...
    variances : Expectation and variance used in the priors of the corresponding ordinal value
    level : Parameter for tuning the global length of the mask. 

    Returns
    ----------
    mask : The resulting simulated mask.
    """
    mask_transition_expectations = parameters['mask_transition_expectations']
    mask_transition_variances = parameters['mask_transition_variances']
    mask_memory_length = parameters['memory_length']
    mask_level = parameters['mask_level']

    mask = np.zeros_like(X, dtype=np.bool)
    observed_values = np.zeros_like(X)
    individual_probs = np.empty(
        (X.shape[0], mask_transition_expectations.shape[0]))

    # Assign individual transition probabilities
    for i in range(mask_transition_expectations.shape[0]):
        individual_probs[:, i] = np.random.beta(mask_transition_variances[i]*mask_transition_expectations[i],
                                                mask_transition_variances[i]*(1-mask_transition_expectations[i]), size=(X.shape[0]))

    for t in range(X.shape[1] - 1):
        # Find last remembered values
        last_remembered_values = observed_values[np.arange(observed_values.shape[0]), t+1-np.argmax(
            observed_values[:, t+1:max(0, t-mask_memory_length):-1] != 0, axis=1)]

        indices = ((np.arange(np.max(X)+1))
                   [:, None] == last_remembered_values).T
        p = mask_level*individual_probs[indices]
        r = np.random.uniform(size=(X.shape[0]))
        mask[r <= p, t+1] = True
        observed_values[r <= p, t+1] = X[r <= p, t+1]

    return mask


def simulate_integer_from_float(
    matrix,
    parameters,
    return_float=False,
    seed=None
):
    """Simulation of integer data from floats.

    Parameters
    ----------
    matrix       : Scores.
    parameters   : Parameters for the simulation.
        {
            output_domain    : Subset of the integers included in the output.
            kernel_parameter : Parameter used in the pmf.
        }
    return_float : Return input.
    seed         : Replication of results.

    Returns
    ----------
    res : Simulated integer matrix 
    """
    output_domain = parameters['output_domain']
    kernel_parameter = parameters['kernel_parameter']

    if not(seed is None):
        np.random.seed(seed)

    domain_max = np.max(output_domain)
    domain_min = np.min(output_domain)

    matrix = domain_min + (domain_max - domain_min)*(matrix -
                                                       np.min(matrix))/(np.max(matrix) - np.min(matrix))

    def distribution(x, dom): return np.exp(-kernel_parameter*(x - dom)**2)
    def is_neighbours(x, y): return np.abs(x - y) <= 1

    domain_repeated = np.repeat(output_domain, matrix.shape[0]).reshape(
        (matrix.shape[0], output_domain.shape[0]), order='F')

    res = np.empty_like(matrix)

    # Initialization
    column_repeated = np.repeat(matrix[:, 0], 4).reshape(
        (matrix.shape[0], 4), order='C')
    d = distribution(column_repeated, domain_repeated)
    cdf = np.cumsum(d / np.reshape(np.sum(d, axis=1),
                                   (matrix.shape[0], 1)), axis=1)

    u = np.random.uniform(size=(matrix.shape[0], 1))
    indices = np.argmax(u <= cdf, axis=1)
    res[:, 0] = output_domain[indices]

    # Timestepping
    for j in range(1, matrix.shape[1]):
        column_repeated = np.repeat(matrix[:, j], 4).reshape(
            (matrix.shape[0], 4), order='C')
        last_int_column_repeated = np.repeat(
            res[:, j-1], 4).reshape((matrix.shape[0], 4), order='C')
        neighbours = is_neighbours(last_int_column_repeated, domain_repeated)
        d = distribution(column_repeated, domain_repeated) * neighbours
        cdf = np.cumsum(d / np.reshape(np.sum(d, axis=1),
                                       (matrix.shape[0], 1)), axis=1)

        u = np.random.uniform(size=(matrix.shape[0], 1))
        indices = np.argmax(u <= cdf, axis=1)
        res[:, j] = output_domain[indices]

    if return_float:
        return res, matrix
    else:
        return res


def simulate_synthetic(
    basis, explained_variance_ratios,
    parameters_simulate_ordinal,
    parameters_simulate_mask,
    return_masked=False,
    return_float=False,
):
    """Simulation of a complete synthetic dataset.

    Parameters
    ----------
    parameters_simulate_ordinal : See simulate_ordinal_from_float.
    parameters_simulate_mask : See simulate_mask

    Returns
    ----------
    X_synthetic_final : Simulated dataset in CSR format.
    """
    if return_float:
        X_synthetic_ordinal, X_synthethic_float = simulate_integer_from_float(
            basis, explained_variance_ratios,
            parameters=parameters_simulate_ordinal,
            return_float=True
        )
    else:
        X_synthetic_ordinal = simulate_integer_from_float(
            basis, explained_variance_ratios,
            parameters=parameters_simulate_ordinal
        )

    mask = simulate_mask(
        X_synthetic_ordinal,
        parameters=parameters_simulate_mask
    )

    X_synthetic_final = X_synthetic_ordinal*mask

    if return_masked & return_float:
        return X_synthetic_final, X_synthetic_ordinal, X_synthethic_float
    elif return_masked:
        return X_synthetic_final, X_synthetic_ordinal
    else:
        return X_synthetic_final
