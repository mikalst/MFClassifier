"""
Algorithms for simulating matrices that exhibit the same characteristics as Norwegian cervical cancer screening data.
"""

import numpy as np


def simulate_integer_from_float(
    X_float_unscaled,
    integer_parameters,
    return_float=False,
    seed=None
):
    """Simulation of integer data from floats.

    Parameters
    ----------
    X_float_unscaled       : Scores.
    integer_parameters   : Parameters for the simulation.
        output_domain    : Subset of the integers included in the output.
        kernel_parameter : Parameter used in the pmf.
    return_float : Return input.
    seed         : Replication of results.

    Returns
    ----------
    res : Simulated integer X_float_unscaled 
    """
    output_domain = integer_parameters['output_domain']
    kernel_parameter = integer_parameters['kernel_parameter']

    if not(seed is None):
        np.random.seed(seed)

    domain_max = np.max(output_domain)
    domain_min = np.min(output_domain)

    N = X_float_unscaled.shape[0]
    T = X_float_unscaled.shape[1]
    Z = output_domain.shape[0]

    X_float_scaled = domain_min + (domain_max - domain_min)*(X_float_unscaled -
                                                             np.min(X_float_unscaled))/(np.max(X_float_unscaled) - np.min(X_float_unscaled))

    def distribution(x, dom): return np.exp(-kernel_parameter*(x - dom)**2)

    domain_repeated = np.repeat(output_domain, N).reshape((N, Z), order='F')

    X_integer = np.empty_like(X_float_scaled)

    # Initialization
    column_repeated = np.repeat(
        X_float_scaled[:, 0], 4).reshape((N, 4), order='C')
    pdf = distribution(column_repeated, domain_repeated)
    cdf = np.cumsum(pdf / np.reshape(np.sum(pdf, axis=1), (N, 1)), axis=1)

    u = np.random.uniform(size=(N, 1))
    indices = np.argmax(u <= cdf, axis=1)
    X_integer[:, 0] = output_domain[indices]

    # Timestepping
    for j in range(1, T):
        column_repeated = np.repeat(
            X_float_scaled[:, j], 4).reshape((N, 4), order='C')
        pdf = distribution(column_repeated, domain_repeated)
        cdf = np.cumsum(pdf / np.reshape(np.sum(pdf, axis=1), (N, 1)), axis=1)

        u = np.random.uniform(size=(N, 1))
        indices = np.argmax(u <= cdf, axis=1)
        X_integer[:, j] = output_domain[indices]

    if return_float:
        return X_integer, X_float_scaled
    else:
        return X_integer


def simulate_mask(
    X_integer,
    mask_parameters,
    path_dropout=None
):
    """Simulation of a missing data mask.

    Parameters
    ----------
    X_integer : The unmasked integer-valued matrix.
    mask_parameters : 
        mask_transition_expectations : E[p_ik] for k = 1, 2, ... Z
        mask_transition_variances : Var[p_ik] for k = 1, 2, ... Z
        memory_length : Determines how long each screening result is remembered.
        mask_level : Affects the probability of coming in for a screening across all patients.

    Returns
    ----------
    mask : The resulting mask.
    """
    mask_transition_expectations = mask_parameters['mask_transition_expectations']
    mask_transition_variances = mask_parameters['mask_transition_variances']
    mask_memory_length = mask_parameters['memory_length']
    mask_level = mask_parameters['mask_level']

    N = X_integer.shape[0]
    T = X_integer.shape[1]

    mask = np.zeros_like(X_integer, dtype=np.bool)
    observed_values = np.zeros((N, T))
    individual_probs = np.empty((N, mask_transition_expectations.shape[0]))

    # Assign individual transition probabilities
    for i in range(mask_transition_expectations.shape[0]):
        individual_probs[:, i] = np.random.beta(mask_transition_variances[i]*mask_transition_expectations[i],
                                                mask_transition_variances[i]*(1-mask_transition_expectations[i]), size=N)

    for t in range(T - 1):
        # Find last remembered values
        last_remembered_values = observed_values[np.arange(
            N), t+1-np.argmax(observed_values[:, t+1:max(0, t-mask_memory_length):-1] != 0, axis=1)]

        indices = ((np.arange(np.max(X_integer)+1))
                   [:, None] == last_remembered_values).T
        p = mask_level*individual_probs[indices]
        r = np.random.uniform(size=N)
        mask[r <= p, t+1] = True
        observed_values[r <= p, t+1] = X_integer[r <= p, t+1]

    # Simulate dropout
    if not(path_dropout is None):
        prob_dropout = np.load(path_dropout)
        tpoints = np.arange(X_integer.shape[1])

        for num in range(X_integer.shape[0]):
            t_max = np.random.choice(tpoints, p=prob_dropout, replace=True)
            mask[num, t_max:] = 0

    return mask

def simulate_synthetic(
    X_float_unscaled,
    integer_parameters,
    mask_parameters,
    path_dropout=None,
    seed=None
):
    """Simulation of a complete synthetic dataset.

    Parameters
    ----------
    See simulate_ordinal_from_float and simulate_mask.

    Returns
    ----------
    X_synthetic : Simulated masked integer-valued dataset.
    """

    X_integer = simulate_integer_from_float(
        X_float_unscaled,
        integer_parameters=integer_parameters
    )

    mask = simulate_mask(
        X_integer,
        mask_parameters=mask_parameters,
        path_dropout=path_dropout
    )

    return X_integer*mask

def summary(
    X,
    thresh=2
):
    """Summarize the produced synthetic dataset"""
    
    print('Density:', np.count_nonzero(X) / X.size * 100, '%')

    I = np.where(np.sum(X, axis=1) != 0)
    Xnz = X[np.unique(I)]

    vls, cnts = np.unique(Xnz[Xnz != 0], return_counts=True)   
    print('Unique values:', vls)
    print('Count:', cnts)
    print('Prcnt:', cnts / sum(cnts) * 100)
    print('Num positives: {} Num negatives: {}'.format(np.sum(Xnz > thresh), np.sum(Xnz <= thresh)))
        
    # Several test rows will all zeros.
    nz_rows = np.sum(X != 0, axis=1)
    print('Min trajectory nz count:', min(nz_rows))
    print('Max trajectory nz count:', max(nz_rows))

    v, c = np.unique(np.argmax(Xnz != 0, axis=1), return_counts=True) 
    print('Start indices of test sequences:\n', v)