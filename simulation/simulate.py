import numpy as np
import scipy.sparse


def simulate_mask(X, f_of_t, expectation_of_priors, variances, level=0.3):
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
    mask = np.zeros_like(X, dtype=np.bool)
    individual_probs = np.empty((X.shape[0], expectation_of_priors.shape[0]))

    # Assign individual transition probabilities
    for i in range(expectation_of_priors.shape[0]):
        individual_probs[:, i] = np.random.beta(variances[i]*expectation_of_priors[i],
                                                variances[i]*(1-expectation_of_priors[i]), size=(X.shape[0]))
    
    for t in range(X.shape[1] - 1):
        values_at_current_time = X[:, t]*mask[:, t]
        indices = ((np.arange(5))[:, None] == values_at_current_time).T
        p = 1e3*level * individual_probs[indices] * f_of_t(t)
        r = np.random.uniform(size=(X.shape[0]))
        mask[r <= p, t+1] = True
        
    return mask


def simulate_ordinal_from_float(matrix, pdf, ordinal_domain, kernel_parameter=1.0, truncate_limits=[0.5, 1.0]):
    """Simulation of ordinal data from scores.
    
    Parameters
    ----------
    matrix : Scores.
    pdf : The (approximate) resulting distribution of the ordinal values.
    ordinal_domain : Possible values of the output.
    kernel_parameter : The strength with which a score will seek towards the corresponding ordinal value.
    truncate_limits : Limits for tuning the tails of the resulting distribution.
    
    Returns
    ----------
    res : The resulting matrix of ordinal data.
    """
    lower_truncate_limit, upper_truncate_limit = np.quantile(matrix, truncate_limits)
    domain_max = np.max(ordinal_domain)
    domain_min = np.min(ordinal_domain)

    matrix = domain_min + (domain_max - domain_min) * (matrix - lower_truncate_limit)/(upper_truncate_limit - lower_truncate_limit)
    matrix[matrix < domain_min] = domain_min
    matrix[matrix > domain_max] = domain_max

    distribution = lambda x, dom: np.exp(-kernel_parameter*np.abs(x - dom))
    is_neighbours = lambda x, y: np.abs(x - y) <= 1

    domain_repeated = np.repeat(ordinal_domain, matrix.shape[0]).reshape((matrix.shape[0], ordinal_domain.shape[0]), order='F')

    res = np.empty_like(matrix)

    # Initialization
    column_repeated = np.repeat(matrix[:, 0], 4).reshape((matrix.shape[0], 4), order='C')
    d = distribution(column_repeated, domain_repeated) * pdf
    cdf = np.cumsum(d / np.reshape(np.sum(d, axis=1), (matrix.shape[0], 1)), axis=1)

    u = np.random.uniform(size=(matrix.shape[0], 1))
    indices = np.argmax(u <= cdf, axis=1)
    res[:, 0] = ordinal_domain[indices]

    # Timestepping
    for j in range(1, matrix.shape[1]):
        column_repeated = np.repeat(matrix[:, j], 4).reshape((matrix.shape[0], 4), order='C')
        last_int_column_repeated = np.repeat(res[:, j-1], 4).reshape((matrix.shape[0], 4), order='C')
        neighbours = is_neighbours(last_int_column_repeated, domain_repeated)
        d = distribution(column_repeated, domain_repeated) * neighbours * pdf
        cdf = np.cumsum(d / np.reshape(np.sum(d, axis=1), (matrix.shape[0], 1)), axis=1)

        u = np.random.uniform(size=(matrix.shape[0], 1))
        indices = np.argmax(u <= cdf, axis=1)
        res[:, j] = ordinal_domain[indices]
    
    return res


def simulate_synthetic(parameters_simulate_ordinal, parameters_simulate_mask):
    """Simulation of a complete synthetic dataset.
    
    Parameters
    ----------
    parameters_simulate_ordinal : See simulate_ordinal_from_float.
    parameters_simulate_mask : See simulate_mask

    Returns
    ----------
    X_synthetic_final : Simulated dataset in CSR format.
    """
    basis = parameters_simulate_ordinal['basis']
    explained_variance_ratios = parameters_simulate_ordinal['explained_variance_ratios']
    original_data_pdf = parameters_simulate_ordinal['original_data_pdf']
    output_domain = parameters_simulate_ordinal['output_domain']
    kernel_parameter = parameters_simulate_ordinal['kernel_parameter']
    truncate_limits = parameters_simulate_ordinal['truncate_limits']
    
    marginal_distribution_time = parameters_simulate_mask['marginal_distribution_time']
    mask_transition_expectations = parameters_simulate_mask['mask_transition_expectations']
    mask_transition_variances = parameters_simulate_mask['mask_transition_variances']
    mask_level = parameters_simulate_mask['mask_level']

    weights = np.random.uniform(size=(38001, basis.shape[0])) * explained_variance_ratios
    weights_normalized = weights

    X_synthetic_float = weights_normalized@basis

    X_synthetic_ordinal = simulate_ordinal_from_float(X_synthetic_float, original_data_pdf, ordinal_domain = output_domain,
                                                kernel_parameter=2.5, truncate_limits=[0.75, .9975])

    mask = simulate_mask(X_synthetic_ordinal, marginal_distribution_time, mask_transition_expectations,
                        mask_transition_variances, level=mask_level)

    X_synthetic_final = scipy.sparse.csr_matrix(X_synthetic_ordinal*mask)

    return X_synthetic_final