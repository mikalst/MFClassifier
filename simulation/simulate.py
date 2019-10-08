import numpy as np

def simulate_mask(X, f_of_t, expectation_of_priors=np.array([0.028, 0.023, 0.192, 0.615, 0.228]), variances=(20, 20), level=0.3):
    mask = np.zeros_like(X, dtype=np.bool)

    individual_probs = np.empty((X.shape[0], 5))

    var, var_going_in_check = variances

    individual_probs[:, 0] = np.random.beta(var_going_in_check*expectation_of_priors[0],
                                            var_going_in_check*(1-expectation_of_priors[0]), size=(X.shape[0]))
    individual_probs[:, 1] = np.random.beta(var*expectation_of_priors[1],
                                            var*(1-expectation_of_priors[1]), size=(X.shape[0]))
    individual_probs[:, 2] = np.random.beta(var*expectation_of_priors[2],
                                            var*(1-expectation_of_priors[2]), size=(X.shape[0]))
    individual_probs[:, 3] = np.random.beta(var*expectation_of_priors[3],
                                            var*(1-expectation_of_priors[3]), size=(X.shape[0]))
    individual_probs[:, 4] = np.random.beta(var*expectation_of_priors[4],
                                            var*(1-expectation_of_priors[4]), size=(X.shape[0]))

    for t in range(X.shape[1] - 1):

        r = np.random.uniform(size=(X.shape[0]))

        values_at_current_time = X[:, t]*mask[:, t]

        # Individual probability index
        indices = ((np.arange(5))[:, None] == values_at_current_time).T

        p = 1e3*level * individual_probs[indices] * f_of_t(t)

        mask[r <= p, t+1] = True
        
    return mask

def simulate_ordinal_from_float(matrix, pdf, kernel_parameter=1.0):
    
    matrix = (matrix - matrix.mean())/np.std(matrix) + 1.0

    dom = np.array([1, 2, 3, 4])
    distribution = lambda x, dom: np.exp(-kernel_parameter*np.abs(x - dom))
    neighbours = lambda x, y: np.abs(x - y) <= 1

    res = np.empty_like(matrix)
    
    dom_repeated = np.repeat(dom, matrix.shape[0]).reshape((matrix.shape[0], 4), order='F')

    # Initialization
    column_repeated = np.repeat(matrix[:, 0], 4).reshape((matrix.shape[0], 4), order='C')

    d = distribution(column_repeated, dom_repeated) * pdf
    cdf = np.cumsum(d / np.reshape(np.sum(d, axis=1), (matrix.shape[0], 1)), axis=1)
    u = np.random.uniform(size=(matrix.shape[0], 1))
    indices = np.argmax(u <= cdf, axis=1)

    res[:, 0] = dom[indices]

    for j in range(1, matrix.shape[1]):

        column_repeated = np.repeat(matrix[:, j], 4).reshape((matrix.shape[0], 4), order='C')
        last_int_column_repeated = np.repeat(res[:, j-1], 4).reshape((matrix.shape[0], 4), order='C')
        dom_repeated = np.repeat(dom, matrix.shape[0]).reshape((matrix.shape[0], 4), order='F')
        is_neighbours = neighbours(last_int_column_repeated, dom_repeated)

        d = distribution(column_repeated, dom_repeated) * is_neighbours * pdf
        cdf = np.cumsum(d / np.reshape(np.sum(d, axis=1), (matrix.shape[0], 1)), axis=1)

        u = np.random.uniform(size=(matrix.shape[0], 1))

        indices = np.argmax(u <= cdf, axis=1)

        res[:, j] = dom[indices]
    
    return res