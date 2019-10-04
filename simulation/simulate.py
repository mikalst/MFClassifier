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