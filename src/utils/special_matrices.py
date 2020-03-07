import numpy as np

def finite_difference_matrix(T):
    return (np.diag(np.pad(-np.ones(T - 1), (0, 1), 'constant')) + np.diag(np.ones(T-1), 1))