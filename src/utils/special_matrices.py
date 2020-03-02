import numpy as np

def finite_difference_matrix(shape):
    return (np.diag(np.pad(-np.ones(shape[1] - 1), (0, 1), 'constant')) + np.diag(np.ones(shape[1]-1), 1))