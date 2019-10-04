import numpy as np
import scipy.sparse

def compute_discrete_distribution_from_sparse_matrix(matrix):
    
    if scipy.sparse.issparse(matrix):
        x, px = np.unique(matrix.todense().A1, return_counts=True)
    else:
        x, px = np.unique(matrix.flat, return_counts=True)
    
    px_nonzero = px[x != 0] / np.sum(px[x != 0])
    
    return (x[x != 0], px_nonzero, np.cumsum(px_nonzero))


def distribution_per_age(matrix, density=True):
    
    if density:
        if scipy.sparse.issparse(matrix):
            data1 = np.sum(matrix == 3, axis=0).A1 / np.sum(matrix == 3)
            data2 = np.sum(matrix == 4, axis=0).A1 / np.sum(matrix == 4)
            data3 = np.sum(matrix != 0, axis=0).A1 / matrix.count_nonzero()

        else:
            data1 = np.sum(matrix == 3, axis=0) / np.sum(matrix == 3)
            data2 = np.sum(matrix == 4, axis=0) / np.sum(matrix == 4)
            data3 = np.sum(matrix != 0, axis=0) / np.sum(matrix != 0)
            
    else:
        if scipy.sparse.issparse(matrix):
            data1 = np.sum(matrix == 3, axis=0).A1
            data2 = np.sum(matrix == 4, axis=0).A1
            data3 = np.sum(matrix != 0, axis=0).A1

        else:
            data1 = np.sum(matrix == 3, axis=0)
            data2 = np.sum(matrix == 4, axis=0)
            data3 = np.sum(matrix != 0, axis=0)
    
    return data1, data2, data3