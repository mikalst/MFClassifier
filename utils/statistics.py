import numpy as np
import scipy.sparse

def compute_discrete_distribution_from_sparse_matrix(matrix):
    
    if scipy.sparse.issparse(matrix):
        x, px = np.unique(matrix.todense().A1, return_counts=True)
    else:
        x, px = np.unique(matrix.flat, return_counts=True)
    
    px_nonzero = px[x != 0] / np.sum(px[x != 0])
    
    return (x[x != 0], px_nonzero, np.cumsum(px_nonzero))