import numpy as np

def compare_gradient_to_fin_diff(f, g, x_0, eps=1e-4, dims_to_compare = 30):
    
    ndim = x_0.shape[0]
    
    random_dims = np.random.randint(0, ndim, size=dims_to_compare)
    verbose_indices = np.random.choice(range(dims_to_compare), size=min(5, dims_to_compare), replace=False)
    
    fin_diff_grad = np.empty_like(random_dims, dtype=np.float64)
    grad = g(x_0)[random_dims]
    
    verbose_grad = grad[verbose_indices]
    
    for i, d in enumerate(random_dims):
        x_eps = np.copy(x_0)
        x_eps[d] += eps
        fin_diff_grad[i] = (f(x_eps) - f(x_0))/eps
    
    print(fin_diff_grad[verbose_indices])
    print(verbose_grad)
        
    print("norm of error in {} random elements = {:.4e}".format(dims_to_compare, np.linalg.norm(fin_diff_grad - grad)))