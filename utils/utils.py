import scipy.interpolate
import numpy as np


def extract_smoothened_arch_timelines(pca_components):

    n_storylines = len(pca_components)
    
    index_storyline = np.arange(n_storylines)

    pca_components_smooth = np.empty_like(pca_components)

    for i, tl in enumerate(pca_components):
        spl = scipy.interpolate.UnivariateSpline(np.linspace(0, 320, 321), tl, k=3)
        spl.set_smoothing_factor(2e-4)
        pca_components_smooth[i] = spl(np.linspace(0, 320, 321))

    return pca_components_smooth


def convert_float_to_categories(matrix, cumulative_distribution):
    matrix_flat = matrix.flatten()
    sorted_indices = np.argsort(matrix_flat)[::-1]
    matrix_flat_sorted = matrix_flat[sorted_indices]

    matrix_flat_sorted[0:int(cumulative_distribution[0] * len(sorted_indices))] = 1

    for k in range(1, len(cumulative_distribution)):
        matrix_flat_sorted[int(cumulative_distribution[k-1] * len(sorted_indices)): \
                           int(cumulative_distribution[k] * len(sorted_indices))] = k+1

    matrix_flat_categorical = np.empty_like(matrix_flat, dtype=np.int64)
    matrix_flat_categorical[sorted_indices] = matrix_flat_sorted.astype(np.int64)

    return matrix_flat_categorical.reshape(matrix.shape)