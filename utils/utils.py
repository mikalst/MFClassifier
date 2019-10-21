import scipy.interpolate
import numpy as np


def extract_smoothened_arch_timelines(pca_components):

    n_storylines = len(pca_components)
    
    index_storyline = np.arange(n_storylines)

    pca_components_smooth = np.empty_like(pca_components)

    for i, tl in enumerate(pca_components):
        spl = scipy.interpolate.UnivariateSpline(np.linspace(0, 320, 321), tl, k=3)
        spl.set_smoothing_factor(5e-1)
        pca_components_smooth[i] = spl(np.linspace(0, 320, 321))

    return pca_components_smooth
