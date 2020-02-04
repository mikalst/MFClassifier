#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions used in finite dimensional optimization
"""

import numpy as np


def compare_gradient_to_fin_diff(f, g, x_0, eps=1e-4, n_test_direction=30):

    shape = x_0.shape

    fin_diff_grad = np.empty(n_test_direction, dtype=np.float64)

    idc = np.random.randint(
        low=0,
        high=np.prod(shape),
        size=n_test_direction
    )

    x_0_vec = x_0.flatten()

    for num in range(n_test_direction):
        x_eps_vec = np.copy(x_0_vec)

        x_eps_vec[idc[num]] += eps

        x_eps_unvec = x_eps_vec.reshape(shape)
    
        fin_diff_grad[num] = (f(x_eps_unvec) - f(x_0))/eps

    grad = (g(x_0)).flatten()[idc]

    print(fin_diff_grad)
    print(grad)

    print(
        "norm of error in {} random elements = {:.4e}".format(
            n_test_direction,
            np.linalg.norm(fin_diff_grad - grad)
        )
    )
