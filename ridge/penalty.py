import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics


class MatrixFactorization:
    """A class for the reconstruction of the underlying real-valued matrix M from an
    observed integer-valued matrix Y, where Y ~ M is observed only at the nonzero entries.
    """

    def __init__(self, kwargs):
        # Regularization parameters
        self.lambda0 = kwargs["lambda0"]
        self.lambda1 = kwargs["lambda1"]
        self.lambda2 = kwargs["lambda2"]
        self.lambda3 = kwargs["lambda3"]

        self.X = kwargs["X"]
        self.R = kwargs["R"]
        self.J = kwargs["J"]
        self.kappa = kwargs["kappa"]

        # Set nonzero rows and columns automatically after assigning predict X
        self.nonzero_rows, self.nonzero_cols = np.nonzero(self.X)

        self.N = self.X.shape[0]
        self.T = self.X.shape[1]
        self.k = kwargs["k"]

        # Initialize U, V and S
        self.U = (
            np.ones(self.N * self.k) + (1 - np.random.uniform(size=self.N * self.k))
        ).reshape((self.N, self.k))
        self.V = (
            np.ones(self.T * self.k) + (1 - np.random.uniform(size=self.T * self.k))
        ).reshape((self.T, self.k))
        self.S = self.X.copy()

        self.oldU = np.zeros_like(self.U)
        self.oldV = np.zeros_like(self.V)

        self.iteration = 0
        self.total_iterations = kwargs["total_iterations"]
        self.tol = kwargs["convergence_tol"]

    def f(self, U, V):
        return (
            self.lambda0 * np.sum(np.power((self.X - U @ V.T)[self.X != 0], 2))
            + self.lambda1 * np.linalg.norm(U) ** 2
            + self.lambda2 * np.linalg.norm(V - self.J) ** 2
            + self.lambda3 * np.linalg.norm(self.kappa @ self.R @ V) ** 2
        )

    def f1(self, u):
        U = u.reshape((self.N, self.k))
        return (
            self.lambda0 * np.linalg.norm(self.S - U @ self.V.T) ** 2
            + self.lambda1 * np.linalg.norm(U) ** 2
        )

    def grad1(self, u):
        U = u.reshape((self.N, self.k))
        return (
            -2 * self.lambda0 * (self.S - U @ self.V.T) @ self.V + 2 * self.lambda1 * U
        ).reshape(self.N * self.k)

    def solve1(self):
        U = (
            np.linalg.solve(
                self.V.T @ self.V + (self.lambda1 / self.lambda0) * np.identity(self.k),
                self.V.T @ self.S.T,
            )
        ).T

        return U

    def f2(self, v):
        V = v.reshape((self.T, self.k))

        res1 = self.lambda0 * np.linalg.norm(self.S - self.U @ V.T) ** 2
        res2 = self.lambda2 * np.linalg.norm(V - self.J) ** 2
        res3 = self.lambda3 * np.linalg.norm((self.kappa @ self.R) @ V) ** 2

        return res1 + res2 + res3

    def grad2(self, v):
        V = v.reshape((self.T, self.k))

        res1 = -2 * self.lambda0 * (self.S.T - V @ self.U.T) @ self.U
        res2 = 2 * self.lambda2 * (V - self.J)

        kappaR = self.kappa @ self.R
        res3 = 2 * self.lambda3 * (kappaR).T @ ((kappaR) @ V)

        return (res1 + res2 + res3).reshape(self.T * self.k)

    def solve2(self):
        L1, Q1 = np.linalg.eigh(
            self.U.T @ self.U + (self.lambda2 / self.lambda0) * np.identity(self.k)
        )

        kappaR = self.kappa @ self.R

        L2, Q2 = np.linalg.eigh((self.lambda3 / self.lambda0) * (kappaR).T @ (kappaR))

        # For efficiency purposes, these need to be evaluated in order
        hatV = (
            (Q2.T @ (self.S.T @ self.U + (self.lambda2 / self.lambda0) * self.J))
            @ Q1
            / np.add.outer(L2, L1)
        )

        V = Q2 @ (hatV @ Q1.T)

        return V

    def solve3(self):
        S = self.U @ self.V.T
        S[self.nonzero_rows, self.nonzero_cols] = self.X[
            self.nonzero_rows, self.nonzero_cols
        ]

        return S

    def solve_inner(self):
        self.U = self.solve1()
        self.V = self.solve2()
        self.S = self.solve3()
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration % 50 == 0:
            self.oldU = np.copy(self.U)
            self.oldV = np.copy(self.V)
            self.solve_inner()
            self.iteration += 1
            if (
                np.linalg.norm(self.oldU @ self.oldV.T - self.U @ self.V.T)
                / np.linalg.norm(self.U @ self.V.T)
                < self.tol
            ):
                print("Converged!")
                return True
            return False
        self.solve_inner()
        self.iteration += 1
        return False

    def calculate_loglikelihood(self, X_pred, theta_estimate=2.5):
        """For each row y in X_pred, calculate the loglikelihood of y having originated
        from the reconstructed continuous profile of all the patients in the training set.
        """

        def loglikelihood_func(eta):
            return -theta_estimate * eta ** 2

        trainM = self.U @ self.V.T

        N_1 = trainM.shape[0]
        N_2 = X_pred.shape[0]

        logL = np.ones((N_2, N_1))

        for i in range(N_2):
            row_nonzero_cols = X_pred[i] != 0
            linear_predictors = (X_pred[i, row_nonzero_cols])[None, :] - trainM[
                :, row_nonzero_cols
            ]
            logL[i] = np.sum(loglikelihood_func(linear_predictors), axis=1)

        return logL

    def predict(self, X_pred, t, output_domain=np.arange(1, 5), theta_estimate=2.5):
        """For each row y in X_pred, calculate the most likely integer value for a future
        fixed time t.
        """

        def distribution_z(eta):
            return np.exp(-theta_estimate * eta ** 2)

        logL = self.calculate_loglikelihood(X_pred)

        trainM = self.U @ self.V.T

        p_of_z_given_m = distribution_z(trainM[:, t, None] - output_domain)

        return np.exp(logL) @ p_of_z_given_m
