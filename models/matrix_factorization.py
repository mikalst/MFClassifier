import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics


class MatrixFactorization:
    """A class for solving the inverse problem
    || S - U V^T ||_F^2 + l1 || U ||_F^2 + l2 || V ||_F^2 + l3 || C R V||_F^2
    subject to
    P_{\Omega}(S) = P_{\Omega}(Y)

    This yields a K-rank approximation of the input Y. Designed for the
    approximation of temporal patient data.
    """

    def __init__(self, args):
        # Regularization parameters
        self.lambda0 = args["lambda0"]
        self.lambda1 = args["lambda1"]
        self.lambda2 = args["lambda2"]
        self.lambda3 = args["lambda3"]

        self.Y = args["Y"]  # Input data
        self.R = args["R"]  # Operation on temporal profile
        self.J = args["J"]  # Default temporal profile
        self.C = args["C"]  # Convolution

        self.K = args["K"]  # Rank

        self.nonzero_rows, self.nonzero_cols = np.nonzero(self.Y)
        self.N = self.Y.shape[0]
        self.T = self.Y.shape[1]

        # Initialize U, V and S
        self.U = np.ones((self.N, self.K))
        self.V = np.ones((self.T, self.K)) * \
            np.mean(self.Y[self.nonzero_rows, self.nonzero_cols])
        self.S = self.Y.copy()

        self.U_old = np.zeros((self.N, self.K))
        self.V_old = np.zeros((self.T, self.K))

        self.iteration = 0
        self.total_iterations = args["total_iterations"]
        self.tol = args["convergence_tol"]

        # Code optimization purposes
        # Static variables are computed and stored
        self.RTCTCR = (self.C @ self.R).T@(self.C@self.R)
        self.L2, self.Q2 = np.linalg.eigh((self.lambda3 / self.lambda0) * self.RTCTCR)

    def solve1(self):
        U = (
            np.linalg.solve(
                self.V.T @ self.V +
                (self.lambda1 / self.lambda0) * np.identity(self.K),
                self.V.T @ self.S.T,
            )
        ).T

        return U

    def solve2(self):
        L1, Q1 = np.linalg.eigh(
            self.U.T @ self.U +
            (self.lambda2 / self.lambda0) * np.identity(self.K)
        )

        # For efficiency purposes, these need to be evaluated in order
        hatV = (
            (self.Q2.T @ (self.S.T @ self.U + (self.lambda2 / self.lambda0) * self.J))
            @ Q1
            / np.add.outer(self.L2, L1)
        )
        V = self.Q2 @ (hatV @ Q1.T)

        return V

    def solve3(self):
        S = self.U @ self.V.T
        S[self.nonzero_rows, self.nonzero_cols] = self.Y[
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
            self.U_old = np.copy(self.U)
            self.V_old = np.copy(self.V)
            self.solve_inner()
            self.iteration += 1
            if (
                np.linalg.norm(self.U_old @ self.V_old.T - self.U @ self.V.T)
                / np.linalg.norm(self.U @ self.V.T)
                < self.tol
            ) or self.iteration > self.total_iterations:
                return True
            return False
        self.solve_inner()
        self.iteration += 1
        return False

    def calculate_loglikelihood(self, Y_pred, theta_estimate):
        """For each row y in Y_pred, calculate the loglikelihood of y having originated
        from the reconstructed continuous profile of all the patients in the training set.
        """

        def loglikelihood_func(eta):
            return -theta_estimate * eta ** 2

        M_train = self.U @ self.V.T

        N_1 = M_train.shape[0]
        N_2 = Y_pred.shape[0]

        logL = np.ones((N_2, N_1))

        for i in range(N_2):
            row_nonzero_cols = Y_pred[i] != 0
            linear_predictors = (Y_pred[i, row_nonzero_cols])[None, :] - M_train[
                :, row_nonzero_cols
            ]
            logL[i] = np.sum(loglikelihood_func(linear_predictors), axis=1)

        return logL

    def predict(self, Y_pred, t, output_domain, theta_estimate):
        """For each row y in Y_pred, calculate the most likely integer value for a future
        fixed time t.
        """

        def distribution_z(eta):
            return np.exp(-theta_estimate * eta ** 2)

        logL = self.calculate_loglikelihood(Y_pred, theta_estimate)

        trainM = self.U @ self.V.T

        p_of_z_given_m = distribution_z(trainM[:, t, None] - output_domain)

        return np.exp(logL) @ p_of_z_given_m


class MatrixFactorizationFull(MatrixFactorization):
    """An extension of the MatrixFactorization class. Used for testing.
    """

    def f(self, U, V):
        return (
            self.lambda0 * np.sum(np.power((self.X - U @ V.T)[self.X != 0], 2))
            + self.lambda1 * np.linalg.norm(U) ** 2
            + self.lambda2 * np.linalg.norm(V - self.J) ** 2
            + self.lambda3 * np.linalg.norm(self.C @ self.R @ V) ** 2
        )

    def f1(self, u):
        U = u.reshape((self.N, self.K))
        return (
            self.lambda0 * np.linalg.norm(self.S - U @ self.V.T) ** 2
            + self.lambda1 * np.linalg.norm(U) ** 2
        )

    def grad1(self, u):
        U = u.reshape((self.N, self.K))
        return (
            -2 * self.lambda0 *
            (self.S - U @ self.V.T) @ self.V + 2 * self.lambda1 * U
        ).reshape(self.N * self.K)

    def f2(self, v):
        V = v.reshape((self.T, self.K))

        res1 = self.lambda0 * np.linalg.norm(self.S - self.U @ V.T) ** 2
        res2 = self.lambda2 * np.linalg.norm(V - self.J) ** 2
        res3 = self.lambda3 * np.linalg.norm((self.C @ self.R) @ V) ** 2

        return res1 + res2 + res3

    def grad2(self, v):
        V = v.reshape((self.T, self.K))

        res1 = -2 * self.lambda0 * (self.S.T - V @ self.U.T) @ self.U
        res2 = 2 * self.lambda2 * (V - self.J)

        CR = self.C @ self.R
        res3 = 2 * self.lambda3 * (CR).T @ ((CR) @ V)

        return (res1 + res2 + res3).reshape(self.T * self.K)
