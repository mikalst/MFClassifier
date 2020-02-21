import numpy as np


class MatrixFactorization:
    r"""A class for solving the inverse problem
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
        self.L2, self.Q2 = np.linalg.eigh(
            (self.lambda3 / self.lambda0) * self.RTCTCR)

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

    def train(self):
        while True:
            converged = next(self)
            if converged:
                break

    def loglikelihood(self, Y_pred, theta_estimate):
        r"""For each row y in Y_pred, calculate the loglikelihood of y having originated
        from the reconstructed continuous profile of all the patients in the training set.
        """
        M_train = self.U @ self.V.T

        N_1 = M_train.shape[0]
        N_2 = Y_pred.shape[0]

        logL = np.ones((N_2, N_1))

        for i in range(N_2):
            row_nonzero_cols = Y_pred[i] != 0
            eta_i = (Y_pred[i, row_nonzero_cols])[None, :] - M_train[
                :, row_nonzero_cols
            ]
            logL[i] = np.sum(-theta_estimate*np.power(eta_i, 2), axis=1)

        return logL

    def posterior(self, Y_pred, t, domain, theta_estimate):
        r"""For each row y in Y_pred, calculate the posterior probability
        of each integer value in the output domain for a future
        time t.
        """
        logL = self.loglikelihood(Y_pred, theta_estimate)
        trainM = self.U @ self.V.T

        # Ensure that can be properly indexed.
        domain = np.array(domain)

        p_z = np.empty((Y_pred.shape[0], domain.shape[0]))

        for i in range(Y_pred.shape[0]):
            p_z[i] = np.exp(logL[i]) @ np.exp(-theta_estimate *
                                              (trainM[:, t[i], None] - domain)**2)

        # Normalize
        p_z_normalized = p_z / (np.sum(p_z, axis=1))[:, None]

        return p_z_normalized

    def posterior_rulebased(self, Y_pred, t, domain, theta_estimate, rule, rule_outcomes):
        r"""For each row y in Y_pred, calculate the posterior probability
        of each rule outcome e by mapping rule over the integer values in the
        domain and summing over all integer that result in rule outcome e.
        """
        p_z = self.posterior(Y_pred, t, domain, theta_estimate)

        p_e = np.empty((Y_pred.shape[0], rule_outcomes.shape[0]))

        for i_event, e in enumerate(rule_outcomes):

            values_of_z_where_e_happens = np.argwhere(
                [rule(z) for z in domain] == e)

            p_e[:, i_event] = (
                np.sum(p_z[:, values_of_z_where_e_happens], axis=1)).flatten()

        return p_e

    def predict(self, Y_pred, t, output_domain, theta_estimate):
        r"""For each row y in Y_pred, calculate the highest posterior
        probability integer value for a future time t.
        """
        p_z = self.posterior(Y_pred, t, output_domain, theta_estimate)

        # Ensure that output_domain can be properly indexed.
        output_domain = np.array(output_domain)

        return output_domain[np.argmax(p_z, axis=1)]

    def predict_rulebased(self, Y_pred, t, output_domain, theta_estimate, rule, rule_outcomes, rule_bias):
        r"""For each row y in Y_pred, calculate the highest posterior
        probability rule outcome e for a future time t.
        """
        p_e = self.posterior_rulebased(
            Y_pred, t, output_domain, theta_estimate, rule, rule_outcomes)

        return rule_outcomes[np.argmax(p_e*rule_bias, axis=1)]


class MatrixFactorizationTesting(MatrixFactorization):
    """An extension of the MatrixFactorization class. Used for testing.
    """

    def f(self, U, V):
        return (
            self.lambda0 * np.sum(np.power((self.Y - U @ V.T)[self.Y != 0], 2))
            + self.lambda1 * np.linalg.norm(U) ** 2
            + self.lambda2 * np.linalg.norm(V - self.J) ** 2
            + self.lambda3 * np.linalg.norm(self.C @ self.R @ V) ** 2
        )

    def f1(self, U):
        return (
            self.lambda0 * np.linalg.norm(self.S - U @ self.V.T) ** 2
            + self.lambda1 * np.linalg.norm(U) ** 2
        )

    def grad1(self, U):
        return (
            -2 * self.lambda0 *
            (self.S - U @ self.V.T) @ self.V + 2 * self.lambda1 * U
        ).reshape(self.N * self.K)

    def f2(self, V):
        res1 = self.lambda0 * np.linalg.norm(self.S - self.U @ V.T) ** 2
        res2 = self.lambda2 * np.linalg.norm(V - self.J) ** 2
        res3 = self.lambda3 * np.linalg.norm((self.C @ self.R) @ V) ** 2

        return res1 + res2 + res3

    def grad2(self, V):
        res1 = -2 * self.lambda0 * (self.S.T - V @ self.U.T) @ self.U
        res2 = 2 * self.lambda2 * (V - self.J)
        res3 = 2 * self.lambda3 * self.RTCTCR @ V

        return (res1 + res2 + res3).reshape(self.T * self.K)


class MatrixFactorizationWeighted(MatrixFactorization):
    """Instead solves the inverse problem
    || (S - U V^T)W^T ||_F^2 + l1 || U ||_F^2 + l2 || V ||_F^2 + l3 || C R V||_F^2
    This class is far slower, around 6-7 times the running time.
    """

    def __init__(self, args):
        super(MatrixFactorizationWeighted, self).__init__(args)
        self.W = args["W"]

    def solve1(self):
        U = (
            np.linalg.solve(
                self.V.T@self.W.T@self.W@self.V +
                (self.lambda1 / self.lambda0) * np.identity(self.K),
                self.V.T @ self.W.T @ self.W @ self.S.T,
            )
        ).T

        return U

    def solve2(self):

        A = self.W.T@self.W
        B = self.U.T@self.U

        P0 = np.kron(A, B)
        P1 = np.kron(self.lambda2*np.identity(self.T), np.identity(self.K))
        P2 = np.kron(self.lambda3*self.R.T @ self.C.T @
                     self.C @ self.R, np.identity(self.K))

        V = (np.linalg.solve((P0 + P1 + P2), (self.W.T @ self.W @
                                              self.S.T @ self.U).flatten())).reshape(self.T, self.K)

        return V

    def solve_inner(self):
        self.U = self.solve1()
        self.V = self.solve2()
        self.S = self.solve3()
        return
