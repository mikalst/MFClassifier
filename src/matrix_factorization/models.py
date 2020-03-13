import numpy as np
import sklearn.metrics
import sys
import tqdm.autonotebook as tqdm

sys.path.append('../')

from src.utils.special_matrices import finite_difference_matrix
from src.data import TemporalDatasetPredict, TemporalDatasetKFold


class MatrixFactorization:
    r"""A class for solving the matrix completion problem using a regularized
    Frobenius norm approach. This yields (U, V) that minimize the cost

    || S - U V^T ||_F^2 + l1 || U ||_F^2 + l2 || V ||_F^2 + l3 || C R V||_F^2

    This yields a K-rank approximation of the input data.
    """

    def __init__(
        self,
        lambda0=1.,
        lambda1=0.,
        lambda2=0.,
        lambda3=0.,
        K=5,
        theta=2.5,
        domain_z=np.arange(1, 10),
        T=100,
        R=None,
        J=None,
        C=None,
        total_iterations=1000,
        tolerance=1e-4
    ):
        # Regularization parameters
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.K = K  # Rank
        self.domain_z = domain_z  # Domain of integer values
        self.theta = theta  # Parameter in the gaussian kernel
        self.T = T  # Time granularity

        if (R is None):
            self.R = finite_difference_matrix(T)

        if (J is None):
            self.J = np.zeros((T, K))

        if (C is None):
            self.C = np.identity(T)

        self.iteration = 0
        self.total_iterations = total_iterations
        self.tolerance = tolerance

        # Code optimization: static variables are computed and stored
        self.RTCTCR = (self.C @ self.R).T@(self.C@self.R)
        self.L2, self.Q2 = np.linalg.eigh(
            (self.lambda3 / self.lambda0) * self.RTCTCR)

    def fit(self, Y):
        self.Y = Y

        self.nonzero_rows, self.nonzero_cols = np.nonzero(self.Y)
        self.N = self.Y.shape[0]

        # Initialize U
        self.U = np.ones((self.N, self.K))
        self.U_old = np.zeros((self.N, self.K))
        # Initialize V
        self.V = np.ones((self.T, self.K)) * \
            np.mean(self.Y[self.nonzero_rows, self.nonzero_cols])
        self.V_old = np.zeros((self.T, self.K))
        # Initialize S
        self.S = self.Y.copy()

        self.iteration = 0

        # __train
        self.__train()

    def resetV(self):
        self.V = np.ones((self.T, self.K)) * \
            np.mean(self.Y[self.nonzero_rows, self.nonzero_cols])
        self.V_old = np.zeros((self.T, self.K))

    def _solve1(self):
        U = (
            np.linalg.solve(
                self.V.T @ self.V +
                (self.lambda1 / self.lambda0) * np.identity(self.K),
                self.V.T @ self.S.T,
            )
        ).T

        return U

    def _solve2(self):
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

    def _solve3(self):
        S = self.U @ self.V.T
        S[self.nonzero_rows, self.nonzero_cols] = self.Y[
            self.nonzero_rows, self.nonzero_cols
        ]

        return S

    def _solve_inner(self):
        self.U = self._solve1()
        self.V = self._solve2()
        self.S = self._solve3()
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration % 50 == 0:
            self.U_old = np.copy(self.U)
            self.V_old = np.copy(self.V)
            self._solve_inner()
            self.iteration += 1
            if (
                np.linalg.norm(self.U_old @ self.V_old.T - self.U @ self.V.T)
                / np.linalg.norm(self.U @ self.V.T)
                < self.tolerance
            ) or self.iteration > self.total_iterations:
                return True
            return False
        self._solve_inner()
        self.iteration += 1
        return False

    def __train(self):
        while True:
            converged = next(self)
            if converged:
                break

    def _loglikelihood(self, X_pred):
        r"""For each row y in X_pred, calculate the loglikelihood of y having originated
        from the reconstructed continuous profile of all the patients in the training set.
        """
        M_train = self.U @ self.V.T

        N_1 = M_train.shape[0]
        N_2 = X_pred.shape[0]

        logL = np.ones((N_2, N_1))

        for i in range(N_2):
            row_nonzero_cols = X_pred[i] != 0
            eta_i = (X_pred[i, row_nonzero_cols])[None, :] - M_train[
                :, row_nonzero_cols
            ]
            logL[i] = np.sum(-self.theta*np.power(eta_i, 2), axis=1)

        return logL

    def predict_proba(self, X_pred, t):
        r"""For each row y in X_pred, calculate the predict_proba probability
        of each integer value in the output domain for a future
        time t.
        """
        logL = self._loglikelihood(X_pred)
        trainM = self.U @ self.V.T

        p_z = np.empty((X_pred.shape[0], self.domain_z.shape[0]))

        for i in range(X_pred.shape[0]):
            p_z[i] = np.exp(logL[i]) @ np.exp(-self.theta *
                                              (trainM[:, t[i], None] - self.domain_z)**2)

        # Normalize
        p_z_normalized = p_z / (np.sum(p_z, axis=1))[:, None]

        return p_z_normalized

    def predict_proba_event(self, X_pred, t, rule_z_to_e, domain_e, p_z_precomputed=None):
        r"""For each row y in X_pred, calculate the predict_proba probability
        of each rule outcome e by mapping rule over the integer values in the
        domain and summing over all integer that result in rule outcome e.
        """
        if p_z_precomputed is None:
            p_z = self.predict_proba(X_pred, t)
        else:
            p_z = p_z_precomputed

        p_e = np.empty((X_pred.shape[0], domain_e.shape[0]))

        for i_event, e in enumerate(domain_e):

            values_of_z_where_e_happens = np.argwhere(
                [rule_z_to_e(z) for z in self.domain_z] == e)

            p_e[:, i_event] = (
                np.sum(p_z[:, values_of_z_where_e_happens], axis=1)).flatten()

        return p_e

    def predict(self, X_pred, t, bias_z=None, p_z_precomputed=None):
        r"""For each row y in X_pred, calculate the highest predict_proba
        probability integer value for a future time t.
        """
        if p_z_precomputed is None:
            p_z = self.predict_proba(X_pred, t)
        else:
            p_z = p_z_precomputed

        if bias_z is None:
            return self.domain_z[np.argmax(p_z, axis=1)]
        else:
            return self.domain_z[np.argmax(p_z*bias_z, axis=1)]

    def predict_event(self, X_pred, t, rule_z_to_e, domain_e, p_z_precomputed=None, bias_e=None, p_e_precomputed=None):
        r"""For each row y in X_pred, calculate the highest predict_proba
        probability rule outcome e for a future time t.
        """
        if p_e_precomputed is None:
            p_e = self.predict_proba_event(
                X_pred, t, rule_z_to_e, domain_e, p_z_precomputed
            )
        else:
            p_e = p_e_precomputed

        if bias_e is None:
            return domain_e[np.argmax(p_e, axis=1)]
        else:
            return domain_e[np.argmax(p_e*bias_e, axis=1)]

    def _score_single(self, data_obj, output_obj, idx_output):
        # Store input parameters
        output_obj['lambda0'][idx_output] = self.lambda0
        output_obj['lambda1'][idx_output] = self.lambda1
        output_obj['lambda2'][idx_output] = self.lambda2
        output_obj['lambda3'][idx_output] = self.lambda3
        output_obj['K'][idx_output] = self.K
        output_obj['theta'][idx_output] = self.theta

        # Store scoring measures
        if not(data_obj.ground_truth is None):
            output_obj['recMSE'][idx_output] = np.mean(
                ((data_obj.ground_truth_train -
                  self.U@(self.V.T))[data_obj.X_train == 0])**2
            )

        output_obj['predSSE'][idx_output] = np.mean(
            ((data_obj.X_train - self.U@self.V.T)[data_obj.X_train != 0])**2
        )

        posterior_probability = self.predict_proba(
            data_obj.X_pred_regressor,
            data_obj.time_of_prediction,
        )

        N_STEPS_BIAS = output_obj.attrs['N_STEPS_BIAS']
        for i_bias, b in enumerate(np.linspace(0, 1, N_STEPS_BIAS)):

            predicted_e = self.predict_event(
                data_obj.X_pred_regressor,
                data_obj.time_of_prediction,
                p_z_precomputed=posterior_probability,
                rule_z_to_e=lambda x: 0 if x == 1 else 1,
                domain_e=np.arange(0, 2),
                bias_e=np.array([1-b, b])
            )

            cm = sklearn.metrics.confusion_matrix(
                data_obj.y_pred > 1, predicted_e)
            output_obj['sensitivity_with_bias'][N_STEPS_BIAS *
                                                idx_output + i_bias] = cm[1, 1] / np.sum(cm[1, :])
            output_obj['specificity_with_bias'][N_STEPS_BIAS *
                                                idx_output + i_bias] = cm[0, 0] / np.sum(cm[0, :])

        # Classification full
        predicted_integers = self.predict(
            data_obj.X_pred_regressor,
            data_obj.time_of_prediction,
            p_z_precomputed=posterior_probability
        )

        N_Z = output_obj.attrs['N_Z']
        output_obj['cms'][(N_Z**2)*idx_output:(N_Z**2)*idx_output+(N_Z**2)] = (sklearn.metrics.confusion_matrix(
            data_obj.y_pred, predicted_integers, labels=range(1, N_Z+1))).flatten()

    def score(self, data_obj, output_obj, idx_output=0):

        if isinstance(data_obj, TemporalDatasetKFold):
            N_FOLDS = data_obj.n_splits
            for i_fold in range(N_FOLDS):
                data_obj.i_fold = i_fold
                self.fit(data_obj.X_train)
                self._score_single(
                    data_obj,
                    output_obj,
                    N_FOLDS*idx_output+i_fold
                )
        elif isinstance(data_obj, TemporalDatasetPredict):
            self._score_single(
                data_obj.X,
                output_obj,
                idx_output
            )


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

    def __train(self):
        with tqdm.tqdm(total=self.total_iterations, file=sys.stdout) as pbar:
            while True:
                converged = next(self)
                pbar.update(1)
                if converged:
                    break


class MatrixFactorizationWeighted(MatrixFactorization):
    """Instead solves the inverse problem
    || (S - U V^T)W^T ||_F^2 + l1 || U ||_F^2 + l2 || V ||_F^2 + l3 || C R V||_F^2
    This class is far slower, around 6-7 times the running time.
    """

    def __init__(self, args):
        super(MatrixFactorizationWeighted, self).__init__(args)
        self.W = args["W"]

    def _solve1(self):
        U = (
            np.linalg.solve(
                self.V.T@self.W.T@self.W@self.V +
                (self.lambda1 / self.lambda0) * np.identity(self.K),
                self.V.T @ self.W.T @ self.W @ self.S.T,
            )
        ).T

        return U

    def _solve2(self):

        A = self.W.T@self.W
        B = self.U.T@self.U

        P0 = np.kron(A, B)
        P1 = np.kron(self.lambda2*np.identity(self.T), np.identity(self.K))
        P2 = np.kron(self.lambda3*self.R.T @ self.C.T @
                     self.C @ self.R, np.identity(self.K))

        V = (np.linalg.solve((P0 + P1 + P2), (self.W.T @ self.W @
                                              self.S.T @ self.U).flatten())).reshape(self.T, self.K)

        return V

    def _solve_inner(self):
        self.U = self._solve1()
        self.V = self._solve2()
        self.S = self._solve3()
        return
