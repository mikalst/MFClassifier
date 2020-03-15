import sys
import copy
import numpy as np
import hashlib
import sklearn.metrics
from sklearn.base import ClassifierMixin
import tqdm.autonotebook as tqdm

from ..data import TemporalDatasetPredict, TemporalDatasetKFold
from ..utils.special_matrices import finite_difference_matrix


class TemporalMCClassifier(ClassifierMixin):
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
        z_to_event_mapping=None,
        domain_event=None,
        z_to_binary_mapping=None,
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
        self.theta = theta  # Parameter in the gaussian kernel
        self.domain_z = domain_z  # Domain of integer values

        self.z_to_event_mapping = z_to_event_mapping # Mapping to secondary classifier
        self.domain_event = domain_event # Domain of secondary classifier

        self.z_to_binary_mapping = z_to_binary_mapping

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

        # Initialize prediction probabilities
        self.__proba_z_precomputed = None
        self.__ds_X_hash = None
        self.__ds_t_hash = None

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

    def set_params(
        self,
        lambda0=1.,
        lambda1=0.,
        lambda2=0.,
        lambda3=0.,
        K=5,
        theta=2.5,
        domain_z=np.arange(1, 10),
        z_to_e_mapping=None,
        domain_e=None,
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
        self.theta = theta  # Parameter in the gaussian kernel
        self.domain_z = domain_z  # Domain of integer values

        self.z_to_e_mapping = z_to_e_mapping # Mapping to secondary classifier
        self.domain_e = domain_e # Domain of secondary classifier

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

    def get_params(self, deep=True):
        # Regularization parameters
        params = {
            'lambda0': self.lambda0,
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'lambda3': self.lambda3,
            'K': self.K,
            'theta': self.theta,
            'domain_z': self.domain_z,
            'z_to_e_mapping': self.z_to_e_mapping,
            'domain_e': self.domain_e,
            'T': self.T,
            'R': self.R,
            'J': self.J,
            'C': self.C,
            'total_iterations': self.total_iterations,
            'tolerance': self.tolerance
        }
        if deep:
            for key in params.keys():
                params[key] = copy.deepcopy(params[key])

        return params

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

    def __is_match_ds_hash(self, X_and_t):
        X, t = X_and_t

        if (self.__ds_X_hash is None) or (self.__ds_t_hash is None):
            return False
        elif (hashlib.sha1(X).hexdigest() == self.__ds_X_hash) and (hashlib.sha1(t).hexdigest() == self.__ds_t_hash):
            return True
        return False

    def __store_ds_hash(self, X_and_t):
        X, t = X_and_t
        self.__ds_X_hash = hashlib.sha1(X).hexdigest()
        self.__ds_t_hash = hashlib.sha1(t).hexdigest()

    def predict_proba(self, X_and_t):
        r"""For each row y in X_pred, calculate the predict_proba probability
        of each integer value in the output domain for a future
        time t.
        """
        # If evaluating several scoring methods subsequently, 
        #  significant computational time can be saved by storing
        #  the class probabilities
        if self.__is_match_ds_hash(X_and_t):
            return self.__proba_z_precomputed

        X_pred, t = X_and_t

        logL = self._loglikelihood(X_pred)
        trainM = self.U @ self.V.T

        proba_z = np.empty((X_pred.shape[0], self.domain_z.shape[0]))

        for i in range(X_pred.shape[0]):
            proba_z[i] = np.exp(logL[i]) @ np.exp(-self.theta *
                                              (trainM[:, t[i], None] - self.domain_z)**2)

        # Normalize
        proba_z_normalized = proba_z / (np.sum(proba_z, axis=1))[:, None]

        # Store probabilities
        self.__proba_z_precomputed = proba_z_normalized
        self.__store_ds_hash(X_and_t)

        return proba_z_normalized

    def predict_proba_event(self, X_and_t):
        r"""For each row y in X_pred, calculate the predict_proba probability
        of each rule outcome e by mapping rule over the integer values in the
        domain and summing over all integer that result in rule outcome e.
        """
        proba_z = self.predict_proba(X_and_t)

        proba_event = np.empty((proba_z.shape[0], self.domain_e.shape[0]))

        for i_event, e in enumerate(self.domain_e):

            values_of_z_where_e_happens = np.argwhere(
                [self.z_to_e_mapping(z) for z in self.domain_z] == e)

            proba_event[:, i_event] = (
                np.sum(proba_z[:, values_of_z_where_e_happens], axis=1)).flatten()

        return proba_event

    def predict_proba_binary(self, X_and_t):
        # If evaluating several scoring methods subsequently, 
        #  significant computational time can be saved by storing
        #  the class probabilities
        proba_z = self.predict_proba(X_and_t)

        values_of_z_where_true = [self.z_to_binary_mapping(z) for z in self.domain_z] 
        proba_bin = np.sum(proba_z[:, values_of_z_where_true], axis=1).flatten()

        return proba_bin

    def predict(self, X_and_t, bias_z=None):
        r"""For each row y in X_pred, calculate the highest predict_proba
        probability integer value for a future time t.
        """
        proba_z = self.predict_proba(X_and_t)

        if bias_z is None:
            return self.domain_z[np.argmax(proba_z, axis=1)]
        else:
            return self.domain_z[np.argmax(proba_z*bias_z, axis=1)]

    def predict_event(self, X_and_t, bias_e=None):
        r"""For each row y in X_pred, calculate the highest predict_proba
        probability rule outcome e for a future time t.
        """
        proba_e = self.predict_proba_event(X_and_t)

        if bias_e is None:
            return self.domain_e[np.argmax(proba_e, axis=1)]
        else:
            return self.domain_e[np.argmax(proba_e*bias_e, axis=1)]

    def predict_binary(self, X_and_t, bias_bin=None):
        r"""For each row y in X_pred, calculate the highest probability
        binary outcome for a future time t.
        """
        proba_bin = self.predict_proba_binary(X_and_t)
        if bias_bin is None:
            return np.ones_like(proba_bin)*(proba_bin >= 0.5)
        else:
            return np.ones_like(proba_bin)*(proba_bin >= 1 - bias_bin)


class TemporalMClassifierTesting(TemporalMClassifier):
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