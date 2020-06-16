import sys
import numpy as np
import tqdm.autonotebook as tqdm

from .model import MFClassifier

class MFClassifierTesting(MFClassifier):
    """An extension of the MatrixFactorization class. Used for testing.
    """

    def f(self, U, V):
        return (
            self.lambda0 * np.sum(np.power((self.X_train - U @ V.T)[self.X_train != 0], 2))
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
        with tqdm.tqdm(total=self.max_iter, file=sys.stdout) as pbar:
            while True:
                converged = next(self)
                pbar.update(1)
                if converged:
                    break

    def set_data(self, X_train):
        """Fit model.

        Prepare model for training procedure.

        Parameters
        ----------
        X_train : array_like, shape (n_samples_train, time_granularity)
            The training set.

        Returns
        -------
        self
            Fitted estimator
        """
        self.X_train = X_train
        self.nonzero_rows, self.nonzero_cols = np.nonzero(self.X_train)
        self.N = self.X_train.shape[0]

        # Initialize U
        self.U_ = np.ones((self.N, self.R))
        self.U_old = np.zeros((self.N, self.R))
        # Initialize V
        self.V_ = np.ones((self.T, self.R)) * np.linspace(self.domain_z[0], self.domain_z[-1], self.R)
        self.V_old = np.zeros((self.T, self.R))
        # Initialize S
        self.Xi_ = self.X_train.copy()

        # Train
        self.n_iter_ = 0
