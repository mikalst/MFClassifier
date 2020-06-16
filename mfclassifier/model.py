import sys
import copy
import numpy as np
import hashlib
import sklearn.metrics
from sklearn.base import ClassifierMixin

from .data import TemporalDatasetPredict, TemporalDatasetKFold
from .utils.special_matrices import finite_difference_matrix


class MFClassifier(ClassifierMixin):
    """
    Matrix Factorization classifier for sparse temporal data.

    MFClassifier fits a matrix factorization model M = U V^T as the latent
    matrix of the training data.

    Parameters
    ----------
    lambda0 : float, default=1.0
        Regularization parameter
    
    lambda1 : float, default=0.0
        Regularization parameter
    
    lambda2 : float, default=0.0
        Regularization parameter
    
    lambda3 : float, default=0.0
        Regularization parameter

    R : int, default=5
        Rank estimate of decomposition

    domain_z : array of shape=(n_classes_z), default=np.arange(1, 10),
        Allowed integer classes.

    z_to_binary_mapping : map, default=None
        Mapping from allowed integer classes to True/False.
    
    T : float, default=100
        Number of timesteps.

    D : array of shape (T, T), default=None
        Linear mapping used in regularization term.

    J : array of shape (T, R), default=None
        Offset matrix used in regularization term.

    K : array of shape (T, T), default=None
        Linear mapping used in regularization term.

    max_iter : int, default=100-
        Maximum number of n_iter_s for the solver.

    tol : float, default=1e-4
        Stopping critertion. 


    Attributes
    ----------
    U_ : array of shape (n_samples, R)
        Estimated profile weights for the matrix decomposition problem.
        
    V_ : array of shape (T, R)
        Estimated time profiles for the matrix decomposition problem.

    theta_ : float
        Estimated theta in the Gaussian sampled distribution.

    n_iter_ : int
        Actual number of iterations.
        
    """

    def __init__(
        self,
        lambda0=1.,
        lambda1=1.,
        lambda2=1.,
        lambda3=0.,
        R=5,
        domain_z=np.arange(1, 10),
        z_to_binary_mapping=None,
        T=100,
        D=None,
        J=None,
        K=None,
        max_iter=1000,
        tol=1e-4,
    ):
        # Regularization parameters
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.R = R  # Rank
        self.domain_z = domain_z  # Domain of integer values
        self.z_to_binary_mapping = z_to_binary_mapping  # Mapping to binary classifier
        self.T = T  # Time granularity

        self.D = D
        if (self.D is None):
            self.D = finite_difference_matrix(T)
        self.J = J
        if (self.J is None):
            self.J = np.ones((T, R))   
        self.K = K
        if (self.K is None):
            self.K = np.identity(T)

        self.n_iter_ = 0
        self.max_iter = max_iter
        self.tol = tol

        # Code optimization: static variables are computed and stored
        self.DTKTKV = (self.K @ self.D).T@(self.K@self.D)
        self.L2, self.Q2 = np.linalg.eigh(
            (self.lambda3 / self.lambda0) * self.DTKTKV)

        # Initialize prediction probabilities
        self.__proba_z_precomputed = None
        self.__ds_X_hash = None
        self.__ds_t_hash = None

    def set_params(
        self,
        lambda0=1.,
        lambda1=0.,
        lambda2=0.,
        lambda3=0.,
        R=5,
        domain_z=np.arange(1, 10),
        z_to_binary_mapping=None,
        T=100,
        D=None,
        J=None,
        K=None,
        max_iter=1000,
        tol=1e-4
    ):
        # Regularization parameters
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.R = R  # Rank
        self.domain_z = domain_z  # Domain of integer values
        self.z_to_binary_mapping = z_to_binary_mapping  # Mapping to binary classifier
        self.T = T  # Time granularity

        if (D is None):
            self.D = finite_difference_matrix(T)

        if (J is None):
            self.J = np.zeros((T, R))

        if (K is None):
            self.K = np.identity(T)

        self.n_iter_ = 0
        self.max_iter = max_iter
        self.tol = tol

        # Code optimization: fixed variables are computed and stored
        self.DTKTKV = (self.K @ self.D).T@(self.K@self.D)
        self.L2, self.Q2 = np.linalg.eigh(
            (self.lambda3 / self.lambda0) * self.DTKTKV)

    def get_params(self, deep=True):
        # Regularization parameters
        params = {
            'lambda0': self.lambda0,
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'lambda3': self.lambda3,
            'R': self.R,
            'theta': self.theta,
            'domain_z': self.domain_z,
            'z_to_e_mapping': self.z_to_e_mapping,
            'domain_e': self.domain_e,
            'T': self.T,
            'D': self.D,
            'J': self.J,
            'K': self.K,
            'max_iter': self.max_iter,
            'tol': self.tol
        }
        if deep:
            for key in params.keys():
                params[key] = copy.deepcopy(params[key])

        return params

    def _solve1(self):
        U = (
            np.linalg.solve(
                self.V_.T @ self.V_ +
                (self.lambda1 / self.lambda0) * np.identity(self.R),
                self.V_.T @ self.Xi_.T,
            )
        ).T

        return U

    def _solve2(self):
        L1, Q1 = np.linalg.eigh(
            self.U_.T @ self.U_ +
            (self.lambda2 / self.lambda0) * np.identity(self.R)
        )

        # For efficiency purposes, these need to be evaluated in order
        hatV = (
            (self.Q2.T @ (self.Xi_.T @ self.U_ + (self.lambda2 / self.lambda0) * self.J))
            @ Q1
            / np.add.outer(self.L2, L1)
        )
        V = self.Q2 @ (hatV @ Q1.T)

        return V

    def _solve3(self):
        Xi = self.U_ @ self.V_.T
        Xi[self.nonzero_rows, self.nonzero_cols] = self.X_train[
            self.nonzero_rows, self.nonzero_cols
        ]

        return Xi

    def _solve_inner(self):
        self.U_ = self._solve1()
        self.V_ = self._solve2()
        self.Xi_ = self._solve3()
        return

    def __next__(self):
        if self.n_iter_ % 50 == 0:
            self.U_old = np.copy(self.U_)
            self.V_old = np.copy(self.V_)
            self._solve_inner()
            self.n_iter_ += 1
            if (
                np.linalg.norm(self.U_old @ self.V_old.T - self.U_ @ self.V_.T)
                / np.linalg.norm(self.U_ @ self.V_.T)
                < self.tol
            ) or self.n_iter_ > self.max_iter:
                return True
            return False
        self._solve_inner()
        self.n_iter_ += 1
        return False

    def _fit(self):
        while True:
            converged = next(self)
            if converged:
                break

    def fit(self, X_train):
        """Fit model.

        Fits model using X_train as input

        Parameters
        ----------
        X_train : array_like, shape (n_samples_train, n_timesteps)
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
        self._fit()

        # Estimate theta
        self.theta = np.sqrt(np.mean(((self.X_train - self.U_@self.V_.T)[self.X_train > 0])**2))

    def _loglikelihood(self, X):
        """Compute loglikelihood of X having originated from 
        the fitted profiles (U V^T).

        For all x_i in X, compute the log of the estimated
        likelihood that x_i originated from m_j for j = 1, ..., N
        where N = n_samples_train is the number of samples used in
        training the model. 

        Parameters
        ----------
        X : array_like, shape (n_samples, n_timesteps)
            The regressor set.

        Returns
        -------
        logL : array_like, shape (n_samples, n_samples_train)
            The logs of the estimated likelihoods.
        """
        M_train = self.U_ @ self.V_.T

        N_1 = M_train.shape[0]
        N_2 = X.shape[0]

        logL = np.ones((N_2, N_1))

        for i in range(N_2):
            row_nonzero_cols = X[i] != 0
            eta_i = (X[i, row_nonzero_cols])[None, :] - M_train[
                :, row_nonzero_cols
            ]
            logL[i] = np.sum(-self.theta*np.power(eta_i, 2), axis=1)

        return logL

    def __is_match_ds_hash(self, X, t):
        """Check if hash of (X, t) matches stored

        Checks if the stored hexadecimal hash
        matches the hexademical hash of the input 
        (X, t). 

        Parameters
        ----------
        X : array_like, shape (n_samples, n_timesteps)
            The regressor set.

        t : array_like, shape (n_samples, )
            Time of prediction.

        Returns
        -------
        match : bool
            True if match
        """
        if (self.__ds_X_hash is None) or (self.__ds_t_hash is None):
            return False
        elif (hashlib.sha1(X).hexdigest() == self.__ds_X_hash) and (hashlib.sha1(t).hexdigest() == self.__ds_t_hash):
            return True
        return False

    def __store_ds_hash(self, X, t):
        """Store hash of dataset.

        Stores a hexadecimal hash of the dataset X used
        in predict_proba.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_timesteps)
            The regressor set.

        t : array_like, shape (n_samples, )
            Time of prediction.

        Returns
        -------
        self
            Model with stored hash
        """
        self.__ds_X_hash = hashlib.sha1(X).hexdigest()
        self.__ds_t_hash = hashlib.sha1(t).hexdigest()
        return self

    def predict_proba(self, X, t):
        """Compute class probabilities.

        For all (x_i, t_i) in (X, t), compute the estimated
        probability that row i will at time t be in the state
        z for z in the domain_z of the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_timesteps)
            The regressor set.

        t : array_like, shape (n_samples, )
            Time of prediction.

        Returns
        -------
        proba_z_normalized
            The probalities
        """
        # If evaluating several scoring methods subsequently,
        #  significant computational time can be saved by storing
        #  the class probabilities
        if self.__is_match_ds_hash(X, t):
            return self.__proba_z_precomputed

        X, t = X, t

        logL = self._loglikelihood(X)
        trainM = self.U_ @ self.V_.T

        proba_z = np.empty((X.shape[0], self.domain_z.shape[0]))

        for i in range(X.shape[0]):
            proba_z[i] = np.exp(logL[i]) @ np.exp(-self.theta *
                                                  (trainM[:, t[i], None] - self.domain_z)**2)

        # Normalize
        proba_z_normalized = proba_z / (np.sum(proba_z, axis=1))[:, None]

        # Store probabilities
        self.__proba_z_precomputed = proba_z_normalized
        self.__store_ds_hash(X, t)

        return proba_z_normalized

    def predict_proba_binary(self, X, t):
        """Compute binary probabilities.

        For all (x_i, t_i) in (X, t), compute the estimated
        probability that row i will at time t be True.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_timesteps)
            The regressor set.

        t : array_like, shape (n_samples, )
            Time of prediction.

        Returns
        -------
        proba_bin
            The probalities.
        """
        # If evaluating several scoring methods subsequently,
        #  significant computational time can be saved by storing
        #  the class probabilities
        proba_z = self.predict_proba(X, t)

        values_of_z_where_true = [
            self.z_to_binary_mapping(z) for z in self.domain_z]
        proba_bin = np.sum(
            proba_z[:, values_of_z_where_true], axis=1).flatten()

        return proba_bin

    def predict(self, X, t, bias_z=None):
        """Predict future state.

        For all (x_i, t_i) in (X, t), predict the most probable
        state z at time t.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_timesteps)
            The regressor set.

        t : array_like, shape (n_samples, )
            Time of prediction.

        bias : array_like, shape (n_states_z, )
            The bias of the model.

        Returns
        -------
        z_states : (n_samples, )
            The predicted states.
        """
        proba_z = self.predict_proba(X, t)

        if bias_z is None:
            return self.domain_z[np.argmax(proba_z, axis=1)]
        else:
            return self.domain_z[np.argmax(proba_z*bias_z, axis=1)]

    def predict_binary(self, X, t, bias_bin=None):
        """Predict future binary outcome.

        For all (x_i, t_i) in (X, t), predict the most probable
        binary outcome at time t.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_timesteps)
            The regressor set.

        t : array_like, shape (n_samples, )
            Time of prediction.

        bias : array_like, shape (n_states_e, )
            The bias of the model.

        Returns
        -------
        bin_states : (n_samples, )
            The predicted states.
        """
        proba_bin = self.predict_proba_binary(X, t)
        if bias_bin is None:
            return np.ones_like(proba_bin)*(proba_bin >= 0.5)
        else:
            return np.ones_like(proba_bin)*(proba_bin >= bias_bin)
