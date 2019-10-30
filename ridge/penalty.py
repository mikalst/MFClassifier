import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics


class RidgePenalty:
    def __init__(self, kwargs):
        self.lambda0 = kwargs['lambda0']
        self.lambda1 = kwargs['lambda1']
        self.lambda2 = kwargs['lambda2']
        self.lambda3 = kwargs['lambda3']

        self.predict_method = kwargs['predict_method']

        self.X = kwargs['X']
        self.Xtrain = np.copy(self.X)

        if kwargs['predict_method'] == 'naive':
            self.predict_entries = self.select_random_entries(percentage=0.10)
            self.Xtrain[self.predict_entries] = 0.0

        if kwargs['predict_method'] == 'realistic':
            self.predict_window = kwargs['predict_window']
            predict_rows, predict_cols = [], []

            row_candidates = np.random.choice(
                np.arange(self.X.shape[0]), size=int(0.10*self.X.shape[0]), replace=False)
            is_nonzero = self.Xtrain[row_candidates] != 0
            counts_nonzero = np.cumsum(is_nonzero, axis=1)
            # Choose rth nonzero entry in row randomly in the range
            # between 5 and half the number of nonzero entries
            rth_nonzero_predict = np.random.randint(
                5*np.ones(len(row_candidates)),
                np.amax((counts_nonzero[:, -1]//2, 6*np.ones(len(row_candidates))), axis=0)
            )
            # Find column index of rth nonzero entry in row
            col_candidates = np.argmax(counts_nonzero == rth_nonzero_predict[:, None], axis=1)
            for i, row in enumerate(row_candidates):
                if col_candidates[i] - self.predict_window > 5:
                    self.Xtrain[row, col_candidates[i] -
                                self.predict_window:] = 0.0
                    predict_rows.append(row)
                    predict_cols.append(col_candidates[i])

            self.predict_rows = np.array(predict_rows)
            self.predict_cols = np.array(predict_cols)

        print(
            'Assigning {:.2e} / {:.2e} observations to training'.format(
                np.sum(self.Xtrain != 0), np.sum(self.X != 0))
        )

        self.R = kwargs['R']
        self.J = kwargs['J']
        self.kappa = kwargs['kappa']

        # Set nonzero rows and columns automatically after assigning predict X
        self.nonzero_rows_Xtrain, self.nonzero_cols_Xtrain = np.nonzero(
            self.Xtrain)

        self.N = self.X.shape[0]
        self.T = self.X.shape[1]
        self.k = kwargs['k']

        self.V = (np.ones(self.T*self.k) + (1 -
                                            np.random.uniform(size=self.T*self.k))).reshape((self.T, self.k))
        self.U = (np.ones(self.N*self.k) + (1 -
                                            np.random.uniform(size=self.N*self.k))).reshape((self.N, self.k))
        self.S = self.Xtrain.copy()

        self.iteration = 0
        self.total_iterations = kwargs['total_iterations']

    def f(self, U, V):
        return self.lambda0 * np.sum(np.power((self.Xtrain - U@V.T)[self.Xtrain != 0], 2)) + \
            self.lambda1*np.linalg.norm(U)**2 + \
            self.lambda2*np.linalg.norm(V - self.J)**2 + \
            self.lambda3*np.linalg.norm(self.kappa@self.R@V)**2

    def f1(self, u):
        U = u.reshape((self.N, self.k))
        return self.lambda0*np.linalg.norm(self.S - U@self.V.T)**2 + self.lambda1*np.linalg.norm(U)**2

    def g1(self, u):
        U = u.reshape((self.N, self.k))
        return (-2*self.lambda0*(self.S - U@self.V.T)@self.V + 2*self.lambda1*U).reshape(self.N*self.k)

    def solve1(self):
        U = (np.linalg.solve(self.V.T@self.V+(self.lambda1/self.lambda0)
                             * np.identity(self.k), self.V.T@self.S.T)).T

        return U

    def f2(self, v):
        V = v.reshape((self.T, self.k))

        res1 = self.lambda0*np.linalg.norm(self.S - self.U@V.T)**2
        res2 = self.lambda2*np.linalg.norm(V - self.J)**2
        res3 = self.lambda3*np.linalg.norm((self.kappa@self.R)@V)**2

        return res1 + res2 + res3

    def g2(self, v):
        V = v.reshape((self.T, self.k))

        res1 = -2*self.lambda0*(self.S.T - V@self.U.T)@self.U
        res2 = 2*self.lambda2*(V - self.J)

        kappaR = self.kappa@self.R
        res3 = 2*self.lambda3*(kappaR).T@((kappaR)@V)

        return (res1 + res2 + res3).reshape(self.T*self.k)

    def solve2(self):
        L1, Q1 = np.linalg.eigh(
            self.U.T@self.U+(self.lambda2/self.lambda0)*np.identity(self.k))

        kappaR = self.kappa@self.R

        L2, Q2 = np.linalg.eigh(
            (self.lambda3/self.lambda0)*(kappaR).T@(kappaR))

        # For efficiency purposes, these need to be evaluated in order
        hatV = (Q2.T@(self.S.T@self.U+(self.lambda2/self.lambda0)
                      * self.J))@Q1 / np.add.outer(L2, L1)

        V = Q2@(hatV@Q1.T)

        return V

    def solve3(self):
        S = self.U@self.V.T
        S[self.nonzero_rows_Xtrain,
            self.nonzero_cols_Xtrain] = self.Xtrain[self.nonzero_rows_Xtrain, self.nonzero_cols_Xtrain]

        return S

    def solve_inner(self):
        self.U = self.solve1()
        self.V = self.solve2()
        self.S = self.solve3()
        return

    def __iter__(self):
        return self

    def __next__(self):
        self.solve_inner()
        self.iteration += 1

        return self.iteration

    def select_random_entries(self, percentage):
        nonzero_rows, nonzero_cols = np.nonzero(self.X)
        random_indices = np.random.choice(
            np.arange(len(nonzero_rows)),
            size=int(percentage*len(nonzero_rows))
        )
        return nonzero_rows[random_indices], nonzero_cols[random_indices]

    def confusion_matrix(self):
        if self.predict_method == 'naive':
            true = self.X[self.predict_entries]
            pred = np.round((self.U@self.V.T)[self.predict_entries])
            pred[pred < 1] = 1
            pred[pred > 4] = 4
            return sklearn.metrics.confusion_matrix(true, pred)

        elif self.predict_method == 'realistic':
            true = self.X[self.predict_rows, self.predict_cols]
            pred = np.round((self.U@self.V.T)[
                            self.predict_rows, self.predict_cols])
            pred[pred < 1] = 1
            pred[pred > 4] = 4
            return sklearn.metrics.confusion_matrix(true, pred)


def plot_ridge(ridge):
    fig = plt.figure(figsize=(12, 12))

    # Plot U, V, S and X
    plt.subplot2grid((2, 6), (0, 0), colspan=2)
    plt.title("$U$")
    plt.imshow(ridge.U, aspect='auto')
    plt.colorbar(orientation='horizontal', fraction=0.10, pad=0.10)

    plt.subplot2grid((2, 6), (0, 2), colspan=2)
    plt.title("$V$")
    plt.imshow(ridge.V, aspect='auto')
    plt.colorbar(orientation='horizontal', fraction=0.10, pad=0.10)

    plt.subplot2grid((2, 6), (0, 4), colspan=2)
    plt.title("$V$")
    for j in range(ridge.V.shape[1]):
        plt.plot(ridge.V[:, j])

    ax1 = plt.subplot2grid((2, 6), (1, 0), colspan=3)
    plt.title("$S$")
    plt.imshow(ridge.S, aspect='auto')
    plt.colorbar(orientation='horizontal', fraction=0.10, pad=0.10)

    plt.subplot2grid((2, 6), (1, 3), colspan=3, sharey=ax1)
    plt.title("$X$")
    plt.imshow(ridge.X, aspect='auto')
    plt.colorbar(orientation='horizontal', fraction=0.10, pad=0.10)

    fig.tight_layout()
    plt.show()

    np.random.seed(12)
    # Plot predictions for women with cancer
    fig = plt.figure(figsize=(12, 20))

    X_approx = ridge.U@ridge.V.T
    X_approx_int = np.round(X_approx)
    indices_with_risk = np.argwhere(np.any(ridge.X > 3, axis=1)).flatten()
    indices = np.random.choice(indices_with_risk, size=32, replace=False)

    for number, i in enumerate(indices):
        plt.subplot(8, 4, number+1)
        plt.plot(X_approx[i, :])
        plt.plot(X_approx_int[i, :], linestyle='--')

        # Plot predicted values in training set
        x_i_train = (ridge.Xtrain[i, :])
        plt.scatter(np.argwhere(x_i_train != 0),
                    x_i_train[x_i_train != 0], color='k', marker='x')

        # Plot predicted values not in training set
        x_i = (ridge.X[i, :])
        is_predicted = (x_i - x_i_train > 0)
        plt.scatter(np.argwhere(is_predicted),
                    x_i[is_predicted], color='g', marker='x')

    fig.tight_layout()
    plt.show()
