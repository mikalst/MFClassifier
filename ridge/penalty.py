import numpy as np

class RidgePenalty:
    def __init__(self, kwargs):
        self.lambda0 = kwargs['lambda0']
        self.lambda1 = kwargs['lambda1']
        self.lambda2 = kwargs['lambda2']
        self.lambda3 = kwargs['lambda3']

        self.X = kwargs['X']
        self.R = kwargs['R']
        self.J = kwargs['J']
        self.kappa = kwargs['kappa']

        self.nonzero_rows = kwargs['nonzero_rows']
        self.nonzero_cols = kwargs['nonzero_cols']
        
        self.N = self.X.shape[0]
        self.T = self.X.shape[1]
        self.k = kwargs['k']
        
        self.V = (np.ones(self.T*self.k) + (1 - np.random.uniform(size = self.T*self.k))).reshape((self.T, self.k))
        self.U = (np.ones(self.N*self.k) + (1 - np.random.uniform(size = self.N*self.k))).reshape((self.N, self.k))
        self.S = self.X.copy()
        
        self.iteration = 0
        self.total_iterations = kwargs['total_iterations']
    
    def f(self, U, V):
        return self.lambda0 * np.sum(np.power((self.X - U@V.T)[self.X != 0], 2)) + \
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
        U = (np.linalg.solve(self.V.T@self.V+(self.lambda1/self.lambda0)*np.identity(self.k), self.V.T@self.S.T)).T
        
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
        L1, Q1 = np.linalg.eigh(self.U.T@self.U+(self.lambda2/self.lambda0)*np.identity(self.k))

        kappaR = self.kappa@self.R

        L2, Q2 = np.linalg.eigh((self.lambda3/self.lambda0)*(kappaR).T@(kappaR))

        # For efficiency purposes, these need to be evaluated in order
        hatV = (Q2.T@(self.S.T@self.U+(self.lambda2/self.lambda0)*self.J))@Q1 / np.add.outer(L2, L1)

        V = Q2@(hatV@Q1.T)

        return V
    
    def solve3(self):        
        S = self.U@self.V.T
        S[self.nonzero_rows, self.nonzero_cols] = self.X[self.nonzero_rows, self.nonzero_cols]

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
        