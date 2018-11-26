import pdb, time
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import lu
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm, cholesky

"""
Author  : Alex Bujan (adapted from http://www.stanford.edu/~boyd)
Date    : 12/06/2015
"""


def lasso_admm(X, y, alpha, rho=1., rel_par=1., QUIET=True, \
               MAX_ITER=10, ABSTOL=1e-3, RELTOL=1e-2):
    """
     Solve lasso problem via ADMM

     [z, history] = lasso_admm(X,y,alpha,rho,rel_par)

     Solves the following problem via ADMM:

       minimize 1/2*|| Ax - y ||_2^2 + alpha || x ||_1

     The solution is returned in the vector z.

     history is a dictionary containing the objective value, the primal and
     dual residual norms, and the tolerances for the primal and dual residual
     norms at each iteration.

     rho is the augmented Lagrangian parameter.

     rel_par is the over-relaxation parameter (typical values for rel_par are
     between 1.0 and 1.8).

     More information can be found in the paper linked at:
     http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    """

    def objective(X, y, alpha, x, z):
        return .5 * np.square(X.dot(x) - y).sum() + alpha * norm(z, 1)

    def shrinkage(x, kappa):
        return np.maximum(0., x - kappa) - np.maximum(0., -x - kappa)

    def factor(X, rho):
        m, n = X.shape
        if m >= n:
            _, L, _ = lu(X.T.dot(X) + rho * sparse.eye(n))
        else:
            _, L, _ = lu(sparse.eye(m) + 1. / rho * (X.dot(X.T)))
        L = sparse.csc_matrix(L)
        U = sparse.csc_matrix(L.T)

        return L, U

    # Data preprocessing
    m, n = X.shape
    # save a matrix-vector multiply
    Xty = X.T.dot(y)

    # ADMM solver
    x = np.zeros((n, 1))
    z = np.zeros((n, 1))
    u = np.zeros((n, 1))

    # cache the (Cholesky) factorization
    L, U = factor(X, rho)

    '''
    # Saving state
    h = {}
    h['objval'] = np.zeros(MAX_ITER)
    h['r_norm'] = np.zeros(MAX_ITER)
    h['s_norm'] = np.zeros(MAX_ITER)
    h['eps_pri'] = np.zeros(MAX_ITER)
    h['eps_dual'] = np.zeros(MAX_ITER)
    '''

    for k in range(MAX_ITER):

        print("Iteration: {iter}".format(iter=k))

        # x-update
        q = Xty + rho * (z - u)  # (temporary value)
        if m >= n:
            x = spsolve(U, spsolve(L, q))[..., np.newaxis]
        else:
            ULXq = spsolve(U, spsolve(L, X.dot(q)))[..., np.newaxis]
            x = (q * 1. / rho) - ((X.T.dot(ULXq)) * 1. / (rho ** 2))

        # z-update with relaxation
        zold = np.copy(z)
        x_hat = rel_par * x + (1. - rel_par) * zold
        z = shrinkage(x_hat + u, alpha * 1. / rho)

        # u-update
        u = u + (x_hat - z)

        '''
        # diagnostics, reporting, termination checks
        h['objval'][k] = 0#objective(X, y, alpha, x, z)
        h['r_norm'][k] = norm(x - z)
        h['s_norm'][k] = norm(-rho * (z - zold))
        h['eps_pri'][k] = np.sqrt(n) * ABSTOL + \
                          RELTOL * np.maximum(norm(x), norm(-z))
        h['eps_dual'][k] = np.sqrt(n) * ABSTOL + \
                           RELTOL * norm(rho * u)
        '''

    return z.ravel()
