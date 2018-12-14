"""
Blind Source Separation using Tweaked Independent Vector Analysis with Auxiliary Function

2018 (c) Yaron Dibner & Virgile Hernicot, MIT License
"""
import numpy as np

from pyroomacoustics import stft, istft
from .common import projection_back
from scipy.linalg import dft

# A few contrast functions
f_contrasts = {
    'norm': {'f': (lambda r, c, m: c * r), 'df': (lambda r, c, m: c)},
    'cosh': {'f': (lambda r, c, m: m * np.log(np.cosh(c * r))), 'df': (lambda r, c, m: c * m * np.tanh(c * r))}
}


def demix(Y, X, S, W):
    freq = S.shape[0]
    for f in range(freq):
        Y[:, S[f], :] = np.dot(X[:, S[f], :], np.conj(W[S[f], :, :]))


def sparseauxiva(X, S, n_iter, proj_back=True, return_filters=False, lasso=True):
    n_frames, n_freq, n_chan = X.shape

    k_freq = S.shape[0]

    # default to determined case
    n_src = n_chan

    # initialize the demixing matrices
    W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)

    f_contrast = f_contrasts['norm']
    f_contrast_args = [1, 1]

    I = np.eye(n_src, n_src)
    Y = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)
    V = np.zeros((n_freq, n_src, n_chan, n_chan), dtype=X.dtype)
    r = np.zeros((n_frames, n_src))
    G_r = np.zeros((n_frames, n_src))

    print("Init done, proceeding to sparse AuxIVA...")

    for epoch in range(n_iter):

        demix(Y, X, S, W)

        # simple loop as a start
        # shape: (n_frames, n_src)
        r[:, :] = np.sqrt(np.sum(np.abs(Y * np.conj(Y)), axis=1))

        # Apply derivative of contrast function
        G_r[:, :] = f_contrast['df'](r, *f_contrast_args) / r  # shape (n_frames, n_src)

        # Compute Auxiliary Variable
        for f in range(k_freq):
            for s in range(n_src):
                V[S[f], s, :, :] = (np.dot(G_r[None, :, s] * X[:, S[f], :].T, np.conj(X[:, S[f], :]))) / X.shape[0]

        # Update now the demixing matrix
        for f in range(k_freq):
            for s in range(n_src):
                WV = np.dot(np.conj(W[S[f], :, :].T), V[S[f], s, :, :])
                W[S[f], :, s] = np.linalg.solve(WV, I[:, s])
                W[S[f], :, s] /= np.sqrt(np.inner(np.conj(W[S[f], :, s]), np.dot(V[S[f], s, :, :], W[S[f], :, s])))

    print("Successfully computed the sparse weights, proceeding to lasso...")

    np.set_printoptions(precision=2)

    if lasso:
        # Here comes Lassoooooooooo
        Z = np.zeros((n_src, k_freq), dtype=W.dtype)
        G = np.zeros((n_src, n_freq, 1), dtype=Z.dtype)
        hrtf = np.zeros((n_freq, n_src), dtype=W.dtype)  # h in the time domain
        Hrtf = np.zeros((n_freq, n_src), dtype=W.dtype)  # H in the frequency domain
        DFT_matrix = dft(n_freq)
        # print(np.all(np.linalg.eigvals(DFT_matrix.T.dot(DFT_matrix)) > 0))
        for i in range(n_src):

            # sparse relative transfer function
            Z[i, :] = np.array([-W[S[f], 0, i] / W[S[f], 1, i] for f in range(k_freq)]).conj().T

            G[i, S] = (np.expand_dims(Z[i, :], axis=1))

            # compute the extrapolated relative impulse response solving LASSO
            hrtf[:, i] = sparir(G[i, :], S)

            # convert to transfer function
            Hrtf[:, i] = np.dot(DFT_matrix, hrtf[:, i])

            # Finally, you could assemble W
            for f in range(n_freq):
                W[f, :, i] = np.conj([Hrtf[f, i], -1])

    demix(Y, X, np.array(range(n_freq)), W)

    # applying projection_back in the end to solve the scale ambiguity
    if proj_back:
        z = projection_back(Y, X[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, W
    else:
        return Y


def sparir(G, S, delay=0, weights=np.array([]), gini=0):
    L = G.shape[0]  # n_freq

    y = np.concatenate((np.real(G[S]), np.imag(G[S])), axis=0)
    M = y.shape[0]

    if gini == 0:  # if no initialization is given
        g = np.zeros((L, 1))
        g[delay] = 1
    else:
        g = gini

    if weights == 0:
        tau = np.sqrt(L) / (y.conj().T.dot(y))  # * ones(L,1)
        tau = tau * np.exp(0.11 * np.abs((np.arange(1., L + 1.).T - delay)) ** 0.3)
    elif weights.size == 0:
        tau = np.sqrt(L) / (y.conj().T.dot(y))
        tau = tau * np.exp(0.11 * np.abs((np.arange(1., L + 1.).T - delay)) ** 0.3)
        tau = tau.T
    elif weights.shape[0] == 1:
        tau = np.ones((L, 1)) * weights
    else:
        tau = np.tile(weights.T, (1, 1)).reshape(L)

    def soft(x, T):
        if np.sum(np.abs(T).flatten()) == 0:
            u = x
        else:
            u = np.max(np.abs(x) - T, 0)
            u = u / (u + T) * x
        return u

    maxiter = 50
    alphamax = 1e5  # maximum step - length parameter alpha
    alphamin = 1e-7  # minimum step - length parameteralpha
    tol = 10

    aux = np.zeros((L, 1),dtype=complex)
    G = np.fft.fft(g.flatten())
    Ag = np.concatenate((np.real(G[S]), np.imag(G[S])), axis=0)
    r = Ag - y.flatten()  # instead of r = A * g - y
    aux[S] = np.expand_dims(r[0:M // 2] + 1j * r[M // 2:], axis=1)
    gradq = L * np.fft.irfft(aux.flatten(), L)  # instead of gradq = A'*r
    gradq = np.expand_dims(gradq, axis=1)
    alpha = 10
    support = g != 0
    iter_ = 0

    crit = np.zeros((maxiter, 1))

    criterion = -tau[support] * np.sign(g[support]) - gradq[support]
    crit[iter_] = np.sum(criterion ** 2)
    # print("iteration: ", iter_ + 1, ", criterion: ", crit[iter_])

    while (crit[iter_] > tol) and (iter_ < maxiter - 1):
        prev_r = r
        prev_g = g
        g = soft(prev_g - gradq * (1 / alpha), tau / alpha)
        dg = g - prev_g
        DG = np.fft.fft(dg.flatten())
        Adg = np.concatenate((np.real(DG[S]), np.imag(DG[S])), axis=0)
        r = prev_r + Adg.flatten()  # faster than A * g - y
        dd = dg.flatten().conj().T @ dg.flatten()
        dGd = Adg.flatten().conj().T @ Adg.flatten()
        alpha = min(alphamax, max(alphamin, dGd / (np.finfo(np.float32).eps + dd)))
        iter_ = iter_ + 1
        support = g != 0
        aux[S] = np.expand_dims(r[0:M // 2] + 1j * r[M // 2:], axis=1)
        gradq = L * np.fft.irfft(aux.flatten(), L)
        gradq = np.expand_dims(gradq, axis=1)
        criterion = -tau[support] * np.sign(g[support]) - gradq[support]
        crit[iter_] = sum(criterion ** 2) + sum(abs(gradq[~support]) - tau[~support] > tol)
        # if iter_ % 100 == 0:
        # print("iteration: ", iter_+1, ", criterion: ", crit[iter_])

    print('SpaRIR: {0} iterations done.'.format(iter_))

    return g.flatten()
