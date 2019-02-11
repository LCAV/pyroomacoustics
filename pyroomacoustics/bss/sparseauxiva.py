"""
Blind Source Separation for sparsely mixed signals based on Independent Vector Analysis (IVA) with Auxiliary Function

2018 (c) Yaron Dibner, Virgile Hernicot, Juan Azcarreta, MIT License
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


def sparseauxiva(X, S=None, n_src=None, n_iter=30, proj_back=True, W0=None,
                 f_contrast=None, f_contrast_args=[],
                 return_filters=False, callback=False):

    '''
    Implementation of sparse AuxIVA algorithm for BSS presented in

    Janský, Jakub & Koldovský, Zbyněk & Ono, Nobutaka. (2016). A computationally cheaper method for blind speech
    separation based on AuxIVA and incomplete demixing transform. 1-5. IWAENC2016

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components
    S: ndarray ()
        Index set of active frequency bins for sparse AuxIVA
    n_iter: int, optional
        The number of iterations (default 30)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for demixing matrix
    f_contrast: dict of functions
        A dictionary with two elements 'f' and 'df' containing the contrast
        function taking 3 arguments This should be a ufunc acting element-wise
        on any array
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    '''

    n_frames, n_freq, n_chan = X.shape

    if S is None:
        k_freq = n_freq
    else:
        k_freq = S.shape[0]

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    # for now it only supports determined case
    assert n_chan == n_src

    # initialize the demixing matrices
    if W0 is None:
        W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)
    else:
        W = W0.copy()

    if f_contrast is None:
        f_contrast = f_contrasts['norm']
        f_contrast_args = [1, 1]

    I = np.eye(n_src, n_src)
    Y = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)
    r = np.zeros((n_frames, n_src))
    G_r = np.zeros((n_frames, n_src))

    def demixsparse(Y, X, S, W):
        for f in range(k_freq):
            Y[:, S[f], :] = np.dot(X[:, S[f], :], np.conj(W[S[f], :, :]))

    # Conventional AuxIVA in the frequency bins k_freq selected by S
    for epoch in range(n_iter):
        demixsparse(Y, X, S, W)

        #if callback is not None and epoch % 10 == 0:
        #    if proj_back:
        #        z = projection_back(Y, X[:, :, 0])
        #        callback(Y * np.conj(z[None, :, :]))
        #    else:
        #        callback(Y)

        # simple loop as a start
        # shape: (n_frames, n_src)
        r[:, :] = np.sqrt(np.sum(np.abs(Y * np.conj(Y)), axis=1))

        # Apply derivative of contrast function
        G_r[:, :] = f_contrast['df'](r, *f_contrast_args) / r  # shape (n_frames, n_src)

        # Compute Auxiliary Variable
        V = np.mean(
            (X[:, :, None, :, None] * G_r[:, None, :, None, None])
            * np.conj(X[:, :, None, None, :]),
            axis=0,
        )

        # Update now the demixing matrix
        for s in range(n_src):
            W_H = np.conj(np.swapaxes(W, 1, 2))
            WV = np.matmul(W_H, V[:, s, :, :])
            rhs = I[None, :, s][[0] * WV.shape[0], :]
            W[:, :, s] = np.linalg.solve(WV, rhs)

            # normalize
            P1 = np.conj(W[:, :, s])
            P2 = np.sum(V[:, s, :, :] * W[:, None, :, s], axis=-1)
            W[:, :, s] /= np.sqrt(
               np.sum(P1 * P2, axis=1)
               )[:, None]

    np.set_printoptions(precision=2)

    # LASSO regularization to reconstruct the complete Hrtf
    Z = np.zeros((n_src, k_freq), dtype=W.dtype)
    G = np.zeros((n_src, n_freq, 1), dtype=Z.dtype)
    hrtf = np.zeros((n_freq, n_src), dtype=W.dtype)
    Hrtf = np.zeros((n_freq, n_src), dtype=W.dtype)

    for i in range(n_src):

        # sparse relative transfer function
        Z[i, :] = np.array([-W[S[f], 0, i] / W[S[f], 1, i] for f in range(k_freq)]).conj().T

        # mask frequencies Z with S and copy the result into G
        G[i, S] = (np.expand_dims(Z[i, :], axis=1))

        # solve LASSO in the time domain by applying the fast proximal algorithm
        hrtf[:, i] = sparir(G[i, :], S)

        # convert transfer function from time domain to frequency domain
        Hrtf[:, i] = np.fft.fft(hrtf[:, i])
        W[:, :, i] = np.conj(np.insert(Hrtf[:, i, None], 1, -1, axis=1))

    # final demixing
    demixsparse(Y, X, np.array(range(n_freq)), W)

    if proj_back:
        z = projection_back(Y, X[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, W
    else:
        return Y

def sparir(G, S, delay=0, weights=np.array([]), gini=0):
    """
    Natural-gradient estimation of the complete HRTF from a sparsely recovered HRTF based on
    Koldovský, Zbyněk & Nesta, Francesco & Tichavsky, Petr & Ono, Nobutaka. (2016). Frequency-domain blind speech
     separation using incomplete de-mixing transform. EUSIPCO.2016.
    :param G: sparse HRTF in the frequency domain
    :param S:
    :param delay:
    :param weights:
    :param gini:
    :return:
        g: an (n_frames, n_src) array. The reconstructed hrtf in the time domain
    """
    L = G.shape[0]  # n_freq

    y = np.concatenate((np.real(G[S]), np.imag(G[S])), axis=0)
    M = y.shape[0]

    if gini == 0:  # if no initialization is given
        g = np.zeros((L, 1))
        g[delay] = 1
    else:
        g = gini

    if weights.size == 0:
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
    alphamin = 1e-7  # minimum step - length parameter alpha
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

    while (crit[iter_] > tol) and (iter_ < maxiter - 1):
        prev_r = r
        prev_g = g
        g = soft(prev_g - gradq * (1.0 / alpha), tau / alpha)
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


    return g.flatten()
