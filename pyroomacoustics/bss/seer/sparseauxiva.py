'''
Blind Source Separation using Tweaked Independent Vector Analysis with Auxiliary Function

2018 (c) Yaron Dibner & Virgile Hernicot, MIT License
'''
import numpy as np

from pyroomacoustics import stft, istft
from pyroomacoustics.bss.common import projection_back

# A few contrast functions
f_contrasts = {
    'norm': {'f': (lambda r, c, m: c * r), 'df': (lambda r, c, m: c)},
    'cosh': {'f': (lambda r, c, m: m * np.log(np.cosh(c * r))), 'df': (lambda r, c, m: c * m * np.tanh(c * r))}
}


def sparseauxiva(X, S, mu, n_iter, return_filters=False):
    n_frames = X.shape[0]

    n_chan = X.shape[2]

    k_freq = S.shape[0]
    n_freq = X.shape[1]

    # default to determined case

    n_src = X.shape[2]

    # initialize the demixing matrices
    W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)

    f_contrast = f_contrasts['norm']
    f_contrast_args = [1, 1]

    I = np.eye(n_src, n_src)
    Y = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)
    V = np.zeros((n_freq, n_src, n_chan, n_chan), dtype=X.dtype)
    r = np.zeros((n_frames, n_src))
    G_r = np.zeros((n_frames, n_src))


    for epoch in range(n_iter):

        demix(Y, X, W, k_freq, True)

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

    # Here comes Lassoooooooooo
    Z = np.zeros((n_src, k_freq), dtype=W.dtype)
    Hrtf = np.zeros((n_chan, n_src), dtype=W.dtype)
    for i in range(n_src):
        Z[i, :] = np.array([W[S[f], i, 0] / W[S[f], i, 1] for f in range(k_freq)]).conj().T
        #Hrtf[:, i] = np.argmin(np.linalg.norm(Z[i,:] - ))

    demix(Y, X, W, n_freq)

    if return_filters:
        return Y, W
    else:
        return Y
