'''
Blind Source Separation using Tweaked Independent Vector Analysis with Auxiliary Function

2018 (c) Yaron Dibner & Virgile Hernicot, MIT License
'''
import numpy as np

from pyroomacoustics import stft, istft
from pyroomacoustics.bss.common import projection_back
from lasso import lasso
from demix import *

# A few contrast functions
f_contrasts = {
    'norm': {'f': (lambda r, c, m: c * r), 'df': (lambda r, c, m: c)},
    'cosh': {'f': (lambda r, c, m: m * np.log(np.cosh(c * r))), 'df': (lambda r, c, m: c * m * np.tanh(c * r))}
}

def sparseauxiva(X, S, mu, n_iter, return_filters=False):
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

    lasso(W, S, mu)

    demix(Y, X, np.array(range(n_freq)) ,W)

    # Note: Remember applying projection_back in the end (in ../bss/.common.py) to solve the scale ambiguity
    if return_filters:
        return Y, W
    else:
        return Y
