import numpy as np


def demix(Y, X, W,freq, partial=False):
    for f in range(freq):
        if partial:
            Y[:, S[f], :] = np.dot(X[:, S[f], :], np.conj(W[S[f], :, :]))
        else:
            Y[:, f, :] = np.dot(X[:, f, :], np.conj(W[f, :, :]))
