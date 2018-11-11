import numpy as np


def demix(Y,X,S,W):
    freq = S.shape[0]
    for f in range(freq):
        Y[:, S[f], :] = np.dot(X[:, S[f], :], np.conj(W[S[f], :, :]))
