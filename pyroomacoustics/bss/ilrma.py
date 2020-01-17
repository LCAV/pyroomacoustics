# Copyright (c) 2018-2019 Juan Azcarreta, Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
ILRMA
=====

Blind Source Separation using Independent Low-Rank Matrix Analysis (ILRMA).
"""
import numpy as np
from .common import projection_back


def ilrma(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    W0=None,
    n_components=2,
    return_filters=False,
    callback=None,
):
    """
    Implementation of ILRMA algorithm without partitioning function for BSS presented in

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, *Determined blind
    source separation unifying independent vector analysis and nonnegative matrix
    factorization,* IEEE/ACM Trans. ASLP, vol. 24, no. 9, pp. 1626-1641, Sept. 2016

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, and H. Saruwatari *Determined
    Blind Source Separation with Independent Low-Rank Matrix Analysis,* in
    Audio Source Separation, S. Makino, Ed. Springer, 2018, pp. 125-156.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the observed signal n_src: int, optional
        The number of sources or independent components
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for demixing matrix
    n_components: int
        Number of components in the non-negative spectrum
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix W (nfrequencies, nchannels, nsources)
    if ``return_filters`` keyword is True.
    """
    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    # Only supports determined case
    assert n_chan == n_src, "There should be as many microphones as sources"

    # initialize the demixing matrices
    # The demixing matrix has the following dimensions (nfrequencies, nchannels, nsources),
    if W0 is None:
        W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)
    else:
        W = W0.copy()

    # initialize the nonnegative matrixes with random values
    T = np.array(0.1 + 0.9 * np.random.rand(n_src, n_freq, n_components))
    V = np.array(0.1 + 0.9 * np.random.rand(n_src, n_frames, n_components))
    R = np.zeros((n_src, n_freq, n_frames))
    I = np.eye(n_src, n_src)
    U = np.zeros((n_freq, n_src, n_chan, n_chan), dtype=X.dtype)
    product = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    lambda_aux = np.zeros(n_src)
    eps = 1e-15
    eyes = np.tile(np.eye(n_chan, n_chan), (n_freq, 1, 1))

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_src, n_frames), dtype=X.dtype)
    X_original = X
    X = X.transpose([1, 2, 0]).copy()

    np.matmul(T, V.swapaxes(1, 2), out=R)

    # Compute the demixed output
    def demix(Y, X, W):
        Y[:, :, :] = np.matmul(W, X)

    demix(Y, X, W)

    # P.shape == R.shape == (n_src, n_freq, n_frames)
    P = np.power(abs(Y.transpose([1, 0, 2])), 2.0)
    iR = 1 / R

    for epoch in range(n_iter):
        if callback is not None and epoch % 10 == 0:
            Y_t = Y.transpose([2, 0, 1])
            if proj_back:
                z = projection_back(Y_t, X_original[:, :, 0])
                callback(Y_t * np.conj(z[None, :, :]))
            else:
                callback(Y_t)

        # simple loop as a start
        for s in range(n_src):
            ## NMF
            ######

            T[s, :, :] *= np.sqrt(
                np.dot(P[s, :, :] * iR[s, :, :] ** 2, V[s, :, :])
                / np.dot(iR[s, :, :], V[s, :, :])
            )
            T[T < eps] = eps

            R[s, :, :] = np.dot(T[s, :, :], V[s, :, :].T)
            iR[s, :, :] = 1 / R[s, :, :]

            V[s, :, :] *= np.sqrt(
                np.dot(P[s, :, :].T * iR[s, :, :].T ** 2, T[s, :, :])
                / np.dot(iR[s, :, :].T, T[s, :, :])
            )
            V[V < eps] = eps

            R[s, :, :] = np.dot(T[s, :, :], V[s, :, :].T)
            iR[s, :, :] = 1 / R[s, :, :]

            ## IVA
            ######

            # Compute Auxiliary Variable
            # shape: (n_freq, n_chan, n_chan)
            C = np.matmul((X * iR[s, :, None, :]), np.conj(X.swapaxes(1, 2)) / n_frames)

            WV = np.matmul(W, C)
            W[:, s, :] = np.conj(np.linalg.solve(WV, eyes[:, :, s]))

            # normalize
            denom = np.matmul(
                np.matmul(W[:, None, s, :], C[:, :, :]), np.conj(W[:, s, :, None])
            )
            W[:, s, :] /= np.sqrt(denom[:, :, 0])

        demix(Y, X, W)
        np.power(abs(Y.transpose([1, 0, 2])), 2.0, out=P)

        for s in range(n_src):
            lambda_aux[s] = 1 / np.sqrt(np.mean(P[s, :, :]))

            W[:, :, s] *= lambda_aux[s]
            P[s, :, :] *= lambda_aux[s] ** 2
            R[s, :, :] *= lambda_aux[s] ** 2
            T[s, :, :] *= lambda_aux[s] ** 2

    Y = Y.transpose([2, 0, 1]).copy()

    if proj_back:
        z = projection_back(Y, X_original[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, W
    else:
        return Y
