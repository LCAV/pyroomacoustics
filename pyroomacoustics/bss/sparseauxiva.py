# coding=utf-8
# Blind Source Separation for sparsely mixed signals based on Independent Vector Analysis (IVA)
# with Auxiliary Function
# Copyright (C) 2018 Yaron Dibner, Virgile Hernicot, Juan Azcarreta, MIT License
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
#
# You should have received a copy of the MIT License along with this program. If
# not, see <https://opensource.org/licenses/MIT>.

import numpy as np
from .common import projection_back, sparir


def sparseauxiva(
    X,
    S=None,
    n_src=None,
    n_iter=20,
    proj_back=True,
    W0=None,
    model="laplace",
    return_filters=False,
    callback=None,
):

    """
    Implementation of sparse AuxIVA algorithm for BSS presented in

    J. Janský, Z. Koldovský, and N. Ono, *A computationally cheaper method
    for blind speech separation based on AuxIVA and incomplete demixing transform,*
    Proc. IEEE, IWAENC, pp. 1-5, Sept. 2016.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components
    S: ndarray (k_freq)
        Indexes of active frequency bins for sparse AuxIVA
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for demixing matrix
    model: str
        The model of source distribution 'gauss' or 'laplace' (default)
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    n_frames, n_freq, n_chan = X.shape

    if S is None:
        k_freq = n_freq
    else:
        k_freq = S.shape[0]

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    # for now it only supports determined case
    assert (
        n_chan == n_src
    ), "Only the determined case is implemented (n_src == n_channels)."

    if model not in ["laplace", "gauss"]:
        raise ValueError("Model should be either " "laplace" " or " "gauss" ".")

    # initialize the demixing matrices
    if W0 is None:
        W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)
    else:
        W = W0.copy()

    eps = 1e-15
    I = np.eye(n_src, n_src)
    Y = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)
    r = np.zeros((n_frames, n_src))

    def demixsparse(Y, X, S, W):
        for f in range(k_freq):
            Y[:, S[f], :] = np.dot(X[:, S[f], :], np.conj(W[S[f], :, :]))

    # Conventional AuxIVA in the frequency bins k_freq selected by S
    for epoch in range(n_iter):
        demixsparse(Y, X, S, W)

        if callback is not None and epoch % 10 == 0:
            if proj_back:
                z = projection_back(Y, X[:, :, 0])
                callback(Y * np.conj(z[None, :, :]))
            else:
                callback(Y)

        # shape: (n_frames, n_src)
        if model == "laplace":
            r[:, :] = 2.0 * np.linalg.norm(Y, axis=1)
        elif model == "gauss":
            r[:, :] = (np.linalg.norm(Y, axis=1) ** 2) / n_freq

        # ensure some numerical stability
        r[r < eps] = eps

        r_inv = 1.0 / r

        # Compute Auxiliary Variable
        V = np.mean(
            (X[:, :, None, :, None] * r_inv[:, None, :, None, None])
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
            W[:, :, s] /= np.sqrt(np.sum(P1 * P2, axis=1))[:, None]

    # LASSO regularization to reconstruct the complete relative transfer function
    Z = np.zeros((n_src, k_freq), dtype=W.dtype)
    G = np.zeros((n_src, n_freq, 1), dtype=Z.dtype)
    hrtf = np.zeros((n_freq, n_src), dtype=W.dtype)

    for i in range(n_src):
        # calculate sparse relative transfer function from demixing matrix
        Z[i, :] = np.conj(-W[S, 0, i] / W[S, 1, i]).T

        # copy selected active frequencies in Z to sparse G
        G[i, S] = np.expand_dims(Z[i, :], axis=1)

        # apply fast proximal algorithm to reconstruct the complete real-valued relative transfer function
        hrtf[:, i] = sparir(G[i, :], S)

        # recover relative transfer function back to the frequency domain
        hrtf[:, i] = np.fft.fft(hrtf[:, i])
        # assemble back the complete demixing matrix
        W[:, :, i] = np.conj(np.insert(hrtf[:, i, None], 1, -1, axis=1))

    # apply final demixing in the whole frequency range
    demixsparse(Y, X, np.array(range(n_freq)), W)

    # Projection back to correct scale ambiguity
    if proj_back:
        z = projection_back(Y, X[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, W
    else:
        return Y
