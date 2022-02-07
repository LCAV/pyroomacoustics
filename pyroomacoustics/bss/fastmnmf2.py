# Copyright (c) 2019 Kouhei Sekiguchi, Yoshiaki Bando
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
FastMNMF2
=========

Blind Source Separation using Fast Multichannel Nonnegative Matrix Factorization 2 (FastMNMF2)
"""
import numpy as np


def fastmnmf2(
    X,
    n_src=None,
    n_iter=30,
    n_components=8,
    mic_index=0,
    W0=None,
    accelerate=True,
    callback=None,
):
    """
    Implementation of FastMNMF2 algorithm presented in

    K. Sekiguchi, Y. Bando, A. A. Nugraha, K. Yoshii, T. Kawahara, *Fast Multichannel Nonnegative
    Matrix Factorization With Directivity-Aware Jointly-Diagonalizable Spatial
    Covariance Matrices for Blind Source Separation*, IEEE/ACM TASLP, 2020.
    [`IEEE <https://ieeexplore.ieee.org/abstract/document/9177266>`_]

    The code of FastMNMF2 with GPU support and more sophisticated initialization
    is available on  https://github.com/sekiguchi92/SoundSourceSeparation

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the observed signal
    n_src: int, optional
        The number of sound sources (default None).
        If None, n_src is set to the number of microphones
    n_iter: int, optional
        The number of iterations (default 30)
    n_components: int, optional
        Number of components in the non-negative spectrum (default 8)
    mic_index: int or 'all', optional
        The index of microphone of which you want to get the source image (default 0).
        If 'all', return the source images of all microphones
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for diagonalizer Q (default None).
        If None, identity matrices are used for all frequency bins.
    accelerate: bool, optional
        If true, the basis and activation of NMF are updated simultaneously (default True)
    callback: func, optional
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    If mic_index is int, returns an (nframes, nfrequencies, nsources) array.
    If mic_index is 'all', returns an (nchannels, nframes, nfrequencies, nsources) array.
    """
    eps = 1e-10
    g_eps = 5e-2
    interval_update_Q = 1  # 2 may work as well and is faster
    interval_normalize = 10
    TYPE_FLOAT = X.real.dtype
    TYPE_COMPLEX = X.dtype

    # initialize parameter
    X_FTM = X.transpose(1, 0, 2)
    n_freq, n_frames, n_chan = X_FTM.shape
    XX_FTMM = np.matmul(X_FTM[:, :, :, None], X_FTM[:, :, None, :].conj())
    if n_src is None:
        n_src = X_FTM.shape[2]

    if W0 is not None:
        Q_FMM = W0
    else:
        Q_FMM = np.tile(np.eye(n_chan).astype(TYPE_COMPLEX), [n_freq, 1, 1])

    g_NM = np.ones([n_src, n_chan], dtype=TYPE_FLOAT) * g_eps
    for m in range(n_chan):
        g_NM[m % n_src, m] = 1

    for m in range(n_chan):
        mu_F = (Q_FMM[:, m] * Q_FMM[:, m].conj()).sum(axis=1).real
        Q_FMM[:, m] /= np.sqrt(mu_F[:, None])

    H_NKT = np.random.rand(n_src, n_components, n_frames).astype(TYPE_FLOAT)
    W_NFK = np.random.rand(n_src, n_freq, n_components).astype(TYPE_FLOAT)
    lambda_NFT = W_NFK @ H_NKT
    Qx_power_FTM = np.abs(np.einsum("fij, ftj -> fti", Q_FMM, X_FTM)) ** 2
    Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM)

    def separate():
        Qx_FTM = np.einsum("fij, ftj -> fti", Q_FMM, X_FTM)
        Qinv_FMM = np.linalg.inv(Q_FMM)
        Y_NFTM = np.einsum("nft, nm -> nftm", lambda_NFT, g_NM)

        if mic_index == "all":
            return np.einsum(
                "fij, ftj, nftj -> itfn", Qinv_FMM, Qx_FTM / Y_NFTM.sum(axis=0), Y_NFTM
            )
        elif type(mic_index) is int:
            return np.einsum(
                "fj, ftj, nftj -> tfn",
                Qinv_FMM[:, mic_index],
                Qx_FTM / Y_NFTM.sum(axis=0),
                Y_NFTM,
            )
        else:
            raise ValueError("mic_index should be int or 'all'")

    # update parameters
    for epoch in range(n_iter):
        if callback is not None and epoch % 10 == 0:
            callback(separate())

        # update W and H (basis and activation of NMF)
        tmp1_NFT = np.einsum("nm, ftm -> nft", g_NM, Qx_power_FTM / (Y_FTM**2))
        tmp2_NFT = np.einsum("nm, ftm -> nft", g_NM, 1 / Y_FTM)

        numerator = np.einsum("nkt, nft -> nfk", H_NKT, tmp1_NFT)
        denominator = np.einsum("nkt, nft -> nfk", H_NKT, tmp2_NFT)
        W_NFK *= np.sqrt(numerator / denominator)

        if not accelerate:
            tmp1_NFT = np.einsum("nm, ftm -> nft", g_NM, Qx_power_FTM / (Y_FTM**2))
            tmp2_NFT = np.einsum("nm, ftm -> nft", g_NM, 1 / Y_FTM)
            lambda_NFT = W_NFK @ H_NKT + eps
            Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM) + eps

        numerator = np.einsum("nfk, nft -> nkt", W_NFK, tmp1_NFT)
        denominator = np.einsum("nfk, nft -> nkt", W_NFK, tmp2_NFT)
        H_NKT *= np.sqrt(numerator / denominator)

        lambda_NFT = W_NFK @ H_NKT + eps
        Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM) + eps

        # update g_NM (diagonal element of spatial covariance matrices)
        numerator = np.einsum("nft, ftm -> nm", lambda_NFT, Qx_power_FTM / (Y_FTM**2))
        denominator = np.einsum("nft, ftm -> nm", lambda_NFT, 1 / Y_FTM)
        g_NM *= np.sqrt(numerator / denominator)
        Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM) + eps

        # udpate Q (joint diagonalizer)
        if (interval_update_Q <= 0) or (epoch % interval_update_Q == 0):
            for m in range(n_chan):
                V_FMM = (
                    np.einsum("ftij, ft -> fij", XX_FTMM, 1 / Y_FTM[..., m]) / n_frames
                )
                tmp_FM = np.linalg.solve(
                    np.matmul(Q_FMM, V_FMM), np.eye(n_chan)[None, m]
                )
                Q_FMM[:, m] = (
                    tmp_FM
                    / np.sqrt(
                        np.einsum("fi, fij, fj -> f", tmp_FM.conj(), V_FMM, tmp_FM)
                    )[:, None]
                ).conj()
                Qx_power_FTM = np.abs(np.einsum("fij, ftj -> fti", Q_FMM, X_FTM)) ** 2

        # normalize
        if (interval_normalize <= 0) or (epoch % interval_normalize == 0):
            phi_F = np.einsum("fij, fij -> f", Q_FMM, Q_FMM.conj()).real / n_chan
            Q_FMM /= np.sqrt(phi_F)[:, None, None]
            W_NFK /= phi_F[None, :, None]

            mu_N = g_NM.sum(axis=1)
            g_NM /= mu_N[:, None]
            W_NFK *= mu_N[:, None, None]

            nu_NK = W_NFK.sum(axis=1)
            W_NFK /= nu_NK[:, None]
            H_NKT *= nu_NK[:, :, None]

            lambda_NFT = W_NFK @ H_NKT + eps
            Qx_power_FTM = np.abs(np.einsum("fij, ftj -> fti", Q_FMM, X_FTM)) ** 2
            Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM) + eps

    return separate()
