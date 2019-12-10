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
FastMNMF
========

Blind Source Separation based on Fast Multichannel Nonnegative Matrix Factorization (FastMNMF)
"""
import numpy as np


def fastmnmf(
    X,
    n_src=None,
    n_iter=30,
    W0=None,
    n_components=4,
    callback=None,
    mic_index=0,
    interval_update_Q=3,
    interval_normalize=10,
    initialize_ilrma=False
):
    """
    Implementation of FastMNMF algorithm presented in

    K. Sekiguchi, A. A. Nugraha, Y. Bando, K. Yoshii, *Fast Multichannel Source 
    Separation Based on Jointly Diagonalizable Spatial Covariance Matrices*, EUSIPCO, 2019. [`arXiv <https://arxiv.org/abs/1903.03237>`_]

    The code of FastMNMF with GPU support and FastMNMF-DP which integrates DNN-based source model
    into FastMNMF is available on https://github.com/sekiguchi92/SpeechEnhancement 

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the observed signal 
    n_src: int, optional
        The number of sound sources (if n_src = None, n_src is set to the number of microphone)
    n_iter: int, optional
        The number of iterations
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for diagonalizer Q
        Demixing matrix can be used as the initial value
    n_components: int, optional
        Number of components in the non-negative spectrum
    callback: func, optional
        A callback function called every 10 iterations, allows to monitor convergence
    mic_index: int, optional
        The index of microphone of which you want to get the source image
    interval_update_Q: int, optional
        The interval of updating Q
    interval_normalize: int, optional
        The interval of parameter normalization
    initialize_ilrma: boolean, optional
        Initialize diagonalizer Q by using ILRMA

    Returns
    -------
    separated spectrogram: numpy.ndarray
        An (nframes, nfrequencies, nsources) array. 
    """

    # initialize parameter
    X_FTM = X.transpose(1, 0, 2)
    n_freq, n_frames, n_chan = X_FTM.shape
    XX_FTMM = np.matmul(X_FTM[:, :, :, None], X_FTM[:, :, None, :].conj())
    if n_src is None:
        n_src = X_FTM.shape[2] # determined case (the number of source = the number of microphone)

    covarianceDiag_NFM = np.random.rand(n_src, n_freq, n_chan)
    power_observation_FT = (np.abs(X_FTM) ** 2).mean(axis=2) # F T
    if initialize_ilrma: # initialize by using ILRMA
        from pyroomacoustics.bss.ilrma import ilrma
        _, W = ilrma(X, n_iter=10, n_components=2, proj_back=False, return_filters=True)
        diagonalizer_FMM = W
        covarianceDiag_NFM[0] = 1e-4
        covarianceDiag_NFM[0, :, 0] = 1
    elif W0 != None: # initialize by W0
        diagonalizer_FMM = W0
    else: # initialize by using observed signals
        covarianceMatrix_FMM = XX_FTMM.sum(axis=1) / power_observation_FT.sum(axis=1)[:, None, None] # F M M
        covarianceMatrix_FMM = covarianceMatrix_FMM / np.trace(covarianceMatrix_FMM, axis1=1 ,axis2=2)[:, None, None]
        eig_val, eig_vec = np.linalg.eig(covarianceMatrix_FMM)
        diagonalizer_FMM = eig_vec.transpose(0, 2, 1).conj()
        covarianceDiag_NFM[1] = eig_val.real
    for m in range(n_chan):
        mu_F = (diagonalizer_FMM[:, m] * diagonalizer_FMM[:, m].conj()).sum(axis=1).real
        diagonalizer_FMM[:, m] = diagonalizer_FMM[:, m] / np.sqrt(mu_F[:, None])
    H_NKT = np.random.rand(n_src, n_components, n_frames)
    W_NFK = np.random.rand(n_src, n_freq, n_components)
    lambda_NFT = np.matmul(W_NFK, H_NKT)
    Qx_power_FTM = np.abs((  np.matmul(diagonalizer_FMM[:, None], X_FTM[:, :, :, None])  )[:, :, :, 0]) ** 2
    Y_FTM = (lambda_NFT[..., None] * covarianceDiag_NFM[:, :, None]).sum(axis=0)

    def separate():
        Qx_FTM = (diagonalizer_FMM[:, None] * X_FTM[:, :, None]).sum(axis=3)
        for n in range(n_src):
            tmp = (  np.matmul(np.linalg.inv(diagonalizer_FMM)[:, None], (Qx_FTM * ( (lambda_NFT[n, :, :, None] * covarianceDiag_NFM[n, :, None]) / (lambda_NFT[..., None]* covarianceDiag_NFM[:, :, None]).sum(axis=0) ) )[..., None])  )[:, :, mic_index, 0]
            if n == 0:
                separated_spec = np.zeros([n_src, tmp.shape[0], tmp.shape[1]], dtype=np.complex)
            separated_spec[n] = tmp
        return separated_spec.transpose(2, 1, 0)

    # update parameters
    for epoch in range(n_iter):
        if callback is not None and epoch % 10 == 0:
            separated_spec = separate()
            callback(separated_spec)

        # update_WH (basis and activation of NMF)
        tmp_yb1 = (covarianceDiag_NFM[:, :, None] * (Qx_power_FTM / (Y_FTM ** 2))[None]).sum(axis=3)  # [N, F, T]
        tmp_yb2 = (covarianceDiag_NFM[:, :, None] / Y_FTM[None]).sum(axis=3)  # [N, F, T]
        a_1 = (H_NKT[:, None, :, :] * tmp_yb1[:, :, None]).sum(axis=3)  # [N, F, K]
        b_1 = (H_NKT[:, None, :, :] * tmp_yb2[:, :, None]).sum(axis=3)  # [N, F, K]
        W_NFK *= np.sqrt(a_1 / b_1)

        a_1 = (W_NFK[:, :, :, None] * tmp_yb1[:, :, None]).sum(axis=1)  # [N, K, T]
        b_1 = (W_NFK[:, :, :, None] * tmp_yb2[:, :, None]).sum(axis=1)  # [N, F, K]
        H_NKT *= np.sqrt(a_1 / b_1)

        lambda_NFT = np.matmul(W_NFK, H_NKT)
        Y_FTM = (lambda_NFT[..., None] * covarianceDiag_NFM[:, :, None]).sum(axis=0)

        # update diagonal element of spatial covariance matrix
        a_1 = (lambda_NFT[..., None] * (Qx_power_FTM / (Y_FTM ** 2))[None]).sum(axis=2) # N F T M
        b_1 = (lambda_NFT[..., None] / Y_FTM[None]).sum(axis=2)
        covarianceDiag_NFM = covarianceDiag_NFM * np.sqrt(a_1 / b_1) +  1e-8
        Y_FTM = (lambda_NFT[..., None] * covarianceDiag_NFM[:, :, None]).sum(axis=0)

        # udpate Diagonalizer which jointly diagonalize spatial covariance matrix
        if (interval_update_Q <= 0) or (epoch % interval_update_Q == interval_update_Q-1):
            for m in range(n_chan):
                V_FMM = (XX_FTMM / Y_FTM[:, :, m, None, None]).mean(axis=1)
                tmp_FM = np.linalg.solve(np.matmul(diagonalizer_FMM, V_FMM), np.eye(n_chan)[None, m])
                diagonalizer_FMM[:, m] = (tmp_FM / np.sqrt(( (tmp_FM.conj()[:, :, None] * V_FMM).sum(axis=1) * tmp_FM).sum(axis=1) )[:, None]).conj()
            Qx_power_FTM = np.abs((np.matmul(diagonalizer_FMM[:, None], X_FTM[:, :, :, None]))[:, :, :, 0]) ** 2

        # normalize
        if (interval_normalize <= 0) or (epoch % interval_normalize == 0):
            phi_F = np.sum(diagonalizer_FMM * diagonalizer_FMM.conj(), axis=(1, 2)).real / n_chan
            diagonalizer_FMM = diagonalizer_FMM / np.sqrt(phi_F)[:, None, None]
            covarianceDiag_NFM = covarianceDiag_NFM / phi_F[None, :, None]

            mu_NF = (covarianceDiag_NFM).sum(axis=2).real
            covarianceDiag_NFM = covarianceDiag_NFM / mu_NF[:, :, None]
            lambda_NFT = lambda_NFT * mu_NF[:, :, None]
            Qx_power_FTM = np.abs((  np.matmul(diagonalizer_FMM[:, None], X_FTM[:, :, :, None])  )[:, :, :, 0]) ** 2
            Y_FTM = (lambda_NFT[..., None] * covarianceDiag_NFM[:, :, None]).sum(axis=0)

    return separate()
