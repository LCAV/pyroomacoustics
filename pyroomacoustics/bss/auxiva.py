# Copyright (c) 2018-2019 Robin Scheibler
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
AuxIVA
======

Blind Source Separation using independent vector analysis based on auxiliary function.
This function will separate the input signal into statistically independent sources
without using any prior information.

The algorithm in the determined case, i.e., when the number of sources is equal to
the number of microphones, is AuxIVA [1]_. When there are more microphones (the overdetermined case),
a computationaly cheaper variant (OverIVA) is used [2]_.

Example
-------
.. code-block:: python

    from scipy.io import wavfile
    import pyroomacoustics as pra

    # read multichannel wav file
    # audio.shape == (nsamples, nchannels)
    fs, audio = wavfile.read("my_multichannel_audio.wav")

    # STFT analysis parameters
    fft_size = 4096  # `fft_size / fs` should be ~RT60
    hop == fft_size // 2  # half-overlap
    win_a = pra.hann(fft_size)  # analysis window
    # optimal synthesis window
    win_s = pra.transform.compute_synthesis_window(win_a, hop)

    # STFT
    # X.shape == (nframes, nfrequencies, nchannels)
    X = pra.transform.analysis(audio, fft_size, hop, win=win_a)

    # Separation
    Y = pra.bss.auxiva(X, n_iter=20)

    # iSTFT (introduces an offset of `hop` samples)
    # y contains the time domain separated signals
    # y.shape == (new_nsamples, nchannels)
    y = pra.transform.synthesis(Y, fft_size, hop, win=win_s)

References
----------

.. [1] N. Ono, *Stable and fast update rules for independent vector analysis based
    on auxiliary function technique,* Proc. IEEE, WASPAA, pp. 189-192, Oct. 2011.

.. [2] R. Scheibler and N. Ono, Independent Vector Analysis with more Microphones
    than Sources, arXiv, 2019.  https://arxiv.org/abs/1905.07880
"""
import numpy as np

from .common import projection_back


def auxiva(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    W0=None,
    model="laplace",
    init_eig=False,
    return_filters=False,
    callback=None,
):

    """
    This is an implementation of AuxIVA/OverIVA that separates the input
    signal into statistically independent sources. The separation is done
    in the time-frequency domain and the FFT length should be approximately
    equal to the reverberation time.

    Two different statistical models (Laplace or time-varying Gauss) can
    be used by using the keyword argument `model`. The performance of Gauss
    model is higher in good conditions (few sources, low noise), but Laplace
    (the default) is more robust in general.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components. When
        ``n_src==nchannels``, the algorithms is identical to AuxIVA. When
        ``n_src==1``, then it is doing independent vector extraction.
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nsrc, nchannels), optional
        Initial value for demixing matrix
    model: str
        The model of source distribution 'gauss' or 'laplace' (default)
    init_eig: bool, optional (default ``False``)
        If ``True``, and if ``W0 is None``, then the weights are initialized
        using the principal eigenvectors of the covariance matrix of the input
        data. When ``False``, the demixing matrices are initialized with identity
        matrix.
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor
        convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = n_chan

    assert (
        n_src <= n_chan
    ), "The number of sources cannot be more than the number of channels."

    if model not in ["laplace", "gauss"]:
        raise ValueError("Model should be either ""laplace"" or ""gauss"".")

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)

    W_hat = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    W = W_hat[:, :n_src, :]
    J = W_hat[:, n_src:, :n_src]

    def tensor_H(T):
        return np.conj(T).swapaxes(1, 2)

    def update_J_from_orth_const():
        tmp = np.matmul(W, Cx)
        J[:, :, :] = tensor_H(np.linalg.solve(tmp[:, :, :n_src], tmp[:, :, n_src:]))

    # initialize A and W
    if W0 is None:

        if init_eig:
            # Initialize the demixing matrices with the principal
            # eigenvectors of the input covariance
            v, w = np.linalg.eig(Cx)
            for f in range(n_freq):
                ind = np.argsort(v[f])[-n_src:]
                W[f, :, :] = np.conj(w[f][:, ind])

        else:
            # Or with identity
            for f in range(n_freq):
                W[f, :, :n_src] = np.eye(n_src)

    else:
        W[:, :, :] = W0

    # We still need to initialize the rest of the matrix
    if n_src < n_chan:
        update_J_from_orth_const()
        for f in range(n_freq):
            W_hat[f, n_src:, n_src:] = -np.eye(n_chan - n_src)

    eps = 1e-15
    eyes = np.tile(np.eye(n_chan, n_chan), (n_freq, 1, 1))
    V = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    r_inv = np.zeros((n_src, n_frames))
    r = np.zeros((n_src, n_frames))

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_src, n_frames), dtype=X.dtype)
    X_original = X
    X = X.transpose([1, 2, 0]).copy()

    # Compute the demixed output
    def demix(Y, X, W):
        Y[:, :, :] = np.matmul(W, X)

    for epoch in range(n_iter):

        demix(Y, X, W)

        if callback is not None and epoch % 10 == 0:
            Y_tmp = Y.transpose([2, 0, 1])
            if proj_back:
                z = projection_back(Y_tmp, X_original[:, :, 0])
                callback(Y_tmp * np.conj(z[None, :, :]))
            else:
                callback(Y_tmp)

        # shape: (n_frames, n_src)
        if model == "laplace":
            r[:, :] = 2.0 * np.linalg.norm(Y, axis=0)
        elif model == "gauss":
            r[:, :] = (np.linalg.norm(Y, axis=0) ** 2) / n_freq

        # ensure some numerical stability
        r[r < eps] = eps

        r_inv[:, :] = 1.0 / r

        # Update now the demixing matrix
        for s in range(n_src):
            # Compute Auxiliary Variable
            # shape: (n_freq, n_chan, n_chan)
            V[:, :, :] = np.matmul(
                (X * r_inv[None, s, None, :]), np.conj(X.swapaxes(1, 2)) / n_frames
            )

            WV = np.matmul(W_hat, V)
            W[:, s, :] = np.conj(np.linalg.solve(WV, eyes[:, :, s]))

            # normalize
            denom = np.matmul(
                np.matmul(W[:, None, s, :], V[:, :, :]), np.conj(W[:, s, :, None])
            )
            W[:, s, :] /= np.sqrt(denom[:, :, 0])

            # Update the mixing matrix according to orthogonal constraints
            if n_src < n_chan:
                update_J_from_orth_const()

    demix(Y, X, W)

    Y = Y.transpose([2, 0, 1]).copy()

    if proj_back:
        z = projection_back(Y, X_original[:, :, 0])
        Y *= np.conj(z[None, :, :])

    if return_filters:
        return Y, W
    else:
        return Y
