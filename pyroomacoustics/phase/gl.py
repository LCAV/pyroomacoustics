# coding=utf-8
# Phase reconstruction with the Griffin-Lim algorithm
#
# Copyright (C) 2019 Robin Scheibler, MIT License
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
"""
Implementation of the classic phase reconstruction from Griffin and Lim [1]_.
The input to the algorithm is the magnitude from STFT measurements.

The algorithm works by starting by assigning a (possibly random) initial phase to the
measurements, and then iteratively

1. Reconstruct the time-domain signal
2. Re-apply STFT
3. Enforce the known magnitude of the measurements

The implementation supports different types of initialization via the keyword argument ``ini``.

1. If omitted, the initial phase is uniformly zero
2. If ``ini="random"``, a random phase is used
3. If ``ini=A`` for a ``numpy.ndarray`` of the same shape as the input magnitude, ``A / numpy.abs(A)`` is used for initialization

Example
-------

.. code-block:: python

    import numpy as np
    from scipy.io import wavfile
    import pyroomacoustics as pra

    # We open a speech sample
    filename = "examples/input_samples/cmu_arctic_us_axb_a0004.wav"
    fs, audio = wavfile.read(filename)

    # These are the parameters of the STFT
    fft_size = 512
    hop = fft_size // 4
    win_a = np.hamming(fft_size)
    win_s = pra.transform.compute_synthesis_window(win_a, hop)
    n_iter = 200

    engine = pra.transform.STFT(
        fft_size, hop=hop, analysis_window=win_a, synthesis_window=win_s
    )
    X = engine.analysis(audio)
    X_mag = np.abs(X)
    X_mag_norm = np.linalg.norm(X_mag) ** 2

    # monitor convergence
    errors = []

    # the callback to track the spectral distance convergence
    def cb(epoch, Y, y):
        # we measure convergence via spectral distance
        Y_2 = engine.analysis(y)
        sd = np.linalg.norm(X_mag - np.abs(Y_2)) ** 2 / X_mag_norm
        # save in the list every 10 iterations
        if epoch % 10 == 0:
            errors.append(sd)

    pra.phase.griffin_lim(X_mag, hop, win_a, n_iter=n_iter, callback=cb)

    plt.semilogy(np.arange(len(errors)) * 10, errors)
    plt.show()


References
----------

.. [1] D. Griffin and J. Lim, “Signal estimation from modified short-time Fourier
    transform,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol.
    32, no. 2, pp. 236–243, 1984.
"""
# This is needed to check for string types
# in a way compatible between python 2 and 3
try:
    basestring
except NameError:
    basestring = str

import numpy as np
from ..transform import STFT, compute_synthesis_window


def griffin_lim(
    X,
    hop,
    analysis_window,
    fft_size=None,
    stft_kwargs={},
    n_iter=100,
    ini=None,
    callback=None,
):
    """
    Implementation of the Griffin-Lim phase reconstruction algorithm from STFT magnitude measurements.

    Parameters
    ----------
    X: array_like, shape (n_frames, n_freq)
        The STFT magnitude measurements
    hop: int
        The frame shift of the STFT
    analysis_window: array_like, shape (fft_size,)
        The window used for the STFT analysis
    fft_size: int, optional
        The FFT size for the STFT, if omitted it is computed from the dimension of ``X``
    stft_kwargs: dict, optional
        Dictionary of extra parameters for the STFT
    n_iter: int, optional
        The number of iteration
    ini: str or array_like, np.complex, shape (n_frames, n_freq), optional
        The initial value of the phase estimate. If "random", uses a random guess. If ``None``, uses ``0`` phase.
    callback: func, optional
        A callable taking as argument an int and the reconstructed STFT and time-domain signals
    """

    if isinstance(ini, basestring) and ini == "random":
        ini = np.exp(1j * 2 * np.pi * np.random.rand(*X.shape))
    elif ini is None:
        ini = np.ones(X.shape, dtype=np.complex128)
    else:
        # make sure the modulus is one
        ini /= np.abs(ini)

    # take care of the STFT parameters
    if fft_size is None:
        fft_size = 2 * (X.shape[1] - 1)

    # the optimal GL window
    synthesis_window = compute_synthesis_window(analysis_window, hop)

    # create the STFT object
    engine = STFT(
        fft_size,
        hop=hop,
        analysis_window=analysis_window,
        synthesis_window=synthesis_window,
        **stft_kwargs
    )

    # Initialize the signal
    Y = X * ini
    y = engine.synthesis(Y)

    # the successive application of analysis/synthesis introduces
    # a shift of ``fft_size - hop`` that we must correct
    the_shift = fft_size - hop
    y[:-the_shift,] = y[the_shift:,]

    for epoch in range(n_iter):

        # possibly monitor the reconstruction
        if callback is not None:
            callback(epoch, Y, y)

        # back to STFT domain
        Y[:, :] = engine.analysis(y)

        # enforce magnitudes
        Y *= X / np.abs(Y)

        # back to time domain
        y[:-the_shift,] = engine.synthesis(Y)[the_shift:,]

    # last callback
    if callback is not None:
        callback(epoch, Y, y)

    return y
