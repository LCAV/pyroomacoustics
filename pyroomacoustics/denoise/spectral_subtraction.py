# Single Channel Noise Removal using Spectral Subtraction
# Copyright (C) 2019  Eric Bezzam, Alexis Mermet
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


class SpectralSub(object):
    """
    Here we have a class for performing **single channel** noise reduction via
    spectral subtraction. The instantaneous signal energy and noise floor is
    estimated at each time instance (for each frequency bin) and this is used
    to compute a gain filter with which to perform spectral subtraction.

    For a given frame `n`, the gain for frequency bin `k` is given by:

    .. math::

        G[k, n] = \max \\left \{ \\left ( \dfrac{P[k, n]-\\beta P_N[k, n]}{P[k, n]} \\right )^\\alpha, G_{min} \\right \},

    where :math:`G_{min} = 10^{-(db\_reduc/20)}` and :math:`db\_reduc` is the
    maximum reduction (in dB) that we are willing to perform for each bin (a
    high value can actually be detrimental, see below). The instantaneous
    energy :math:`P[k,n]` is computed by simply squaring the frequency
    amplitude at the bin `k`. The time-frequency decomposition of the input
    signal is typically done with the STFT and overlapping frames. The noise
    estimate :math:`P_N[k, n]` for frequency bin `k` is given by looking back a
    certain number of frames :math:`L` and selecting the bin with the lowest
    energy:

    .. math::

        P_N[k, n] = \min_{[n-L, n]} P[k, n]

    This approach works best when the SNR is positive and the noise is rather
    stationary. An alternative approach for the noise estimate (also in the
    case of stationary noise) would be to apply a lowpass filter for each
    frequency bin.

    With a large suppression, i.e. large values for :math:`db\_reduc`, we can
    observe a typical artefact of such spectral subtraction approaches, namely
    "musical noise".
    `Here <https://www.vocal.com/noise-reduction/musical-noise/>`_ is nice
    article about noise reduction and musical noise.

    Adjusting the constants :math:`\\beta` and :math:`\\alpha` also presents a
    trade-off between suppression and undesirable artefacts, i.e. more
    noticeable musical noise.

    Below is an example of how to use this class to emulate a streaming/online
    input. A full example can be found
    `here <https://github.com/LCAV/pyroomacoustics/blob/master/examples/noise_reduction_spectral_subtraction.py>`__.

    ::

        # initialize STFT and SpectralSub objects
        nfft = 512
        stft = pra.transform.STFT(nfft, hop=nfft//2,
                                  analysis_window=pra.hann(nfft))
        scnr = pra.denoise.SpectralSub(nfft, db_reduc=10, lookback=5,
                                       beta=20, alpha=3)

        # apply block-by-block
        for n in range(num_blocks):

            # go to frequency domain for noise reduction
            stft.analysis(mono_noisy)
            gain_filt = scnr.compute_gain_filter(stft.X)

            # estimating input convolved with unknown response
            mono_denoised = stft.synthesis(gain_filt*stft.X)


    There also exists a "one-shot" function.

    ::

        # import or create `noisy_signal`
        denoised_signal = apply_spectral_sub(noisy_signal, nfft=512,
                                             db_reduc=10, lookback=5,
                                             beta=20, alpha=3)


    Parameters
    ----------
    nfft: int
        FFT size. Length of gain filter, i.e. the number of frequency bins, is
        given by ``nfft//2+1``.
    db_reduc: float
        Maximum reduction in dB for each bin.
    lookback: int
        How many frames to look back for the noise estimate.
    beta: float
        Overestimation factor to "push" the gain filter value (at each
        frequency) closer to the dB reduction specified by ``db_reduc``.
    alpha: float, optional
        Exponent factor to modify transition behavior towards the dB reduction 
        specified by ``db_reduc``. Default is 1.

    """

    def __init__(self, nfft, db_reduc, lookback, beta, alpha=1):

        self.beta = beta
        self.alpha = alpha

        self.n_bins = nfft//2+1
        self.p_prev = np.zeros((self.n_bins, lookback+1))
        self.gmin = 10**(-db_reduc/20)

        self.p_sn = np.zeros(self.n_bins)
        self.p_n = np.zeros(self.n_bins)

    def compute_gain_filter(self, X):
        """
        Parameters
        ----------
        X: numpy array
            Complex spectrum of length ``nfft//2+1``.

        Returns
        -------
        numpy array
            Gain filter to multiply given spectrum with.
        """

        # estimate of signal + noise at current time
        self.p_sn[:] = np.real(np.conj(X)*X)

        # estimate of noise level
        self.p_prev[:, -1] = self.p_sn
        self.p_n[:] = np.min(self.p_prev, axis=1)

        # compute gain filter
        gain_filter = [max((max(self.p_sn[k]-self.beta*self.p_n[k], 0) /
                            self.p_sn[k])**self.alpha, self.gmin)
                       for k in range(self.n_bins)]

        # update
        self.p_prev = np.roll(self.p_prev, -1, axis=1)

        return gain_filter


def apply_spectral_sub(noisy_signal, nfft=512, db_reduc=25, lookback=12,
                       beta=30, alpha=1):
    """
    One-shot function to apply spectral subtraction approach.

    Parameters
    ----------
    noisy_signal : numpy array
        Real signal in time domain.
    nfft: int
        FFT size. Length of gain filter, i.e. the number of frequency bins, is
        given by ``nfft//2+1``.
    db_reduc: float
        Maximum reduction in dB for each bin.
    lookback: int
        How many frames to look back for the noise estimate.
    beta: float
        Overestimation factor to "push" the gain filter value (at each
        frequency) closer to the dB reduction specified by ``db_reduc``.
    alpha: float, optional
        Exponent factor to modify transition behavior towards the dB reduction
        specified by ``db_reduc``. Default is 1.

    Returns
    -------
    numpy array
        Enhanced/denoised signal.
    """

    from pyroomacoustics import hann
    from pyroomacoustics.transform import STFT

    hop = nfft // 2
    window = hann(nfft, flag='asymmetric', length='full')
    stft = STFT(nfft, hop=hop, analysis_window=window, streaming=True)
    scnr = SpectralSub(nfft, db_reduc, lookback, beta, alpha)

    processed_audio = np.zeros(noisy_signal.shape)
    n = 0
    while noisy_signal.shape[0] - n >= hop:
        # SCNR in frequency domain
        stft.analysis(noisy_signal[n:(n + hop), ])
        gain_filt = scnr.compute_gain_filter(stft.X)

        # back to time domain
        processed_audio[n:n + hop, ] = stft.synthesis(gain_filt * stft.X)

        # update step
        n += hop

    return processed_audio
