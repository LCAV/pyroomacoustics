# Single Channel Noise Removal using Iterative Wiener Filtering
# Copyright (C) 2019  Eric Bezzam, Laurent Colbois, Lionel Desarzens
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
from scipy import integrate
from pyroomacoustics import lpc


class IterativeWiener(object):
    """
    A class for performing **single channel** noise reduction in the frequency
    domain with a Wiener filter that is iteratively computed. This
    implementation is based off of the approach presented in:

        J. Lim and A. Oppenheim, *All-Pole Modeling of Degraded Speech,*
        IEEE Transactions on Acoustics, Speech, and Signal Processing 26.3
        (1978): 197-210.

    For each frame, a Wiener filter of the following form is computed and
    applied to the noisy samples in the frequency domain:

    .. math::

        H(\\omega) = \dfrac{P_S(\\omega)}{P_S(\\omega) + \\sigma_d^2},


    where :math:`P_S(\omega)` is the speech power spectral density and
    :math:`\sigma_d^2` is the noise variance.

    The following assumptions are made in order to arrive at the above filter
    as the optimal solution:

    - The noisy samples :math:`y[n]` can be written as:

      .. math::

        y[n] = s[n] + d[n],

      where :math:`s[n]` is the desired signal and :math:`d[n]` is the
      background noise.
    - The signal and noise are uncorrelated.
    - The noise is white Gaussian, i.e. it has a flat power spectrum with
      amplitude :math:`\sigma_d^2`.

    Under these assumptions, the above Wiener filter minimizes the mean-square
    error between the true samples :math:`s[0:N-1]` and the estimated one
    :math:`\hat{s[0:N-1]}` by filtering :math:`y[0:N-1]` with the above filter
    (with :math:`N` being the frame length).

    The fundamental part of this approach is correctly (or as well as possible)
    estimating the speech power spectral density :math:`P_S(\omega)` and the
    noise variance :math:`\sigma_d^2`. For this, we need a **voice activity
    detector** in order to determine when we have incoming speech. In this
    implementation, we use a simple energy threshold on the input frame, which
    is set with the `thresh` input parameter.

    **When no speech is identified**, the input frame is used to update the
    noise variance :math:`\sigma_d^2`. We could simply set :math:`\sigma_d^2`
    to the energy of the input frame. However, we employ a simple IIR filter in
    order to avoid abrupt changes in the noise level (thus adding an assumption
    of stationary):

    .. math::

        \sigma_d^2[k] = \\alpha \cdot \sigma_d^2[k-1] + (1-\\alpha) \cdot \sigma_y^2,

    where :math:`\\alpha` is the smoothing parameter and :math:`\sigma_y^2` is
    the energy of the input frame. A high value of :math:`\\alpha` will update
    the noise level very slowly, while a low value will make it very sensitive
    to changes at the input. The value for :math:`\\alpha` can be set with the
    `alpha` parameter.

    **When speech is identified in the input frame**, an iterative procedure is
    employed in order to estimate :math:`P_S(\omega)` (and therefore the Wiener
    filter :math:`H` as well). This procedure consists of computing :math:`p`
    `linear predictive coding (LPC) coefficients <https://en.wikipedia.org/wiki/Linear_predictive_coding>`_
    of the input frame. The number of LPC coefficients is set with the
    parameter `lpc_order`. These LPC coefficients form an all-pole filter that
    models the vocal tract as described in the above paper (Eq. 1). With these
    coefficients, we can then obtain an estimate of the speech power spectral
    density (Eq. 41b) and thus the corresponding Wiener filter (Eq. 41a). This
    Wiener filter is used to denoise the input frame. Moreover, with this
    denoised frame, we can compute new LPC coefficients and therefore a new
    Wiener filter. The idea behind this approach is that by iteratively
    computing the LPC coefficients as such, we can obtain a better estimate of
    the speech power spectral density. The number of iterations can be set with
    the `iterations` parameter.

    Below is an example of how to use this class to emulate a streaming/online
    input. A full example can be found
    `here <https://github.com/LCAV/pyroomacoustics/blob/master/examples/noise_reduction_wiener_filtering.py>`_.


    ::

        # initialize STFT and IterativeWiener objects
        nfft = 512
        stft = pra.transform.STFT(nfft, hop=nfft//2,
                                  analysis_window=pra.hann(nfft))
        scnr = IterativeWiener(frame_len=nfft, lpc_order=20, iterations=2,
                               alpha=0.8, thresh=0.01)

        # apply block-by-block
        for n in range(num_blocks):

            # go to frequency domain, 50% overlap
            stft.analysis(mono_noisy)

            # compute wiener output
            X = scnr.compute_filtered_output(
                    current_frame=stft.fft_in_buffer,
                    frame_dft=stft.X)

            # back to time domain
            mono_denoised = stft.synthesis(X)


    There also exists a "one-shot" function.

    ::

        # import or create `noisy_signal`
        denoised_signal = apply_iterative_wiener(noisy_signal, frame_len=512,
                                                 lpc_order=20, iterations=2,
                                                 alpha=0.8, thresh=0.01)


    Parameters
    ----------
    frame_len : int
        Frame length in samples.
    lpc_order : int
        Number of LPC coefficients to compute
    iterations : int
        How many iterations to perform in updating the Wiener filter for each
        signal frame.
    alpha : int
        Smoothing factor within [0,1] for updating noise level. Closer to `1`
        gives more weight to the previous noise level, while closer to `0`
        gives more weight to the current frame's level. Closer to `0` can track
        more rapid changes in the noise level. However, if a speech frame is
        incorrectly identified as noise, you can end up removing desired
        speech.
    thresh : float
        Threshold to distinguish between (signal+noise) and (noise) frames. A
        high value will classify more frames as noise but might remove desired
        signal!

    """

    def __init__(self, frame_len, lpc_order, iterations, alpha=0.8,
                 thresh=0.01):

        if frame_len % 2:
            raise ValueError("Frame length should be even as this method "
                             "relies on 50% overlap.")

        if (alpha > 1) or (alpha < 0):
            raise ValueError("`alpha` parameter should be within [0,1].")

        self.frame_len = frame_len
        self.hop = frame_len // 2
        self.lpc_order = lpc_order
        self.iterations = iterations
        self.alpha = alpha

        # simple energy-based voice activity detector
        self.thresh = thresh

        # initialize power spectral densities
        self.speech_psd = np.ones(self.hop+1)
        self.noise_psd = 0
        self.wiener_filt = np.ones(self.hop+1)

    def compute_filtered_output(self, current_frame, frame_dft=None):
        """
        Compute Wiener filter in the frequency domain.

        Parameters
        ----------
        current_frame : numpy array
            Noisy samples.
        frame_dft : numpy array
            DFT of input samples. If not provided, it will be computed.

        Returns
        -------
        numpy array
            Output of denoising in the frequency domain.
        """

        if frame_dft is None:
            frame_dft = np.fft.rfft(current_frame)
        frame_dft /= np.sqrt(self.frame_len)

        frame_energy = np.std(current_frame)**2

        # simple VAD
        if frame_energy < self.thresh:    # noise frame

            # update noise power spectral density
            # assuming white noise, i.e. flat spectrum
            self.noise_psd = self.alpha * self.noise_psd +\
                             (1 - self.alpha) * frame_energy

            # update wiener filter
            self.wiener_filt[:] = compute_wiener_filter(self.speech_psd,
                                                        self.noise_psd)
        else:   # speech frame

            s_i = current_frame

            # iteratively update speech power spectral density / wiener filter
            for i in range(self.iterations):
                a = lpc(s_i, self.lpc_order)
                g2 = compute_squared_gain(a, self.noise_psd, current_frame)
                self.speech_psd[:] = compute_speech_psd(a, g2, self.frame_len)

                # update Wiener filter
                self.wiener_filt[:] = compute_wiener_filter(self.speech_psd,
                                                            self.noise_psd)
                # update current frame with denoised version
                s_i = np.fft.irfft(self.wiener_filt * frame_dft)

        return self.wiener_filt * frame_dft


def compute_speech_psd(a, g2, nfft):
    """
    Compute power spectral density of speech as specified in equation (41b) of

        J. Lim and A. Oppenheim, *All-Pole Modeling of Degraded Speech,*
        IEEE Transactions on Acoustics, Speech, and Signal Processing 26.3
        (1978): 197-210.

    Namely:

    .. math::

        P_S(\\omega) = \dfrac{g^2}{\\left \| 1 - \sum_{k=1}^p a_k \cdot e^{-jk\omega} \\right \|^2},

    where :math:`p` is the LPC order, :math:`a_k` are the LPC coefficients, and
    :math:`g` is an estimated gain factor.

    The power spectral density is computed at the frequencies corresponding to
    a DFT of length `nfft`.

    Parameters
    ----------
    a : numpy array
        LPC coefficients.
    g2 : float
        Squared gain.
    nfft : int
        FFT length.

    Returns
    -------
    numpy array
        Power spectral density from LPC coefficients.
    """
    A = np.fft.rfft(np.r_[np.ones(1), -1*a], nfft)
    return g2 / np.abs(A)**2


def compute_squared_gain(a, noise_psd, y):
    """
    Estimate the squared gain of the speech power spectral density as done on
    p. 204 of

        J. Lim and A. Oppenheim, *All-Pole Modeling of Degraded Speech,*
        IEEE Transactions on Acoustics, Speech, and Signal Processing 26.3
        (1978): 197-210.

    Namely solving for :math:`g^2` such that the following expression is
    satisfied:

    .. math::

        \dfrac{N}{2\pi} \int_{-\pi}^{\pi} \dfrac{g^2}{\\left \| 1 - \sum_{k=1}^p a_k \cdot e^{-jk\omega} \\right \|^2} d\omega = \sum_{n=0}^{N-1} y^2(n) - N\cdot\sigma_d^2,


    where :math:`N` is the number of noisy samples :math:`y`, :math:`a_k`
    are the :math:`p` LPC coefficients, and :math:`\sigma_d^2` is the
    noise variance.

    Parameters
    ----------
    a : numpy array
        LPC coefficients.
    noise_psd : float or numpy array
        Noise variance if white noise, numpy array otherwise.
    y : numpy array
        Noisy time domain samples.

    Returns
    -------
    float
        Squared gain.
    """

    p = len(a)
    def _lpc_all_pole(omega):
        k = np.arange(p) + 1
        return 1 / np.abs(1 - np.dot(a, np.exp(-1j * k * omega))) ** 2

    N = len(y)

    # right hand side of expression
    if np.isscalar(noise_psd):   # white noise, i.e. flat spectrum
        rhs = np.sum(y**2) - N * noise_psd
    else:
        rhs = np.sum(y**2) - np.sum(noise_psd)

    # estimate integral
    d_omega = 2 * np.pi / 1000
    omega_vals = np.arange(-np.pi, np.pi, d_omega)
    vec_integrand = np.vectorize(_lpc_all_pole)
    integral = integrate.trapz(vec_integrand(omega_vals), omega_vals)
    return rhs * 2 * np.pi / N / integral


def compute_wiener_filter(speech_psd, noise_psd):
    """
    Compute Wiener filter in the frequency domain.

    Parameters
    ----------
    speech_psd : numpy array
        Speech power spectral density.
    noise_psd : float or numpy array
        Noise variance if white noise, numpy array otherwise.

    Returns
    -------
    numpy array
        Frequency domain filter, computed at the same frequency values as
        `speech_psd`.
    """

    return speech_psd / (speech_psd + noise_psd)


def apply_iterative_wiener(noisy_signal, frame_len=512, lpc_order=20,
                           iterations=2, alpha=0.8, thresh=0.01):
    """
    One-shot function to apply iterative Wiener filtering for denoising.

    Parameters
    ----------
    noisy_signal : numpy array
        Real signal in time domain.
    frame_len : int
        Frame length in samples. 50% overlap is used with hanning window.
    lpc_order : int
        Number of LPC coefficients to compute
    iterations : int
        How many iterations to perform in updating the Wiener filter for each
        signal frame.
    alpha : int
        Smoothing factor within [0,1] for updating noise level. Closer to `1`
        gives more weight to the previous noise level, while closer to `0`
        gives more weight to the current frame's level. Closer to `0` can track
        more rapid changes in the noise level. However, if a speech frame is
        incorrectly identified as noise, you can end up removing desired
        speech.
    thresh : float
        Threshold to distinguish between (signal+noise) and (noise) frames. A
        high value will classify more frames as noise but might remove desired
        signal!

    Returns
    -------
    numpy array
        Enhanced/denoised signal.
    """

    from pyroomacoustics import hann
    from pyroomacoustics.transform import STFT

    hop = frame_len // 2
    window = hann(frame_len, flag='asymmetric', length='full')
    stft = STFT(frame_len, hop=hop, analysis_window=window, streaming=True)
    scnr = IterativeWiener(frame_len, lpc_order, iterations, alpha, thresh)

    processed_audio = np.zeros(noisy_signal.shape)
    n = 0
    while noisy_signal.shape[0] - n >= hop:
        # SCNR in frequency domain
        stft.analysis(noisy_signal[n:(n + hop), ])
        X = scnr.compute_filtered_output(current_frame=stft.fft_in_buffer,
                                         frame_dft=stft.X)

        # back to time domain
        processed_audio[n:n + hop, ] = stft.synthesis(X)

        # update step
        n += hop

    return processed_audio

