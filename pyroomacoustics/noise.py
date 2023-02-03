# Abstract and concrete classes for noise types
# Copyright (C) 2022  Robin Scheibler
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

r"""
Noise
=====
"""

import abc
import math
from typing import Optional
import numpy as np
from .parameters import constants
from . import transform
from .windows import hamming
from .utilities import requires_matplotlib


def _dist_matrix(mic_array):
    mic_array = mic_array.T
    return np.linalg.norm(mic_array[:, None, :] - mic_array[None, :, :], axis=-1)


def _evd_factor(C):
    eigval, eigvec = np.linalg.eigh(C)
    eigval = np.maximum(eigval, 0.0)
    return eigvec * np.sqrt(eigval[..., None, :])


def compute_power(signal):
    return np.mean(signal**2)


def compute_rmse(signal):
    return np.sqrt(compute_power(signal))


def compute_snr(signal, noise):
    sig_pwr = np.mean(signal**2)
    noz_pwr = np.mean(noise**2)
    if sig_pwr == 0.0:
        return -np.inf
    elif noz_pwr == 0.0:
        return np.inf
    else:
        return 10.0 * np.log10(sig_pwr / noz_pwr)


def scale_signal(signal, reference, snr):
    r"""
    Scales ``signal`` so that the given ``snr`` with respect to ``reference`` is achieved

    .. math::

        \operatorname{snr} = 10 \log_10 \frac{E |\operatorname{signal}|^2}{E |\operatorname{reference}|^2}

    Parameters
    ----------
    signal: ndarray
        The signal
    noise: ndarray
        The noise
    snr: float
        The signal-to-noise ratio

    Returns
    -------
    The scaled signal
    """
    signal_rmse = compute_rmse(signal)
    ref_rmse = compute_rmse(reference)

    # the default is to cange the size of the noise
    signal_scale = (ref_rmse / signal_rmse) * 10 ** (snr / 20.0)
    return signal * signal_scale


def mix_signal_noise(
    signal: np.ndarray,
    noise: np.ndarray,
    snr: float,
    scale_noise: Optional[bool] = True,
) -> np.ndarray:
    r"""
    Mixes signal and noise according to a given signal-to-noise ratio (SNR)

    .. math::

        \operatorname{snr} = 10 \log_10 \frac{E |\operatorname{signal}|^2}{E |\operatorname{noise}|^2}

    Parameters
    ----------
    signal: ndarray
        The signal
    noise: ndarray
        The noise
    snr: float
        The signal-to-noise ratio
    scale_noise: bool
        When set to ``False``, the scale of the signal is changed to meet the SNR
        requirement. If ``True`` (default), then the noise is scaled.

    Returns
    -------
    An additive mixture of signal and noise that meets the required SNR
    """
    if scale_noise:
        noise = scale_signal(noise, signal, -snr)
    else:
        signal = scale_signal(signal, noise, snr)
    return signal + noise


class Noise(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate(
        self,
        mix: np.ndarray,
        room: Optional["Room"] = None,
        premix: Optional[np.ndarray] = None,
    ):
        raise NotImplementedError


class WhiteNoise(Noise):
    r"""

    Parameters
    ----------
    snr: float
        The desired signal-to-noise ratio.

        .. math::

            \mathsf{SNR} = 10 \log_{10} \frac{ K }{ \sigma_n^2 }

    """

    def __init__(self, snr: float):
        self.snr = snr

    def generate_fn(self, signal: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*signal.shape)
        return scale_signal(noise, signal, snr=-self.snr)

    def generate(
        self,
        mix: np.ndarray,
        room: Optional["Room"] = None,
        premix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.generate_fn(mix)


class DiffuseNoise(Noise):
    def __init__(
        self,
        snr: float,
        signal: Optional[np.ndarray] = None,
        padding: Optional[str] = "repeat",
        n_fft: Optional[int] = 512,
        hop: Optional[int] = 128,
        use_cholesky: Optional[bool] = False,
        smooth_filters: Optional[bool] = True,
    ):
        """
        Generates spherical diffuse noise according to the method described in [1], [2].

        [1] E. A. P. Habets, I. Cohen, and S. Gannot, “Generating
        nonstationary multisensor signals under a spatial coherence
        constraint,” The Journal of the Acoustical Society of America, vol.
        124, no. 5, pp. 2911–2917, Nov. 2008, doi: 10.1121/1.2987429.

        [2] D. Mirabilii, S. J. Schlecht, and E. A. P. Habets, “Generating
        coherence-constrained multisensor signals using balanced mixing and
        spectrally smooth filters,” The Journal of the Acoustical Society of
        America, vol. 149, no. 3, pp. 1425–1433, Mar. 2021, doi: 10.1121/10.0003565.

        Parameters
        ----------
        mic_array: (n_dim, n_mics)
            The locations of the sensors (in meters)
        n_samples: int
            The length of the target signal
        signal: numpy.ndarray
            A template noise signal
        padding: none | repeat | reflect
            How to pad signal to the target length. 'none': zero padding,
            'repeat': repeat signal, 'reflect': repeat a reflected copy of
            signal
        n_fft: int
            The length of the fft in the STFT
        hop: int
            The shift of the STFT
        use_cholesky: bool, optional
            If set to ``True``, a Cholesky decomposition is used instead of
            the eigenvalue decomposition (default: ``False``), as suggested in [2].
        smooth_filters: bool, optional
            If set to ``True`` (default), the filters are smoothed over frequencies as
            proposed in [2].

        Returns
        -------
        The diffuse noise signal with shape (n_samples, n_mics)
        """
        self.snr = snr
        self.signal = signal
        self.padding = padding

        # diffusion generation parameters
        self.use_cholesky = use_cholesky
        self.smooth_filters = smooth_filters

        # STFT params
        self.n_fft = n_fft
        self.hop = hop
        self.win_a = hamming(self.n_fft)
        self.win_s = transform.stft.compute_synthesis_window(self.win_a, self.hop)

        if self.padding not in ["none", "repeat", "reflect"]:
            raise ValueError(
                "The value provided for padding must be one of 'none'|'repeat'|'reflect'"
            )

        if self.signal is not None and self.signal.ndim > 1:
            self.signal = self.signal[:, 0]

    def _stft(self, x):
        mono = x.ndim == 2 and x.shape[-1] == 1
        if mono:
            x = x[..., 0]
        out = transform.stft.analysis(x, self.n_fft, self.hop, win=self.win_a)
        if mono:
            out = out[..., None]
        return out

    def _istft(self, x):
        mono = x.ndim == 3 and x.shape[-1] == 1
        if mono:
            x = x[..., 0]
        out = transform.stft.synthesis(x, self.n_fft, self.hop, win=self.win_s)
        if mono:
            out = out[..., None]
        return out

    def _get_padded_signal(self, n_samples):
        slen = self.signal.shape[0]
        if slen >= n_samples:
            return self.signal[:n_samples]

        elif slen < n_samples:

            if self.padding == "none":
                signal = np.concatenate((self.signal, np.zeros(n_samples - slen)))

            elif self.padding == "repeat":
                rep = math.ceil(n_samples / slen)
                signal = np.concatenate([self.signal] * rep)
                signal = signal[:n_samples]

            elif self.padding == "reflect":

                sigs = []
                for i in range(math.ceil(n_samples / slen)):
                    if i % 2 == 0:
                        sigs.append(self.signal)
                    else:
                        sigs.append(self.signal[::-1])
                signal = np.concatenate(sigs)
                signal = signal[:n_samples]

            else:
                raise ValueError(
                    "The value provided for padding must be one of "
                    "'none'|'repeat'|'reflect'"
                )

            return signal

    def compute_coherence_theory(self, mic_array, fs, c):
        distance = _dist_matrix(mic_array)
        omega = 2.0 * np.pi * np.arange(self.n_fft // 2 + 1) / self.n_fft * fs
        coh = np.sinc(omega[:, None, None] / c * distance[None, :, :])
        return coh

    def compute_coherence_empirical(self, noise):
        """used in tests"""
        # empirical coherence
        N = self._stft(noise.T)
        coh_data = np.einsum("nfc,nfd->fcd", N, N.conj()) / N.shape[0]

        coh_diag = np.sqrt(abs(np.diagonal(coh_data, axis1=-2, axis2=-1)))
        coh_norm = coh_data / (coh_diag[..., :, None] * coh_diag[..., None, :])

        return coh_norm

    def _shape_noise_envelope(self, N, n_samples):

        if self.signal is not None:
            padded_signal = self._get_padded_signal(n_samples)
            A = self._stft(padded_signal)

            l_max = min(N.shape[0], A.shape[0])
            N[:l_max, :, :] *= abs(A[:l_max, :, None])

        return N

    def _spectral_smoothing(self, C):
        C = C.swapaxes(-2, -1).conj()
        C_out = [C[0]]
        for f in range(1, C.shape[0]):
            R = C_out[-1] @ C[f].conj()
            U, s, V = np.linalg.svd(R)
            C_out.append(U @ V @ C[f])
        C = np.stack(C_out, axis=0)
        C = C.swapaxes(-2, -1).conj()
        return C

    def make_filters(self, mic_array, fs, c):
        # compute the coherence matrix
        coh = self.compute_coherence_theory(mic_array, fs, c)

        if self.use_cholesky:
            sm0 = _evd_factor(coh[:1])  # can't use Cholesky for freq 0
            sm1 = np.linalg.cholesky(coh[1:])
            shaping_matrix = np.concatenate([sm0, sm1], axis=0)
        else:
            shaping_matrix = _evd_factor(coh)

        if self.smooth_filters:
            shaping_matrix = self._spectral_smoothing(shaping_matrix)

        return shaping_matrix

    def generate_fn(
        self,
        signal: np.ndarray,
        mic_array: np.ndarray,
        fs: float,
        c: float,
    ):
        n_mics = mic_array.shape[1]
        n_samples = signal.shape[1]

        shaping_matrix = self.make_filters(mic_array, fs, c)

        # generate the sensor noise (n_frames, n_freq, n_chan)
        shape = ((n_samples + self.n_fft) // self.hop, self.n_fft // 2 + 1, n_mics)
        N = np.random.randn(*shape) + 1j * np.random.randn(*shape)

        # apply the envelope shape of the noise signal, if provided
        N = self._shape_noise_envelope(N, n_samples)

        # shape the diffuse noise and bring back to time domain
        diffuse_fd = np.einsum("fcd,nfd->nfc", shaping_matrix, N)
        diffuse = self._istft(diffuse_fd)
        diffuse = diffuse[:n_samples, :].T

        # scale the noise to achieve the target snr
        diffuse = scale_signal(diffuse, signal, snr=-self.snr)

        return diffuse

    def generate(
        self,
        mix: np.ndarray,
        room: Optional["Room"] = None,
        premix: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        if room is None:
            raise ValueError(
                "The room argument is required for DiffuseNoise "
                "because the microphone locations are necessary"
            )

        # get the necessary parameters from the room
        mic_array = room.mic_array.R
        fs = room.fs
        c = constants.get("c")
        n_samples = mix.shape[-1]

        return self.generate_fn(mix, mic_array, fs, c)


class WindNoise:
    """
    Placeholder to implement this later.

    References
    - https://www.audiolabs-erlangen.de/fau/professor/habets/software/noise-generators
    - https://github.com/ehabets/Wind-Generator
    """

    def __init__(self):
        raise NotImplementedError

    def add(
        self,
        mix: np.ndarray,
        room: Optional["Room"] = None,
        premix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError
