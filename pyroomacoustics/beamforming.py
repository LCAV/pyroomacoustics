# Various Beamforming Methods
# Copyright (C) 2019  Robin Scheibler, Sidney Barthe, Ivan Dokmanic
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

from __future__ import division

import copy

import numpy as np
import scipy.linalg as la

from . import transform
from . import utilities as u
from . import windows
from .directivities import (
    CardioidFamily,
    DirectionVector,
    Directivity,
    DirectivityPattern,
)
from .parameters import constants
from .soundsource import build_rir_matrix

# =========================================================================
# Free (non-class-member) functions related to beamformer design
# =========================================================================


def H(A, **kwargs):
    """Returns the conjugate (Hermitian) transpose of a matrix."""

    return np.transpose(A, **kwargs).conj()


def sumcols(A):
    """
    Sums the columns of a matrix (np.array).
    The output is a 2D np.array
    of dimensions M x 1.
    """

    return np.sum(A, axis=1, keepdims=1)


def mdot(*args):
    """
    Left-to-right associative matrix multiplication of multiple 2D ndarrays.
    """

    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret, a)

    return ret


def distance(x, y):
    """
    Computes the distance matrix E.
    E[i,j] = sqrt(sum((x[:,i]-y[:,j])**2)).
    x and y are DxN ndarray containing N D-dimensional vectors.
    """

    # Assume x, y are arrays, *not* matrices
    x = np.array(x)
    y = np.array(y)

    # return np.sqrt((x[0,:,np.newaxis]-y[0,:])**2 +
    # (x[1,:,np.newaxis]-y[1,:])**2)

    return np.sqrt(np.sum((x[:, :, np.newaxis] - y[:, np.newaxis, :]) ** 2, axis=0))


def unit_vec2D(phi):
    return np.array([[np.cos(phi), np.sin(phi)]]).T


def linear_2D_array(center, M, phi, d):
    """
    Creates an array of uniformly spaced linear points in 2D
    Parameters
    ----------
    center: array_like
        The center of the array
    M: int
        The number of points
    phi: float
        The counterclockwise rotation of the array (from the x-axis)
    d: float
        The distance between neighboring points
    Returns
    -------
    ndarray (2, M)
        The array of points
    """
    u = unit_vec2D(phi)
    return (
        np.array(center)[:, np.newaxis]
        + d * (np.arange(M)[np.newaxis, :] - (M - 1.0) / 2.0) * u
    )


def circular_2D_array(center, M, phi0, radius):
    """
    Create an array of uniformly spaced circular points in 2D.

    Parameters
    ----------
    center: array_like
        The center of the array
    M: int
        The number of points
    phi0: float
        The counterclockwise rotation of the first element in the array (from
        the x-axis)
    radius: float
        The radius of the array

    Returns
    -------
    ndarray (2, M)
        The array of points
    """
    phi = np.arange(M) * 2.0 * np.pi / M
    return np.array(center)[:, np.newaxis] + radius * np.vstack(
        (np.cos(phi + phi0), np.sin(phi + phi0))
    )


def poisson_2D_array(center, M, d):
    """
    Create array of 2D positions drawn from Poisson process.
    Parameters
    ----------
    center: array_like
        The center of the array
    M: int
        The number of points in the first dimension
    M: int
        The number of points in the second dimension
    phi: float
        The counterclockwise rotation of the array (from the x-axis)
    d: float
        The distance between neighboring points
    Returns
    -------
    ndarray (2, M * N)
        The array of points
    """

    from numpy.random import randint, standard_exponential

    R = d * standard_exponential((2, M)) * (2 * randint(0, 2, (2, M)) - 1)
    R = R.cumsum(axis=1)
    R -= R.mean(axis=1)[:, np.newaxis]
    R += np.array([center]).T

    return R


def square_2D_array(center, M, N, phi, d):
    """
    Creates an array of uniformly spaced grid points in 2D
    Parameters
    ----------
    center: array_like
        The center of the array
    M: int
        The number of points in the first dimension
    N: int
        The number of points in the second dimension
    phi: float
        The counterclockwise rotation of the array (from the x-axis)
    d: float
        The distance between neighboring points
    Returns
    -------
    ndarray (2, M * N)
        The array of points
    """

    c = linear_2D_array(center, M, phi + np.pi / 2.0, d)
    R = np.zeros((2, M * N))
    for i in np.arange(M):
        R[:, i * N : (i + 1) * N] = linear_2D_array(c[:, i], N, phi, d)

    return R


def spiral_2D_array(center, M, radius=1.0, divi=3, angle=None):
    """
    Generate an array of points placed on a spiral
    Parameters
    ----------
    center: array_like
        location of the center of the array
    M: int
        number of microphones
    radius: float
        microphones are contained within a cirle of this radius (default 1)
    divi: int
        number of rotations of the spiral (default 3)
    angle: float
        the angle offset of the spiral (default random)
    Returns
    -------
    ndarray (2, M * N)
        The array of points
    """
    num_seg = int(np.ceil(M / divi))

    pos_array_norm = np.linspace(0, radius, num=M, endpoint=False)

    pos_array_angle = (
        np.reshape(
            np.tile(np.pi * 2 * np.arange(divi) / divi, num_seg), (divi, -1), order="F"
        )
        + np.linspace(0, 2 * np.pi / divi, num=num_seg, endpoint=False)[np.newaxis, :]
    )
    pos_array_angle = np.insert(pos_array_angle.flatten("F")[: M - 1], 0, 0)

    if angle is None:
        pos_array_angle += np.random.rand() * np.pi / divi
    else:
        pos_array_angle += angle

    pos_mic_x = pos_array_norm * np.cos(pos_array_angle)
    pos_mic_y = pos_array_norm * np.sin(pos_array_angle)

    return np.array([pos_mic_x, pos_mic_y])


def fir_approximation_ls(weights, T, n1, n2):

    freqs_plus = np.array(weights.keys())[:, np.newaxis]
    freqs = np.vstack([freqs_plus, -freqs_plus])
    omega = 2 * np.pi * freqs
    omega_discrete = omega * T

    n = np.arange(n1, n2)

    # Create the DTFT transform matrix corresponding to a discrete set of
    # frequencies and the FIR filter indices
    F = np.exp(-1j * omega_discrete * n)

    w_plus = np.array(weights.values())[:, :, 0]
    w = np.vstack([w_plus, w_plus.conj()])

    return np.linalg.pinv(F).dot(w)


def circular_microphone_array_xyplane(
    center, M, phi0, radius, fs, directivity=None, ax=None
):
    """
    Create a microphone array with directivities pointing outwards (if provided).

    Parameters
    ----------
    center: array_like
        The center of the microphone array. 2D or 3D.
    M: int
        The number of microphones.
    phi0: float
        The counterclockwise rotation (in degrees) of the first element in the microphone array
        (from the x-axis).
    radius: float
        The radius of the microphone array.
    fs: int
        The sampling frequency.
    directivity: Directivity object, optional.
        Directivity pattern for each microphone which will be re-oriented to face outward. If not
        provided, microphones are omnidirectional.
    ax: axes object, optional
        Axes on which to plot microphone array with its directivities.

    Returns
    -------
    MicrophoneArray object
    """

    R = circular_2D_array(center=center[:1], M=M, phi0=phi0, radius=radius)
    if len(center) == 3:
        colatitude = 90
        R = np.concatenate((R, np.ones((1, M)) * center[2]))
    else:
        colatitude = None

    if directivity is not None:
        assert isinstance(directivity, Directivity)
    else:
        orientation = DirectionVector(azimuth=0, colatitude=colatitude, degrees=True)
        directivity = CardioidFamily(
            orientation=orientation, pattern_enum=DirectivityPattern.OMNI
        )

    # for plotting
    azimuth_plot = None
    colatitude_plot = None
    if ax is not None:
        azimuth_plot = np.linspace(start=0, stop=360, num=361, endpoint=True)
        if colatitude is not None:
            colatitude_plot = np.linspace(start=0, stop=180, num=180, endpoint=True)

    azimuth_list = np.arange(M) * 360 / M + phi0
    directivity_list = []
    for i in range(M):

        orientation = DirectionVector(
            azimuth=azimuth_list[i], colatitude=colatitude, degrees=True
        )
        directivity.set_orientation(orientation)
        directivity_list.append(copy.copy(directivity))

        if ax is not None:
            ax = directivity.plot_response(
                azimuth=azimuth_plot,
                colatitude=colatitude_plot,
                degrees=True,
                ax=ax,
                offset=R[:, i],
            )

    return MicrophoneArray(R, fs, directivity_list)


# =========================================================================
# Classes (microphone array and beamformer related)
# =========================================================================


class MicrophoneArray(object):

    """Microphone array class."""

    def __init__(self, R, fs, directivity=None):

        R = np.array(R)
        self.dim = R.shape[0]  # are we in 2D or in 3D
        self.nmic = R.shape[1]  # number of microphones

        # Check the shape of the passed array
        if self.dim != 2 and self.dim != 3:
            dim_mismatch = True
        else:
            dim_mismatch = False

        if R.ndim != 2 or dim_mismatch:
            raise ValueError(
                "The location of microphones should be described by an array_like "
                "object with 2 dimensions of shape `(2 or 3, n_mics)` "
                "where `n_mics` is the number of microphones. Each column contains "
                "the location of a microphone."
            )

        self.R = R  # array geometry

        self.fs = fs  # sampling frequency of microphones
        self.directivity = None

        if directivity is not None:
            self.set_directivity(directivity)

        self.signals = None

        self.center = np.mean(R, axis=1, keepdims=True)

    def set_directivity(self, directivities):

        """
        This functions sets self.directivity as a list of directivities with `n_mics` entries,
        where `n_mics` is the number of microphones
        Parameters
        -----------
        directivities:
            single directivity for all microphones or a list of directivities for each microphone

        """

        if isinstance(directivities, list):
            # list of directivities specified
            assert all(isinstance(x, Directivity) for x in directivities)
            assert len(directivities) == self.nmic
            self.directivity = directivities
        else:
            # only 1 directivity specified
            assert isinstance(directivities, Directivity)
            self.directivity = [directivities] * self.nmic

    def record(self, signals, fs):
        """
        This simulates the recording of the signals by the microphones.
        In particular, if the microphones and the room simulation
        do not use the same sampling frequency, down/up-sampling
        is done here.
        Parameters
        ----------
        signals:
            An ndarray with as many lines as there are microphones.
        fs:
            the sampling frequency of the signals.
        """

        if signals.shape[0] != self.M:
            raise NameError(
                "The signals array should have as many lines as "
                "there are microphones."
            )

        if signals.ndim != 2:
            raise NameError("The signals should be a 2D array.")

        if fs != self.fs:
            try:
                import samplerate

                fs_ratio = self.fs / float(fs)
                newL = int(fs_ratio * signals.shape[1]) - 1
                self.signals = np.zeros((self.M, newL))
                # samplerate resample function considers columns as channels
                # (hence the transpose)
                for m in range(self.M):
                    self.signals[m] = samplerate.resample(
                        signals[m], fs_ratio, "sinc_best"
                    )
            except ImportError:
                raise ImportError(
                    "The samplerate package must be installed for"
                    " resampling of the signals."
                )

        else:
            self.signals = signals

    def to_wav(self, filename, mono=False, norm=False, bitdepth=float):
        """
        Save all the signals to wav files.

        Parameters
        ----------
        filename: str
            the name of the file
        mono: bool, optional
            if true, records only the center channel floor(M / 2) (default
            `False`)
        norm: bool, optional
            if true, normalize the signal to fit in the dynamic range (default
            `False`)
        bitdepth: int, optional
            the format of output samples [np.int8/16/32/64 or np.float
            (default)]
        """
        from scipy.io import wavfile

        if mono is True:
            signal = self.signals[self.M // 2]
        else:
            signal = self.signals.T  # each column is a channel

        float_types = [float, np.float32, np.float64]

        if bitdepth in float_types:
            bits = None
        elif bitdepth is np.int8:
            bits = 8
        elif bitdepth is np.int16:
            bits = 16
        elif bitdepth is np.int32:
            bits = 32
        elif bitdepth is np.int64:
            bits = 64
        else:
            raise NameError("No such type.")

        if norm:
            from .utilities import normalize

            signal = normalize(signal, bits=bits)

        signal = np.array(signal, dtype=bitdepth)

        wavfile.write(filename, self.fs, signal)

    def append(self, locs):
        """
        Add some microphones to the array.

        Parameters
        ----------
        locs: numpy.ndarray (2 or 3, n_mics)
            Adds `n_mics` microphones to the array. The coordinates are passed as
            a `numpy.ndarray` with each column containing the coordinates of a
            microphone.
        """
        if isinstance(locs, MicrophoneArray):
            self.R = np.concatenate((self.R, locs.R), axis=1)
        else:
            self.R = np.concatenate((self.R, locs), axis=1)

        # in case there was already some signal recorded, just pad with zeros
        if self.signals is not None:
            self.signals = np.concatenate(
                (
                    self.signals,
                    np.zeros(
                        (locs.shape[1], self.signals.shape[1]), dtype=self.signals.dtype
                    ),
                ),
                axis=0,
            )

    def __len__(self):
        return self.R.shape[1]

    @property
    def M(self):
        return self.__len__()


class Beamformer(MicrophoneArray):
    """
    At some point, in some nice way, the design methods
    should also go here. Probably with generic arguments.

    Parameters
    ----------
    R: numpy.ndarray
        Mics positions
    fs: int
        Sampling frequency
    N: int, optional
        Length of FFT, i.e. number of FD beamforming weights, equally spaced.
        Defaults to 1024.
    Lg: int, optional
        Length of time-domain filters. Default to N.
    hop: int, optional
        Hop length for frequency domain processing. Default to N/2.
    zpf: int, optional
        Front zero padding length for frequency domain processing. Default is 0.
    zpb: int, optional
        Zero padding length for frequency domain processing. Default is 0.
    """

    def __init__(self, R, fs, N=1024, Lg=None, hop=None, zpf=0, zpb=0):
        MicrophoneArray.__init__(self, R, fs)

        # only support even length (in freq)
        if N % 2 == 1:
            N += 1

        self.N = int(N)  # FFT length

        if Lg is None:
            self.Lg = N  # TD filters length
        else:
            self.Lg = int(Lg)

        # setup lengths for FD processing
        self.zpf = int(zpf)
        self.zpb = int(zpb)
        self.L = self.N - self.zpf - self.zpb
        if hop is None:
            self.hop = self.L // 2
        else:
            self.hop = hop

        # for now only support equally spaced frequencies
        self.frequencies = np.arange(0, self.N // 2 + 1) / self.N * float(self.fs)

        # weights will be computed later, the array is of shape (M, N/2+1)
        self.weights = None

        # the TD beamforming filters (M, Lg)
        self.filters = None

    def __add__(self, y):
        """Concatenates two beamformers together."""

        newR = np.concatenate((self.R, y.R), axis=1)
        return Beamformer(
            newR, self.fs, self.Lg, self.N, hop=self.hop, zpf=self.zpf, zpb=self.zpb
        )

    def filters_from_weights(self, non_causal=0.0):
        """
        Compute time-domain filters from frequency domain weights.
        Parameters
        ----------
        non_causal: float, optional
            ratio of filter coefficients used for non-causal part
        """

        if self.weights is None:
            raise NameError("Weights must be defined.")

        self.filters = np.zeros((self.M, self.Lg))

        if self.N <= self.Lg:

            # go back to time domain and shift DC to center
            tw = np.fft.irfft(np.conj(self.weights), axis=1, n=self.N)
            self.filters[:, : self.N] = np.concatenate(
                (tw[:, -self.N // 2 :], tw[:, : self.N // 2]), axis=1
            )

        elif self.N > self.Lg:

            # Least-square projection
            for i in np.arange(self.M):
                Lgp = np.floor((1 - non_causal) * self.Lg)
                Lgm = self.Lg - Lgp
                # the beamforming weights in frequency are the complex
                # conjugates of the FT of the filter
                w = np.concatenate((np.conj(self.weights[i]), self.weights[i, -2:0:-1]))

                # create partial Fourier matrix
                k = np.arange(self.N)[:, np.newaxis]
                l = np.concatenate((np.arange(self.N - Lgm, self.N), np.arange(Lgp)))
                F = np.exp(-2j * np.pi * k * l / self.N)

                self.filters[i] = np.real(np.linalg.lstsq(F, w, rcond=None)[0])

    def weights_from_filters(self):

        if self.filters is None:
            raise NameError("Filters must be defined.")

        # this is what we want to use, really.
        # self.weights = np.conj(np.fft.rfft(self.filters, n=self.N, axis=1))

        # quick hack to be able to use MKL acceleration package from anaconda
        self.weights = np.zeros((self.M, self.N // 2 + 1), dtype=np.complex128)
        for m in range(self.M):
            self.weights[m] = np.conj(np.fft.rfft(self.filters[m], n=self.N))

    def steering_vector_2D(self, frequency, phi, dist, attn=False):

        phi = np.array([phi]).reshape(phi.size)

        # Assume phi and dist are measured from the array's center
        X = dist * np.array([np.cos(phi), np.sin(phi)]) + self.center

        D = distance(self.R, X)
        omega = 2 * np.pi * frequency

        if attn:
            # TO DO 1: This will mean slightly different absolute value for
            # every entry, even within the same steering vector. Perhaps a
            # better paradigm is far-field with phase carrier.
            return 1.0 / (4 * np.pi) / D * np.exp(-1j * omega * D / constants.get("c"))
        else:
            return np.exp(-1j * omega * D / constants.get("c"))

    def steering_vector_2D_from_point(self, frequency, source, attn=True, ff=False):
        """Creates a steering vector for a particular frequency and source
        Args:
            frequency
            source: location in cartesian coordinates
            attn: include attenuation factor if True
            ff:   uses far-field distance if true
        Return:
            A 2x1 ndarray containing the steering vector.
        """

        X = np.array(source)
        if X.ndim == 1:
            X = source[:, np.newaxis]

        omega = 2 * np.pi * frequency

        # normalize for far-field if requested
        if ff:
            # unit vectors pointing towards sources
            p = X - self.center
            p /= np.linalg.norm(p)

            # The projected microphone distances on the unit vectors
            D = -1 * np.dot(self.R.T, p)

            # subtract minimum in each column
            D -= np.min(D)

        else:

            D = distance(self.R, X)

        phase = np.exp(-1j * omega * D / constants.get("c"))

        if attn:
            # TO DO 1: This will mean slightly different absolute value for
            # every entry, even within the same steering vector. Perhaps a
            # better paradigm is far-field with phase carrier.
            return 1.0 / (4 * np.pi) / D * phase
        else:
            return phase

    def response(self, phi_list, frequency):

        i_freq = np.argmin(np.abs(self.frequencies - frequency))

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError(
                "Beamforming weights or filters need to be computed" " first."
            )

        # For the moment assume that we are in 2D
        bfresp = np.dot(
            H(self.weights[:, i_freq]),
            self.steering_vector_2D(
                self.frequencies[i_freq], phi_list, constants.get("ffdist")
            ),
        )

        return self.frequencies[i_freq], bfresp

    def response_from_point(self, x, frequency):

        i_freq = np.argmin(np.abs(self.frequencies - frequency))

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError(
                "Beamforming weights or filters need to be computed" " first."
            )

        # For the moment assume that we are in 2D
        bfresp = np.dot(
            H(self.weights[:, i_freq]),
            self.steering_vector_2D_from_point(
                self.frequencies[i_freq], x, attn=True, ff=False
            ),
        )

        return self.frequencies[i_freq], bfresp

    def plot_response_from_point(self, x, legend=None):

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError(
                "Beamforming weights or filters need to be computed" " first."
            )

        if x.ndim == 0:
            x = np.array([x])

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings

            warnings.warn("Matplotlib is required for plotting")
            return

        HF = np.zeros((x.shape[1], self.frequencies.shape[0]), dtype=complex)
        for k, p in enumerate(x.T):
            for i, f in enumerate(self.frequencies):
                r = np.dot(
                    H(self.weights[:, i]),
                    self.steering_vector_2D_from_point(f, p, attn=True, ff=False),
                )
                HF[k, i] = r[0]

        plt.subplot(2, 1, 1)
        plt.title("Beamformer response")
        for hf in HF:
            plt.plot(self.frequencies, np.abs(hf))
        plt.ylabel("Modulus")
        plt.axis("tight")
        plt.legend(legend)

        plt.subplot(2, 1, 2)
        for hf in HF:
            plt.plot(self.frequencies, np.unwrap(np.angle(hf)))
        plt.ylabel("Phase")
        plt.xlabel("Frequency [Hz]")
        plt.axis("tight")
        plt.legend(legend)

    def plot_beam_response(self):

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError("Beamforming weights or filters need to be computed first.")

        phi = np.linspace(-np.pi, np.pi - np.pi / 180, 360)
        freq = self.frequencies

        resp = np.zeros((freq.shape[0], phi.shape[0]), dtype=complex)

        for i, f in enumerate(freq):
            # For the moment assume that we are in 2D
            resp[i, :] = np.dot(
                H(self.weights[:, i]),
                self.steering_vector_2D(f, phi, constants.get("ffdist")),
            )

        H_abs = np.abs(resp) ** 2
        H_abs /= H_abs.max()
        H_abs = 10 * np.log10(H_abs + 1e-10)

        p_min = 0
        p_max = 100
        vmin, vmax = np.percentile(H_abs.flatten(), [p_min, p_max])

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings

            warnings.warn("Matplotlib is required for plotting")
            return

        plt.imshow(
            H_abs,
            aspect="auto",
            origin="lower",
            interpolation="sinc",
            vmax=vmax,
            vmin=vmin,
        )

        plt.xlabel("Angle [rad]")
        xticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        for i, p in enumerate(xticks):
            xticks[i] = np.argmin(np.abs(p - phi))
        xticklabels = [r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"]
        plt.setp(plt.gca(), "xticks", xticks)
        plt.setp(plt.gca(), "xticklabels", xticklabels)

        plt.ylabel("Freq [kHz]")
        yticks = np.zeros(4)
        f_0 = np.floor(self.fs / 8000.0)
        for i in np.arange(1, 5):
            yticks[i - 1] = np.argmin(np.abs(freq - 1000.0 * i * f_0))
        # yticks = np.array(plt.getp(plt.gca(), 'yticks'), dtype=np.int)
        plt.setp(plt.gca(), "yticks", yticks)
        plt.setp(plt.gca(), "yticklabels", np.arange(1, 5) * f_0)

    def snr(self, source, interferer, f, R_n=None, dB=False):

        i_f = np.argmin(np.abs(self.frequencies - f))

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError(
                "Beamforming weights or filters need to be computed" " first."
            )

        # This works at a single frequency because otherwise we need to pass
        # many many covariance matrices. Easy to change though (you can also
        # have frequency independent R_n).

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        # To compute the SNR, we /must/ use the real steering vectors, so no
        # far field, and attn=True
        A_good = self.steering_vector_2D_from_point(
            self.frequencies[i_f], source.images, attn=True, ff=False
        )

        if interferer is not None:
            A_bad = self.steering_vector_2D_from_point(
                self.frequencies[i_f], interferer.images, attn=True, ff=False
            )
            R_nq = R_n + sumcols(A_bad) * H(sumcols(A_bad))
        else:
            R_nq = R_n

        w = self.weights[:, i_f]
        a_1 = sumcols(A_good)

        SNR = np.real(mdot(H(w), a_1, H(a_1), w) / mdot(H(w), R_nq, w))

        if dB is True:
            SNR = 10 * np.log10(SNR)

        return SNR

    def udr(self, source, interferer, f, R_n=None, dB=False):

        i_f = np.argmin(np.abs(self.frequencies - f))

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError(
                "Beamforming weights or filters need to be computed" " first."
            )

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        A_good = self.steering_vector_2D_from_point(
            self.frequencies[i_f], source.images, attn=True, ff=False
        )

        if interferer is not None:
            A_bad = self.steering_vector_2D_from_point(
                self.frequencies[i_f], interferer.images, attn=True, ff=False
            )
            R_nq = R_n + sumcols(A_bad).dot(H(sumcols(A_bad)))
        else:
            R_nq = R_n

        w = self.weights[:, i_f]

        UDR = np.real(mdot(H(w), A_good, H(A_good), w) / mdot(H(w), R_nq, w))
        if dB is True:
            UDR = 10 * np.log10(UDR)

        return UDR

    def process(self, FD=False):

        if self.signals is None or len(self.signals) == 0:
            raise NameError("No signal to beamform.")

        if FD is True:

            # STFT processing
            if self.weights is None and self.filters is not None:
                self.weights_from_filters()
            elif self.weights is None and self.filters is None:
                raise NameError(
                    "Beamforming weights or filters need to be " "computed first."
                )

            # create window functions
            analysis_win = windows.hann(self.L)

            # perform STFT
            sig_stft = transform.analysis(
                self.signals.T,
                L=self.L,
                hop=self.hop,
                win=analysis_win,
                zp_back=self.zpb,
                zp_front=self.zpf,
            )

            # beamform
            sig_stft_bf = np.sum(sig_stft * self.weights.conj().T, axis=2)

            # back to time domain
            output = transform.synthesis(
                sig_stft_bf, L=self.L, hop=self.hop, zp_back=self.zpb, zp_front=self.zpf
            )

            # remove the zero padding from output signal
            if self.zpb == 0:
                output = output[self.zpf :]
            else:
                output = output[self.zpf : -self.zpb]

        else:

            # TD processing

            if self.weights is not None and self.filters is None:
                self.filters_from_weights()
            elif self.weights is None and self.filters is None:
                raise NameError(
                    "Beamforming weights or filters need to be " "computed first."
                )

            from scipy.signal import fftconvolve

            # do real STFT of first signal
            output = fftconvolve(self.filters[0], self.signals[0])
            for i in range(1, len(self.signals)):
                output += fftconvolve(self.filters[i], self.signals[i])

        return output

    def plot(self, sum_ir=False, FD=True):

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is not None and self.filters is None:
            self.filters_from_weights()
        elif self.weights is None and self.filters is None:
            raise NameError(
                "Beamforming weights or filters need to be " "computed first."
            )

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings

            warnings.warn("Matplotlib is required for plotting")
            return

        if FD is True:
            plt.subplot(2, 2, 1)
            plt.plot(self.frequencies, np.abs(self.weights.T))
            plt.title("Beamforming weights [modulus]")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Weight modulus")

            plt.subplot(2, 2, 2)
            plt.plot(self.frequencies, np.unwrap(np.angle(self.weights.T), axis=0))
            plt.title("Beamforming weights [phase]")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Unwrapped phase")

            plt.subplot(2, 1, 2)

        plt.plot(np.arange(self.Lg) / float(self.fs), self.filters.T)

        plt.title("Beamforming filters")
        plt.xlabel("Time [s]")
        plt.ylabel("Filter amplitude")
        plt.axis("tight")
        plt.tight_layout(pad=0.1)

    def far_field_weights(self, phi):
        """
        This method computes weight for a far field at infinity
        phi: direction of beam
        """

        u = unit_vec2D(phi)
        proj = np.dot(u.T, self.R - self.center)[0]

        # normalize the first arriving signal to ensure a causal filter
        proj -= proj.max()

        self.weights = np.exp(
            2j * np.pi * self.frequencies[:, np.newaxis] * proj / constants.get("c")
        ).T

    def rake_delay_and_sum_weights(
        self, source, interferer=None, R_n=None, attn=True, ff=False
    ):

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        K = source.images.shape[1] - 1

        for i, f in enumerate(self.frequencies):
            W = self.steering_vector_2D_from_point(f, source.images, attn=attn, ff=ff)
            self.weights[:, i] = 1.0 / self.M / (K + 1) * np.sum(W, axis=1)

    def rake_one_forcing_weights(
        self, source, interferer=None, R_n=None, ff=False, attn=True
    ):

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        for i, f in enumerate(self.frequencies):
            if interferer is None:
                A_bad = np.array([[]])
            else:
                A_bad = self.steering_vector_2D_from_point(
                    f, interferer.images, attn=attn, ff=ff
                )

            R_nq = R_n + sumcols(A_bad).dot(H(sumcols(A_bad)))

            A_s = self.steering_vector_2D_from_point(f, source.images, attn=attn, ff=ff)
            R_nq_inv = np.linalg.pinv(R_nq)
            D = np.linalg.pinv(mdot(H(A_s), R_nq_inv, A_s))

            self.weights[:, i] = sumcols(mdot(R_nq_inv, A_s, D))[:, 0]

    def rake_max_sinr_weights(
        self, source, interferer=None, R_n=None, rcond=0.0, ff=False, attn=True
    ):
        """
        This method computes a beamformer focusing on a number of specific
        sources and ignoring a number of interferers.

        Parameters
        ----------
        source : array_like
            source locations
        interferer : array_like
            interferer locations
        """

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        for i, f in enumerate(self.frequencies):

            A_good = self.steering_vector_2D_from_point(
                f, source.images, attn=attn, ff=ff
            )

            if interferer is None:
                A_bad = np.array([[]])
            else:
                A_bad = self.steering_vector_2D_from_point(
                    f, interferer.images, attn=attn, ff=ff
                )

            a_good = sumcols(A_good)
            a_bad = sumcols(A_bad)

            # TO DO: Fix this (check for numerical rank, use the low rank
            # approximation)
            K_inv = np.linalg.pinv(
                a_bad.dot(H(a_bad)) + R_n + rcond * np.eye(A_bad.shape[0])
            )
            self.weights[:, i] = (K_inv.dot(a_good) / mdot(H(a_good), K_inv, a_good))[
                :, 0
            ]

    def rake_max_udr_weights(
        self, source, interferer=None, R_n=None, ff=False, attn=True
    ):

        if source.images.shape[1] == 1:
            self.rake_max_sinr_weights(
                source.images, interferer.images, R_n=R_n, ff=ff, attn=attn
            )
            return

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        for i, f in enumerate(self.frequencies):
            A_good = self.steering_vector_2D_from_point(
                f, source.images, attn=attn, ff=ff
            )

            if interferer is None:
                A_bad = np.array([[]])
            else:
                A_bad = self.steering_vector_2D_from_point(
                    f, interferer.images, attn=attn, ff=ff
                )

            R_nq = R_n + sumcols(A_bad).dot(H(sumcols(A_bad)))

            C = np.linalg.cholesky(R_nq)
            l, v = np.linalg.eig(
                mdot(np.linalg.inv(C), A_good, H(A_good), H(np.linalg.inv(C)))
            )

            self.weights[:, i] = np.linalg.inv(H(C)).dot(v[:, 0])

    def rake_max_udr_filters(
        self, source, interferer=None, R_n=None, delay=0.03, epsilon=5e-3
    ):
        """
        Compute directly the time-domain filters maximizing the
        Useful-to-Detrimental Ratio (UDR).
        This beamformer is not practical. It maximizes the UDR ratio in the time
        domain directly without imposing flat response towards the source of
        interest. This results in severe distortion of the desired signal.

        Parameters
        ----------
        source: pyroomacoustics.SoundSource
            the desired source
        interferer: pyroomacoustics.SoundSource, optional
            the interfering source
        R_n: ndarray, optional
            the noise covariance matrix, it should be (M * Lg)x(M * Lg) where M
            is the number of sensors and Lg the filter length
        delay: float, optional
            the signal delay introduced by the beamformer (default 0.03 s)
        epsilon: float
        """
        if delay > self.Lg / self.fs:
            print("Warning: filter length shorter than beamformer delay")

        if R_n is None:
            R_n = np.zeros((self.M * self.Lg, self.M * self.Lg))

        if interferer is not None:
            H = build_rir_matrix(
                self.R,
                (source, interferer),
                self.Lg,
                self.fs,
                epsilon=epsilon,
                unit_damping=True,
            )
            L = H.shape[1] // 2
        else:
            H = build_rir_matrix(
                self.R, (source,), self.Lg, self.fs, epsilon=epsilon, unit_damping=True
            )
            L = H.shape[1]

        # Delay of the system in samples
        kappa = int(delay * self.fs)
        precedence = int(0.030 * self.fs)

        # the constraint
        n = int(np.minimum(L, kappa + precedence))
        Hnc = H[:, :kappa]
        Hpr = H[:, kappa:n]
        A = np.dot(Hpr, Hpr.T)
        B = np.dot(Hnc, Hnc.T) + np.dot(H[:, L:], H[:, L:].T) + R_n

        if interferer is not None:
            Hc = H[:, n:L]
            B += np.dot(Hc, Hc.T)

        # solve the problem
        SINR, v = la.eigh(
            A,
            b=B,
            eigvals=(self.M * self.Lg - 1, self.M * self.Lg - 1),
            overwrite_a=True,
            overwrite_b=True,
            check_finite=False,
        )
        g_val = np.real(v[:, 0])

        # reshape and store
        self.filters = g_val.reshape((self.M, self.Lg))

        # compute and return SNR
        return SINR[0]

    def rake_perceptual_filters(
        self, source, interferer=None, R_n=None, delay=0.03, d_relax=0.035, epsilon=5e-3
    ):
        """
        Compute directly the time-domain filters for a perceptually motivated
        beamformer. The beamformer minimizes noise and interference, but relaxes
        the response of the filter within the 30 ms following the delay.
        """

        if delay > self.Lg / self.fs:
            print("Warning: filter length shorter than beamformer delay")

        if R_n is None:
            R_n = np.zeros((self.M * self.Lg, self.M * self.Lg))

        # build the channel matrix
        if interferer is not None:
            H = build_rir_matrix(
                self.R,
                (source, interferer),
                self.Lg,
                self.fs,
                epsilon=epsilon,
                unit_damping=True,
            )
            L = H.shape[1] // 2
        else:
            H = build_rir_matrix(
                self.R, (source,), self.Lg, self.fs, epsilon=epsilon, unit_damping=True
            )
            L = H.shape[1]

        # Delay of the system in samples
        tau = int(delay * self.fs)
        kappa = int(d_relax * self.fs)

        # the constraint
        A = np.concatenate((H[:, : tau + 1], H[:, tau + kappa :]), axis=1)
        b = np.zeros((A.shape[1], 1))
        b[tau, 0] = 1

        # We first assume the sample are uncorrelated
        K_nq = R_n
        if interferer is not None:
            K_nq += np.dot(H[:, L:], H[:, L:].T)

        # causal response construction
        C = la.cho_factor(K_nq, overwrite_a=True, check_finite=False)
        B = la.cho_solve(C, A)
        D = np.dot(A.T, B)
        C = la.cho_factor(D, overwrite_a=True, check_finite=False)
        x = la.cho_solve(C, b)
        g_val = np.dot(B, x)

        # reshape and store
        self.filters = g_val.reshape((self.M, self.Lg))

        # compute and return SNR
        A = np.dot(g_val.T, H[:, :L])
        num = np.dot(A, A.T)
        denom = np.linalg.multi_dot([g_val.T, K_nq, g_val])

        return num / denom

    def rake_max_sinr_filters(self, source, interferer, R_n, epsilon=5e-3, delay=0.0):
        """
        Compute the time-domain filters of SINR maximizing beamformer.
        """

        H = build_rir_matrix(
            self.R,
            (source, interferer),
            self.Lg,
            self.fs,
            epsilon=epsilon,
            unit_damping=True,
        )
        L = H.shape[1] / 2

        # We first assume the sample are uncorrelated
        K_s = np.dot(H[:, :L], H[:, :L].T)
        K_nq = np.dot(H[:, L:], H[:, L:].T) + R_n

        # Compute TD filters using generalized Rayleigh coefficient maximization
        SINR, v = la.eigh(
            K_s,
            b=K_nq,
            eigvals=(self.M * self.Lg - 1, self.M * self.Lg - 1),
            overwrite_a=True,
            overwrite_b=True,
            check_finite=False,
        )
        g_val = np.real(v[:, 0])

        self.filters = g_val.reshape((self.M, self.Lg))

        # compute and return SNR
        return SINR[0]

    def rake_distortionless_filters(
        self, source, interferer, R_n, delay=0.03, epsilon=5e-3
    ):
        """
        Compute time-domain filters of a beamformer minimizing noise and
        interference while forcing a distortionless response towards the source.
        """

        H = build_rir_matrix(
            self.R,
            (source, interferer),
            self.Lg,
            self.fs,
            epsilon=epsilon,
            unit_damping=True,
        )
        L = H.shape[1] / 2

        # We first assume the sample are uncorrelated
        K_nq = np.dot(H[:, L:], H[:, L:].T) + R_n

        # constraint
        kappa = int(delay * self.fs)
        A = H[:, :L]
        b = np.zeros((L, 1))
        b[kappa, 0] = 1

        # filter computation
        C = la.cho_factor(K_nq, overwrite_a=True, check_finite=False)
        B = la.cho_solve(C, A)
        D = np.dot(A.T, B)
        C = la.cho_factor(D, overwrite_a=True, check_finite=False)
        x = la.cho_solve(C, b)
        g_val = np.dot(B, x)

        # reshape and store
        self.filters = g_val.reshape((self.M, self.Lg))

        # compute and return SNR
        A = np.dot(g_val.T, H[:, :L])
        num = np.dot(A, A.T)
        denom = np.linalg.multi_dot([g_val.T, K_nq, g_val])

        return num / denom

    def rake_mvdr_filters(self, source, interferer, R_n, delay=0.03, epsilon=5e-3):
        """
        Compute the time-domain filters of the minimum variance distortionless
        response beamformer.
        """

        H = build_rir_matrix(
            self.R,
            (source, interferer),
            self.Lg,
            self.fs,
            epsilon=epsilon,
            unit_damping=True,
        )
        L = H.shape[1] // 2

        # the constraint vector
        kappa = int(delay * self.fs)
        h = H[:, kappa]

        # We first assume the sample are uncorrelated
        R_xx = np.dot(H[:, :L], H[:, :L].T)
        K_nq = np.dot(H[:, L:], H[:, L:].T) + R_n

        # Compute the TD filters
        C = la.cho_factor(R_xx + K_nq, check_finite=False)
        g_val = la.cho_solve(C, h)

        g_val /= np.inner(h, g_val)
        self.filters = g_val.reshape((self.M, self.Lg))

        # compute and return SNR
        num = np.inner(g_val.T, np.dot(R_xx, g_val))
        denom = np.inner(np.dot(g_val.T, K_nq), g_val)

        return num / denom

    def rake_one_forcing_filters(self, sources, interferers, R_n, epsilon=5e-3):
        """
        Compute the time-domain filters of a beamformer with unit response
        towards multiple sources.
        """

        dist_mat = distance(self.R, sources.images)
        s_time = dist_mat / constants.get("c")
        s_dmp = 1.0 / (4 * np.pi * dist_mat)

        dist_mat = distance(self.R, interferers.images)
        i_time = dist_mat / constants.get("c")
        i_dmp = 1.0 / (4 * np.pi * dist_mat)

        # compute offset needed for decay of sinc by epsilon
        offset = np.maximum(s_dmp.max(), i_dmp.max()) / (np.pi * self.fs * epsilon)
        t_min = np.minimum(s_time.min(), i_time.min())
        t_max = np.maximum(s_time.max(), i_time.max())

        # adjust timing
        s_time -= t_min - offset
        i_time -= t_min - offset
        Lh = np.ceil((t_max - t_min + 2 * offset) * float(self.fs))

        # the channel matrix
        K = sources.images.shape[1]
        Lg = self.Lg
        off = (Lg - Lh) / 2
        L = self.Lg + Lh - 1

        H = np.zeros((Lg * self.M, 2 * L))
        As = np.zeros((Lg * self.M, K))

        for r in np.arange(self.M):

            # build constraint matrix
            hs = u.low_pass_dirac(
                s_time[r, :, np.newaxis], s_dmp[r, :, np.newaxis], self.fs, Lh
            )[:, ::-1]
            As[r * Lg + off : r * Lg + Lh + off, :] = hs.T

            # build interferer RIR matrix
            hx = u.low_pass_dirac(
                s_time[r, :, np.newaxis], s_dmp[r, :, np.newaxis], self.fs, Lh
            ).sum(axis=0)
            H[r * Lg : (r + 1) * Lg, :L] = u.convmtx(hx, Lg).T

            # build interferer RIR matrix
            hq = u.low_pass_dirac(
                i_time[r, :, np.newaxis], i_dmp[r, :, np.newaxis], self.fs, Lh
            ).sum(axis=0)
            H[r * Lg : (r + 1) * Lg, L:] = u.convmtx(hq, Lg).T

        ones = np.ones((K, 1))

        # We first assume the sample are uncorrelated
        K_x = np.dot(H[:, :L], H[:, :L].T)
        K_nq = np.dot(H[:, L:], H[:, L:].T) + R_n

        # Compute the TD filters
        K_nq_inv = np.linalg.inv(K_x + K_nq)
        C = np.dot(K_nq_inv, As)
        B = np.linalg.inv(np.dot(As.T, C))
        g_val = np.dot(C, np.dot(B, ones))
        self.filters = g_val.reshape((self.M, Lg))

        # compute and return SNR
        A = np.dot(g_val.T, H[:, :L])
        num = np.dot(A, A.T)
        denom = np.linalg.multi_dot([g_val.T, K_nq, g_val])

        return num / denom
