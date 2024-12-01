# -*- coding: utf-8 -*-
#
# This file contains code related to acoustics (critical / octave bands, etc)
# Copyright (C) 2019  Robin Scheibler
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

import abc
import dataclasses
import itertools
import math
from typing import List

import numpy as np
from scipy.fftpack import dct
from scipy.interpolate import interp1d
from scipy.signal import butter, fftconvolve, hilbert

from .parameters import constants
from .transform import stft


def binning(S, bands):
    """
    This function computes the sum of all columns of S in the subbands
    enumerated in bands
    """
    B = np.zeros((S.shape[0], len(bands)), dtype=S.dtype)
    for i, b in enumerate(bands):
        B[:, i] = np.mean(S[:, b[0] : b[1]], axis=1)

    return B


def bandpass_filterbank(bands, fs=1.0, order=8, output="sos"):
    """
    Create a bank of Butterworth bandpass filters

    Parameters
    ----------
    bands: array_like, shape == (n, 2)
        The list of bands ``[[flo1, fup1], [flo2, fup2], ...]``
    fs: float, optional
        Sampling frequency (default 1.)
    order: int, optional
        The order of the IIR filters (default: 8)
    output: {'ba', 'zpk', 'sos'}
        Type of output: numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba'.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (b) and denominator (a) polynomials of the IIR filter. Only
        returned if output='ba'.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer function. Only
        returned if output='zpk'.
    sos : ndarray
        Second-order sections representation of the IIR filter. Only returned
        if output=='sos'.
    """

    filters = []
    nyquist = fs / 2.0

    for band in bands:
        # remove bands above nyquist frequency
        if band[0] >= nyquist:
            raise ValueError("Bands should be below Nyquist frequency")

        # Truncate the highest band to Nyquist frequency
        norm_band = np.minimum(0.99, np.array(band) / nyquist)

        # Compute coefficients
        coeffs = butter(order / 2, norm_band, "bandpass", output=output)
        filters.append(coeffs)

    return filters


def octave_bands(fc=1000, third=False, start=0.0, n=8):
    """
    Create a bank of octave bands

    Parameters
    ----------
    fc : float, optional
        The center frequency
    third : bool, optional
        Use third octave bands (default False)
    start : float, optional
        Starting frequency for octave bands in Hz (default 0.)
    n : int, optional
        Number of frequency bands (default 8)
    """

    div = 1
    if third:
        div = 3

    # Octave Bands
    fcentre = fc * (
        2.0 ** (np.arange(start * div, (start + n) * div - (div - 1)) / div)
    )
    fd = 2 ** (0.5 / div)
    bands = np.array([[f / fd, f * fd] for f in fcentre])

    return bands, fcentre


def magnitude_response_to_minimum_phase(magnitude_response, n_fft, axis=-1, eps=1e-5):
    """
    Creates a minimum phase filter from its magnitude response following
    the method proposed here.
    https://ccrma.stanford.edu/~jos/sasp/Minimum_Phase_Filter_Design.html

    Parameters
    ----------
    magnitude_response: np.ndarray
        The response
    n_fft: int
        The FFT size to use
    axis: int
        The axis where to make the transformation

    Returns
    -------
    The minimum phase impulse response with given magnitude response.
    """
    magnitude_response = np.moveaxis(magnitude_response, axis, -1)

    n_freq = n_fft // 2 + 1
    if n_fft % 2 == 0:
        padding = n_fft - 2 * (magnitude_response.shape[-1] - 1)
    else:
        padding = n_fft - 2 * (magnitude_response.shape[-1] - 1) - 1

    if padding < 0:
        raise ValueError(
            "The FFT size should at least twice the frequency response size."
        )

    zero_pad = np.zeros(
        magnitude_response.shape[:-1] + (padding,), dtype=magnitude_response.dtype
    )
    freq_resp = np.concatenate(
        (magnitude_response, zero_pad, magnitude_response[..., :0:-1]), axis=-1
    )

    freq_resp = np.maximum(freq_resp, eps)
    m_p = np.imag(-hilbert(np.log(freq_resp), axis=-1))
    freq_resp = freq_resp[..., :n_freq] * np.exp(1j * m_p[..., :n_freq])

    freq_resp = np.moveaxis(freq_resp, -1, axis)
    filters = np.fft.irfft(freq_resp, n=n_fft, axis=axis)
    return filters


def cosine_magnitude_octave_filter_response(n_fft, centers, fs, keep_dc=True):
    """
    Creates the magnitude response of a cosine octave-band filterbank as
    described in D. Schroeder's PhD thesis.
    """

    # This seems to work only for Octave bands out of the box
    n = len(centers)

    new_bands = [[centers[0] / 2, centers[1]]]

    for i in range(1, n - 1):
        new_bands.append([centers[i - 1], centers[i + 1]])
    new_bands.append([centers[-2], fs / 2])

    n_freq = n_fft // 2 + 1
    mag_resp = np.zeros((n_freq, n))

    freq = np.arange(n_freq) / n_fft * fs  # This only contains positive newfrequencies

    for b, (band, center) in enumerate(zip(new_bands, centers)):
        lo = np.logical_and(band[0] <= freq, freq < center)

        mag_resp[lo, b] = 0.5 * (1 + np.cos(2 * np.pi * freq[lo] / center))

        if b == 0 and keep_dc:
            # Converting Octave bands so that the minimum phase filters do not
            # have ripples.
            make_one = freq < center
            mag_resp[make_one, b] = 1.0

        if b != n - 1:
            hi = np.logical_and(center <= freq, freq < band[1])
            mag_resp[hi, b] = 0.5 * (1 - np.cos(2 * np.pi * freq[hi] / band[1]))
        else:
            hi = center <= freq
            mag_resp[hi, b] = 1.0

    n_freq = n_fft // 2 + 1
    km = np.round(centers / fs * n_fft).astype(int)
    k1, k2 = [], []
    freq = np.arange(mag_resp.shape[0])
    for band in range(mag_resp.shape[1]):
        f_nz = freq[mag_resp[:, band] > 0]
        k1.append(f_nz[0])
        k2.append(f_nz[-1] + 1)
    k1 = np.array(k1)
    k2 = np.array(k2)

    return mag_resp.T, n_freq, k1, km, k2


def antoni_magnitude_octave_filter_response(
    n_fft, centers, bands, fs, overlap_ratio, slope
):
    """
    Implementation adapted from
    https://github.com/pyfar/pyfar/blob/main/pyfar/dsp/filter/fractional_octaves.py#L339
    MIT License.
    """
    n_freq = n_fft // 2 + 1

    # Discretize the bands boundaries.
    k1 = np.round(bands[:, 0] / fs * n_fft).astype(int)
    km = np.round(centers / fs * n_fft).astype(int)
    k2 = np.round(bands[:, 1] / fs * n_fft).astype(int)

    G = np.ones((km.shape[0], n_freq))

    P = np.round(overlap_ratio * (k2 - km)).astype(int)

    # Corrects the start and end of the first and last bands, respectively.
    k1[0] = 0
    k2[-1] = n_freq

    k_low, k_high = [0], []
    for band in range(1, km.shape[0]):

        if P[band] > 0:
            p = np.arange(-P[band], P[band] + 1)

            # Compute phi according to eq. (18) and (19).
            phi = p / P[band]
            for _ in range(slope):
                phi = np.sin(np.pi / 2 * phi)
            phi = 0.5 * (phi + 1)

            # Build the decreasing part of the previous band.
            G[band - 1, k1[band] - P[band] : k1[band] + P[band] + 1] = np.cos(
                np.pi / 2 * phi
            )
            # apply fade in in to next channel
            G[band, k1[band] - P[band] : k1[band] + P[band] + 1] = np.sin(
                np.pi / 2 * phi
            )

        # set current and next channel to zero outside their range
        G[band - 1, k1[band] + P[band] :] = 0.0
        G[band, : k1[band] - P[band]] = 0.0

        k_high.append(k1[band] + P[band])
        k_low.append(k1[band] - P[band])
    k_high.append(n_freq)

    return G, n_freq, np.array(k_low), km, np.array(k_high)


class BaseOctaveFilterBank(metaclass=abc.ABCMeta):
    """
    A base class for octave filter banks.
    """

    @abc.abstractmethod
    def get_bw(self):
        pass

    @abc.abstractmethod
    def analysis(self, x, band=None, **kwargs):
        pass

    @abc.abstractmethod
    def synthesis(self, coeffs, min_phase=False, **kwargs):
        pass

    @abc.abstractmethod
    def energy(self, x, **kwargs):
        pass

    def __call__(self, coeffs=0.0, center_freqs=None, interp_kind="linear", **kwargs):
        """
        Takes as input a list of values with optional corresponding center frequency.
        Returns a list with the correct number of octave bands. Interpolation and
        extrapolation are used to fill in the missing values.

        Parameters
        ----------
        coeffs: list
            A list of values to use for the octave bands
        center_freqs: list, optional
            The optional list of center frequencies
        interp_kind: str
            Specifies the kind of interpolation as a string (‘linear’,
            ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
            ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a
            spline interpolation of zeroth, first, second or third order;
            ‘previous’ and ‘next’ simply return the previous or next value of
            the point) or as an integer specifying the order of the spline
            interpolator to use. Default is ‘linear’.
        """

        if not isinstance(coeffs, (list, np.ndarray)):
            # when the parameter is a scalar just do flat extrapolation
            ret = [coeffs] * self.n_bands

        if len(coeffs) == 1:
            ret = coeffs * int(self.n_bands)

        else:
            # by default infer the center freq to be the low ones
            if center_freqs is None:
                center_freqs = self.centers[: len(coeffs)]

            # create the interpolator in log domain
            interpolator = interp1d(
                np.log2(center_freqs),
                coeffs,
                fill_value="extrapolate",
                kind=interp_kind,
            )
            ret = interpolator(np.log2(self.centers))

            # now clip between 0. and 1.
            ret[ret < 0.0] = 0.0
            ret[ret > 1.0] = 1.0

        return ret


class OctaveBandsFactory(BaseOctaveFilterBank):
    """
    A class to process uniformly all properties that are defined on octave
    bands.

    Each property is stored for an octave band.

    Attributes
    ----------
    base_freq: float
        The center frequency of the first octave band
    fs: float
        The target sampling frequency
    n_bands: int
        The number of octave bands needed to cover from base_freq to fs / 2
        (i.e. floor(log2(fs / base_freq)))
    bands: list of tuple
        The list of bin boundaries for the octave bands
    centers
        The list of band centers

    Parameters
    ----------
    base_frequency: float, optional
        The center frequency of the first octave band (default: 125 Hz)
    fs: float, optional
        The sampling frequency used (default: 16000 Hz)
    n_fft: bool, optional
        The FFT size to use
    keep_dc: bool
        If True, include all the lower frequencies in the first filter
    min_phase: bool
        If True, make the filters minimum phase
    """

    def __init__(
        self, base_frequency=125.0, fs=16000, n_fft=512, keep_dc=False, min_phase=False
    ):
        self.base_freq = base_frequency
        self.fs = fs
        self.n_fft = n_fft
        self.keep_dc = keep_dc
        self.min_phase = min_phase

        # compute the number of bands
        self.n_bands = math.floor(np.log2(fs / base_frequency))

        self.bands, self.centers = octave_bands(
            fc=self.base_freq, n=self.n_bands, third=False
        )

        self.filters, self.magnitude_response = self._make_filters()

    def get_bw(self):
        """Returns the bandwidth of the bands"""
        bands = self.bands
        if self.keep_dc:
            bands[0] = [0.0, bands[0][1]]
        bands[-1] = [bands[-1][0], self.fs / 2]

        return np.array([min(b2, self.fs // 2) - max(b1, 0) for b1, b2 in bands])

    def analysis(self, x, band=None, mode="same"):
        """
        Process a signal x through the filter bank

        Parameters
        ----------
        x: ndarray (n_samples)
            The input signal
        band:
            The index of the band to transform. If ``None``, all the bands are
            analyzed and returned.
        mode:
            The mode to use for fftconvolve.

        Returns
        -------
        ndarray (n_samples, n_bands)
            The input signal filters through all the bands
        """
        if band is None:
            bands = np.arange(self.n_bands)
        elif isinstance(band, int):
            bands = [band]
        elif isinstance(band, list) and all(isinstance(b, int) for b in band):
            bands = band
        else:
            raise ValueError(f"band should be an int or list of int (got {band}).")

        filters = self.filters[:, bands]

        x = np.stack([x] * len(bands), axis=-1)
        output = fftconvolve(x, filters, mode=mode, axes=(-2,))

        if output.shape[1] == 1:
            return output[:, 0]
        else:
            return output

    def synthesis(self, coeffs, min_phase=False):
        """
        Creates a filter with the desired band amplitudes.

        Parameters
        ----------
        band_magnitudes: np.ndarray
            The band amplitude coefficents (..., n_bands)
        min_phase: bool
            The filters are made minimum phase if ``True``.

        Returns
        -------
            The impulse responses with the correct levels (..., n_fft)
        """
        ir = np.einsum("...b,tb->...t", coeffs, self.filters)
        if min_phase:
            mag_resp = np.abs(np.fft.rfft(ir, axis=-1))
            return magnitude_response_to_minimum_phase(
                mag_resp, self.n_fft, axis=-1, eps=1e-7
            )
        else:
            return ir

    def energy(self, x):
        """
        Computes the per-band energy of the input signal.

        Parameters
        ----------
        x: np.ndarray (..., n_samples)
            The signal to analyze.

        Returns
        -------
        np.ndarray (..., n_bands)
            The per-band energy of the input signal.
        """
        x_bands = self.analysis(x)
        return np.sum(x_bands**2, axis=-1)

    def _make_filters(self):
        """
        Creates the band-pass filters for the octave bands
        """
        mag_resp, *_ = cosine_magnitude_octave_filter_response(
            self.n_fft, self.centers, self.fs, self.keep_dc
        )
        mag_resp = mag_resp.T

        # Delay the filters to match mode="same" of fftconvolve.
        n_freq = self.n_fft // 2 + 1
        delay = np.exp(
            2j * np.pi * np.arange(n_freq) * (self.n_fft // 2 + 1) / self.n_fft
        )
        filters = np.fft.irfft(mag_resp * delay[:, None], n=self.n_fft, axis=0)

        if self.min_phase:
            magnitude_response = np.abs(np.fft.rfft(filters, axis=0))
            filters = magnitude_response_to_minimum_phase(
                magnitude_response, self.n_fft, axis=0, eps=2e-2
            )

        # Octave band filters in frequency domain
        self.filters_freq_domain = np.fft.fft(filters, axis=0, n=self.n_fft)

        return filters, mag_resp


@dataclasses.dataclass(frozen=True)
class AntoniOctaveFilterBankParameters:
    """
    A data structure to hold the paramters used for the analysis
    of a signal with the Antoni octave filter bank.
    """

    windows: List[np.ndarray]
    n_fft: int
    analyzed_band_indices: List[int]
    bands_lower_bins: np.ndarray
    bands_center_bins: np.ndarray
    bands_upper_bins: np.ndarray
    output_length: int
    output_dtype: type
    padded_length: int


class AntoniOctaveFilterBank(BaseOctaveFilterBank):
    """
    This class implements a type of fractional octave filter bank with
    both perfect reconstruction and energy conservation.

    J. Antoni, Orthogonal-like fractional-octave-band filters, J. Acoust. Soc.
    Am., 127, 2, February 2010

    Attributes
    ----------
    base_freq: float
        The center frequency of the first octave band
    fs: float
        The target sampling frequency
    n_bands: int
        The number of octave bands needed to cover from base_freq to fs / 2
        (i.e. floor(log2(fs / base_freq)))
    bands: list of tuple
        The list of bin boundaries for the octave bands
    centers
        The list of band centers

    Parameters
    ----------
    base_frequency: float, optional
        The center frequency of the first octave band (default: 125 Hz)
    fs: float, optional
        The sampling frequency used (default: 16000 Hz)
    n_fft: bool, optional
        The FFT size to use
    band_overlap_ratio: float
        The overlap between bands. It should be between 0.0 and 0.5.
    slope: int
        A parameter controlling the transition between bands.
        The larger, the sharper the transition.
    third: bool
        If set to True, a third Octave band filter bank is created.
    """

    def __init__(
        self,
        base_frequency: float = 125.0,
        fs: float = 16000,
        n_fft: int = 512,
        band_overlap_ratio: float = 0.5,
        slope: int = 0,
        third: bool = False,
    ):
        if not (0.0 <= band_overlap_ratio <= 0.5):
            raise ValueError("The band overlap ratio should be in [0, 0.5].")

        self.base_freq = base_frequency
        self.fs = fs
        self.n_fft = n_fft
        self.overlap_ratio = band_overlap_ratio
        self.slope = slope

        # Compute the number of octaves.
        n_octaves = math.floor(np.log2(fs / base_frequency))
        self.bands, self.centers = octave_bands(
            fc=self.base_freq, n=n_octaves, third=third
        )
        self.n_bands = self.centers.shape[0]

        G, *_ = self._make_window_function(self.n_fft)
        self.filters = G.T

    def get_bw(self, n_fft=None):
        """Returns the bandwidth of the bands"""
        if n_fft is None:
            n_fft = self.n_fft
        k1 = np.round(self.bands[:, 0] / self.fs * n_fft).astype(int)
        k2 = np.round(self.bands[:, 1] / self.fs * n_fft).astype(int)

        # Corrects the start and end of the first and last bands, respectively.
        k1[0] = 0
        k2[-1] = n_fft // 2 + 1

        diff = k2 - k1
        ratio = diff / diff.sum()
        return ratio * self.fs / 2.0

    def _make_window_function(self, n_fft) -> np.ndarray:
        """
        Implementation adapted from
        https://github.com/pyfar/pyfar/blob/main/pyfar/dsp/filter/fractional_octaves.py#L339
        MIT License.
        """

        return antoni_magnitude_octave_filter_response(
            n_fft, self.centers, self.bands, self.fs, self.overlap_ratio, self.slope
        )

    def wavelet_analysis(self, x, band=None, oversampling=2):
        """
        Compute the decomposition proposed by Antoni 2008.

        Parameters
        ----------
        x: ndarray (..., n_samples)
            The input signal
        band:
            The index of the band to transform. If ``None``, all the bands are
            analyzed and returned.
        oversampling: int
            Oversampling of FFT to use (default: 2).

        Returns
        -------
        signal: list[np.ndarray]
            The coefficients of the input signal filters obtained by
            time-frequency analysis.
        parameters: AntoniOctaveFilterBankParameters
            A data structure that contains the parameters used during
            the analysis.
        """
        if band is None:
            bands = np.arange(self.n_bands)
        elif isinstance(band, int):
            bands = [band]
        elif isinstance(band, list) and all(isinstance(b, int) for b in band):
            bands = band
        else:
            raise ValueError(f"band should be an int or list of int (got {band}).")

        n_fft = 2 ** math.ceil(math.log2(oversampling * x.shape[-1]))
        X = np.fft.rfft(x, axis=-1, n=n_fft) / np.sqrt(n_fft)

        G, n_freq, k1, km, k2 = self._make_window_function(n_fft)

        Nm = np.ceil((k2 - k1) / 2.0).astype(int)  # index of Nm is i in the paper.
        upper = np.max(k1 + 2 * Nm)
        padded_length = X.shape[-1]
        if upper > n_freq:
            padding = np.zeros(X.shape[:-1] + (upper - n_freq,), dtype=X.dtype)
            padded_length += upper - n_freq
            X = np.concatenate((X, padding), axis=-1)
            padding = np.zeros(G.shape[:-1] + (upper - n_freq,), dtype=G.dtype)
            G = np.concatenate((G, padding), axis=-1)

        signal = []  # X_ij in the paper
        windows_nonzero = []
        for band in bands:
            N = Nm[band]
            k = k1[band] + np.arange(2 * N, dtype=int)
            delta = np.sqrt(1 / (2.0 * N))
            windows_nonzero.append(G[band, k])

            # Analysis
            j = np.arange(-N + 1, N + 1)
            factor = 1j**band * delta
            cexp = np.exp(1j * np.pi * (km[band] - k1[band]) * j / N)
            W_pos = np.fft.fft(G[band, k] * X[..., k], axis=-1)
            W_neg = np.conj(np.fft.fft(G[band, k] * np.conj(X[..., k]), axis=-1))
            W_np = np.concatenate(
                (W_neg[..., N - 1 : 0 : -1], W_pos[..., : N + 1]), axis=-1
            )
            signal.append(factor * cexp * W_np)

        return signal, AntoniOctaveFilterBankParameters(
            windows=windows_nonzero,
            n_fft=n_fft,
            analyzed_band_indices=bands,
            bands_lower_bins=k1,
            bands_center_bins=km,
            bands_upper_bins=k2,
            output_length=x.shape[-1],
            output_dtype=X.dtype,
            padded_length=padded_length,
        )

    def wavelet_synthesis(
        self, signal: List[np.ndarray], parameters: AntoniOctaveFilterBankParameters
    ) -> np.ndarray:
        """
        Given the decomposition of the signal by Antoni 2008, compute
        the octave band signals.

        Paramters
        ---------
        coeffs: list[np.ndarray]
            A list containing the coefficients corresponsing to every octave band.
        parameters: AntoniOctaveFilterBankParameters
            The parameters of the analysis filterbank.

        Returns
        -------
        np.ndarray (..., num_samples, num_bands)
            The time domain representation of the octave bands at the original sampling rate.
        """

        W = signal
        n_freq = parameters.n_fft // 2 + 1
        G = parameters.windows
        k1 = parameters.bands_lower_bins
        km = parameters.bands_center_bins

        X_filt = np.zeros(
            W[0].shape[:-1] + (parameters.padded_length, len(W)),
            dtype=parameters.output_dtype,
        )

        for idx, band in enumerate(parameters.analyzed_band_indices):
            coeffs = W[idx]
            window = G[idx]
            N = coeffs.shape[-1] // 2
            k = k1[band] + np.arange(2 * N, dtype=int)
            delta = np.sqrt(1 / (2.0 * N))

            # Synthesis
            factor = (-1j) ** band * window * delta
            cexp1 = np.exp(-1j * np.pi * (k - km[band]) * (N - 1) / N)
            cexp2 = np.exp(-1j * np.pi * (km[band] - k1[band]) * np.arange(2 * N) / N)
            Y = np.conj(np.fft.fft(np.conj(cexp2 * coeffs)))
            X_filt[..., k1[band] : k1[band] + 2 * N, idx] = factor * cexp1 * Y

        # Synthesize the output band-pass signals.
        y = np.fft.irfft(X_filt[..., :n_freq, :], axis=-2) * np.sqrt(parameters.n_fft)
        return y[..., : parameters.output_length, :]

    def energy(self, x, oversampling=2):
        """
        Computes the per-band energy of the input signal.

        Parameters
        ----------
        x: np.ndarray (..., n_samples)
            The signal to analyze.
        oversampling: int, optional
            The oversampling to use in the analysis (default 2).

        Returns
        -------
        np.ndarray (..., n_bands)
            The per-band energy of the input signal.
        """
        coeffs, _ = self.wavelet_analysis(x, oversampling=oversampling)
        energy = 2.0 * np.array([(abs(w) ** 2).sum() for w in coeffs])
        return energy

    def analysis(self, x, band=None, oversampling=2):
        """
        Process a signal x through the filter bank

        Parameters
        ----------
        x: ndarray (..., n_samples)
            The input signal
        band: int
            The index of the band to transform. If ``None``, all the bands are
            analyzed and returned.
        oversampling: int
            Oversampling of FFT to use (default: 2).

        Returns
        -------
        ndarray (..., n_samples, n_bands)
            The input signal filters through all the bands
        """

        coeffs, parameters = self.wavelet_analysis(
            x, band=band, oversampling=oversampling
        )
        output = self.wavelet_synthesis(coeffs, parameters)
        if output.shape[-1] == 1:
            return output[..., 0]
        else:
            return output

    def synthesis(
        self, band_magnitudes, min_phase=False, filter_length=None, oversampling=2
    ):
        """
        Creates a filter with the desired band amplitudes.

        Parameters
        ----------
        band_magnitudes: np.ndarray
            The band amplitude coefficents (..., n_bands)
        min_phase: bool
            The filters are made minimum phase if ``True``.
        filter_length: int
            The length of the filters.
        oversampling: int
            The oversampling to use in the analysis.

        Returns
        -------
            The impulse responses with the correct levels (..., n_fft)
        """
        coeffs = np.array(band_magnitudes)

        if filter_length is None:
            filter_length = self.n_fft

        # We will reshape the response of a unit impulse filter.
        ir = np.zeros(coeffs.shape[:-1] + (filter_length,), dtype=coeffs.dtype)
        ir[..., filter_length // 2] = 1.0

        # Analyze the signal and reshape each band by the target magnitude.
        signal, paramters = self.wavelet_analysis(ir, oversampling=oversampling)
        for band in range(coeffs.shape[-1]):
            signal[band] *= coeffs[..., [band]]

        # Construct the filter.
        ir = self.wavelet_synthesis(signal, paramters).sum(axis=-1)

        if min_phase:
            # Transform the filter to be minimum phase.
            mag_resp = np.abs(np.fft.rfft(ir, axis=-1))
            return magnitude_response_to_minimum_phase(
                mag_resp, ir.shape[-1], axis=-1, eps=1e-7
            )
        else:
            return ir


def critical_bands():
    """
    Compute the Critical bands as defined in the book:
    Psychoacoustics by Zwicker and Fastl. Table 6.1 p. 159
    """

    # center frequencies
    fc = [
        50,
        150,
        250,
        350,
        450,
        570,
        700,
        840,
        1000,
        1170,
        1370,
        1600,
        1850,
        2150,
        2500,
        2900,
        3400,
        4000,
        4800,
        5800,
        7000,
        8500,
        10500,
        13500,
    ]
    # boundaries of the bands (e.g. the first band is from 0Hz to 100Hz
    # with center 50Hz, fb[0] to fb[1], center fc[0]
    fb = [
        0,
        100,
        200,
        300,
        400,
        510,
        630,
        770,
        920,
        1080,
        1270,
        1480,
        1720,
        2000,
        2320,
        2700,
        3150,
        3700,
        4400,
        5300,
        6400,
        7700,
        9500,
        12000,
        15500,
    ]

    # now just make pairs
    bands = [[fb[j], fb[j + 1]] for j in range(len(fb) - 1)]

    return np.array(bands), fc


def bands_hz2s(bands_hz, Fs, N, transform="dft"):
    """
    Converts bands given in Hertz to samples with respect to a given sampling
    frequency Fs and a transform size N an optional transform type is used to
    handle DCT case.
    """

    # set the bin width
    if transform == "dct":
        B = Fs / 2 / N
    else:
        B = Fs / N

    # upper limit of the frequency range
    limit = min(Fs / 2, bands_hz[-1, 1])

    # Convert from Hertz to samples for all bands
    bands_s = [np.around(band / B) for band in bands_hz if band[0] <= limit]

    # Last band ends at N/2
    bands_s[-1][1] = N / 2

    # remove duplicate, if any, (typically, if N is small and Fs is large)
    j = 0
    while j < len(bands_s) - 1:
        if bands_s[j][0] == bands_s[j + 1][0]:
            bands_s.pop(j)
        else:
            j += 1

    return np.array(bands_s, dtype=int)


def melscale(f):
    """Converts f (in Hertz) to the melscale defined according to Huang-Acero-Hon (2.6)"""
    return 1125.0 * np.log(1 + f / 700.0)


def invmelscale(b):
    """Converts from melscale to frequency in Hertz according to Huang-Acero-Hon (6.143)"""
    return 700.0 * (np.exp(b / 1125.0) - 1)


def melfilterbank(M, N, fs=1, fl=0.0, fh=0.5):
    """
    Returns a filter bank of triangular filters spaced according to mel scale

    We follow Huang-Acera-Hon 6.5.2

    Parameters
    ----------
    M : (int)
        The number of filters in the bank
    N : (int)
        The length of the DFT
    fs : (float) optional
        The sampling frequency (default 8000)
    fl : (float)
        Lowest frequency in filter bank as a fraction of fs (default 0.)
    fh : (float)
        Highest frequency in filter bank as a fraction of fs (default 0.5)

    Returns
    -------
    An M times int(N/2)+1 ndarray that contains one filter per row
    """

    # all center frequencies of the filters
    f = (N / fs) * invmelscale(
        melscale(fl * fs)
        + (np.arange(M + 2) * (melscale(fh * fs) - melscale(fl * fs)) / (M + 1))
    )

    # Construct the triangular filter bank
    H = np.zeros((M, N // 2 + 1))
    k = np.arange(N // 2 + 1)
    for m in range(1, M + 1):
        I = np.where(np.logical_and(f[m - 1] < k, k < f[m]))
        H[m - 1, I] = (
            2 * (k[I] - f[m - 1]) / ((f[m + 1] - f[m - 1]) * (f[m] - f[m - 1]))
        )
        I = np.where(np.logical_and(f[m] <= k, k < f[m + 1]))
        H[m - 1, I] = (
            2 * (f[m + 1] - k[I]) / ((f[m + 1] - f[m - 1]) * (f[m + 1] - f[m]))
        )

    return H


def mfcc(x, L=128, hop=64, M=14, fs=8000, fl=0.0, fh=0.5):
    """
    Computes the Mel-Frequency Cepstrum Coefficients (MFCC) according
    to the description by Huang-Acera-Hon 6.5.2 (2001)
    The MFCC are features mimicing the human perception usually
    used for some learning task.

    This function will first split the signal into frames, overlapping
    or not, and then compute the MFCC for each frame.

    Parameters
    ----------
    x : (nd-array)
        Input signal
    L : (int)
        Frame size (default 128)
    hop : (int)
        Number of samples to skip between two frames (default 64)
    M : (int)
        Number of mel-frequency filters (default 14)
    fs : (int)
        Sampling frequency (default 8000)
    fl : (float)
        Lowest frequency in filter bank as a fraction of fs (default 0.)
    fh : (float)
        Highest frequency in filter bank as a fraction of fs (default 0.5)

    Return
    ------
    The MFCC of the input signal
    """

    # perform STFT, X contains frames in rows
    X = stft.analysis(x, L, hop, transform=np.fft.rfft)

    # get and apply the mel filter bank
    # and compute log energy
    H = melfilterbank(M, L, fs=fs, fl=fl, fh=fh)
    S = np.log(np.dot(H, np.abs(X.T) ** 2))

    # Now take DCT of the result
    C = dct(S, type=2, n=M, axis=0)

    return C


def inverse_sabine(rt60, room_dim, c=None):
    """
    Given the desired reverberation time (RT60, i.e. the time for the energy to
    drop by 60 dB), the dimensions of a rectangular room (shoebox), and sound
    speed, computes the energy absorption coefficient and maximum image source
    order needed. The speed of sound used is the package wide default (in
    :py:data:`~pyroomacoustics.parameters.constants`).

    Parameters
    ----------
    rt60: float
        desired RT60 (time it takes to go from full amplitude to 60 db decay) in seconds
    room_dim: list of floats
        list of length 2 or 3 of the room side lengths
    c: float
        speed of sound

    Returns
    -------
    absorption: float
        the energy absorption coefficient to be passed to room constructor
    max_order: int
        the maximum image source order necessary to achieve the desired RT60
    """

    if c is None:
        c = constants.get("c")

    # finding image sources up to a maximum order creates a (possibly 3d) diamond
    # like pile of (reflected) rooms. now we need to find the image source model order
    # so that reflections at a distance of at least up to ``c * rt60`` are included.
    # one possibility is to find the largest sphere (or circle in 2d) that fits in the
    # diamond. this is what we are doing here.
    R = []
    for l1, l2 in itertools.combinations(room_dim, 2):
        R.append(l1 * l2 / np.sqrt(l1**2 + l2**2))

    V = np.prod(room_dim)  # area (2d) or volume (3d)
    # "surface" computation is diff for 2d and 3d
    if len(room_dim) == 2:
        S = 2 * np.sum(room_dim)
        sab_coef = 12  # the sabine's coefficient needs to be adjusted in 2d
    elif len(room_dim) == 3:
        S = 2 * np.sum([l1 * l2 for l1, l2 in itertools.combinations(room_dim, 2)])
        sab_coef = 24

    e_absorption = (
        sab_coef * np.log(10) * V / (c * S * rt60)
    )  # absorption in power (sabine)

    if e_absorption > 1.0:
        raise ValueError(
            "evaluation of parameters failed. room may be too large for required RT60."
        )

    # the int cast is only needed for python 2.7
    # math.ceil returns int for python 3.5+
    max_order = int(math.ceil(c * rt60 / np.min(R) - 1))

    return e_absorption, max_order


def rt60_eyring(S, V, a, m, c):
    """
    This is the Eyring formula for estimation of the reverberation time.

    Parameters
    ----------
    S:
        the total surface of the room walls in m^2
    V:
        the volume of the room in m^3
    a: float
        the equivalent absorption coefficient ``sum(a_w * S_w) / S`` where ``a_w`` and ``S_w`` are the absorption and surface of wall ``w``, respectively.
    m: float
        attenuation constant of air
    c: float
        speed of sound in m/s

    Returns
    -------
    float
        The estimated reverberation time (RT60)
    """

    return -(24 * np.log(10) / c) * V / (S * np.log(1 - a) + 4 * m * V)


def rt60_sabine(S, V, a, m, c):
    """
    This is the Eyring formula for estimation of the reverberation time.

    Parameters
    ----------
    S:
        the total surface of the room walls in m^2
    V:
        the volume of the room in m^3
    a: float
        the equivalent absorption coefficient ``sum(a_w * S_w) / S`` where ``a_w`` and ``S_w`` are the absorption and surface of wall ``w``, respectively.
    m: float
        attenuation constant of air
    c: float
        speed of sound in m/s

    Returns
    -------
    float
        The estimated reverberation time (RT60)
    """

    return (24 * np.log(10) / c) * V / (a * S + 4 * m * V)
