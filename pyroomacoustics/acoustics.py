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

import math
import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve
from scipy.fftpack import dct
from scipy.interpolate import interp1d
from .stft import stft


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


def octave_bands(fc=1000, third=False, start=0., n=8):
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


class OctaveBandsFactory(object):
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
    all_materials: list of Material
        The list of all Material objects created by the factory

    Parameters
    ----------
    base_frequency: float, optional
        The center frequency of the first octave band (default: 125 Hz)
    fs: float, optional
        The sampling frequency used (default: 16000 Hz)
    third_octave: bool, optional
        Use third octave bands if True (default: False)
    """

    def __init__(self, base_frequency=125.0, fs=16000, n_fft=512):

        self.base_freq = base_frequency
        self.fs = fs
        self.n_fft = n_fft

        # compute the number of bands
        self.n_bands = math.floor(np.log2(fs / base_frequency))

        self.bands, self.centers = octave_bands(
            fc=self.base_freq, n=self.n_bands, third=False
        )

        self._make_filters()

    def get_bw(self):
        """ Returns the bandwidth of the bands """
        return np.array([b2 - b1 for b1, b2 in self.bands])

    def analysis(self, x, band=None):
        """
        Process a signal x through the filter bank

        Parameters
        ----------
        x: ndarray (n_samples)
            The input signal

        Returns
        -------
        ndarray (n_samples, n_bands)
            The input signal filters through all the bands
        """

        if band is None:
            bands = range(self.filters.shape[1])
        else:
            bands = [band]

        output = np.zeros((x.shape[0], self.filters.shape[1]), dtype=x.dtype)

        for b in bands:
            output[:, b] = fftconvolve(x, self.filters[:, b], mode="same")

        if band is None:
            return output
        else:
            return output[:, 0]

    def __call__(self, coeffs=0., center_freqs=None, interp_kind="linear", **kwargs):
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

    def _make_filters(self):
        """
        Create the IIR band-pass filters for the octave bands

        Parameters
        ----------
        order: int, optional
            The order of the IIR filters (default: 8)
        output: {'ba', 'zpk', 'sos'}
            Type of output: numerator/denominator ('ba'), pole-zero ('zpk'), or
            second-order sections ('sos'). Default is 'ba'.

        Returns
        -------
        A list of callables that will each apply one of the band-pass filters
        """

        """
        filter_bank = bandpass_filterbank(
            self.bands, fs=self.fs, order=order, output=output
        )

        return [lambda sig: sosfiltfilt(bpf, sig) for bpf in filter_bank]
        """

        # This seems to work only for Octave bands out of the box
        centers = self.centers
        n = len(self.centers)

        new_bands = [[centers[0] / 2, centers[1]]]
        for i in range(1, n - 1):
            new_bands.append([centers[i - 1], centers[i + 1]])
        new_bands.append([centers[-2], self.fs / 2])

        n_freq = self.n_fft // 2 + 1
        freq_resp = np.zeros((n_freq, n))
        freq = np.arange(n_freq) / self.n_fft * self.fs

        for b, (band, center) in enumerate(zip(new_bands, centers)):
            lo = np.logical_and(band[0] <= freq, freq < center)
            freq_resp[lo, b] = 0.5 * (1 + np.cos(2 * np.pi * freq[lo] / center))

            if b != n - 1:
                hi = np.logical_and(center <= freq, freq < band[1])
                freq_resp[hi, b] = 0.5 * (1 - np.cos(2 * np.pi * freq[hi] / band[1]))
            else:
                hi = center <= freq
                freq_resp[hi, b] = 0.5

        filters = np.fft.fftshift(
            np.fft.irfft(freq_resp, n=self.n_fft, axis=0),
            axes=[0],
        )

        # remove the first sample to make them odd-length symmetric filters
        self.filters = filters[1:, :]


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

    return np.array(bands_s, dtype=np.int)


def melscale(f):
    """ Converts f (in Hertz) to the melscale defined according to Huang-Acero-Hon (2.6) """
    return 1125.0 * np.log(1 + f / 700.0)


def invmelscale(b):
    """ Converts from melscale to frequency in Hertz according to Huang-Acero-Hon (6.143) """
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
    X = stft(x, L, hop, transform=np.fft.rfft)

    # get and apply the mel filter bank
    # and compute log energy
    H = melfilterbank(M, L, fs=fs, fl=fl, fh=fh)
    S = np.log(np.dot(H, np.abs(X.T) ** 2))

    # Now take DCT of the result
    C = dct(S, type=2, n=M, axis=0)

    return C
