# Some classes to apply rotate objects or indicate directions in 3D space.
# Copyright (C) 2024  Robin Scheibler
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
Internal routines used for simulation using the ray tracing method.
In particular, how to transform the histograms obtained from the core
simulation engine into impulse responses.
"""

import math

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

from .. import random


def poisson_sequence(volume, duration, c, max_rate=10000, t0=None):
    # Get the random number generator.
    rng = random.get_rng()

    # repeated constant
    fpcv = 4 * np.pi * c**3 / volume

    if t0 is None:
        # initial time
        t0 = ((2 * np.log(2)) / fpcv) ** (1.0 / 3.0)

    times = [t0]

    while times[-1] < t0 + duration:
        # uniform random variable
        z = rng.uniform()
        # rate of the point process at this time
        mu = np.minimum(fpcv * (t0 + times[-1]) ** 2, max_rate)
        # time interval to next point
        dt = np.log(1 / z) / mu

        times.append(times[-1] + dt)

    return t0, np.array(times)


def binary_sequence(times, fs):
    # Get the random number generator.
    rng = random.get_rng()

    indices = (times * fs).astype(np.int64)
    seq = np.zeros(indices[-1] + 1)
    seq[indices] = rng.choice([1, -1], size=len(indices))

    return seq


def directional_sequence(t0, times, response, hist, hist_bin_size, fs):
    """
    This method construct a sequence of delta, possibly convolved with directional
    impulse response according to the directional energy distribution given by ``hist``.
    In addition, the sequence is locally scaled to have unit energy per histogram bin.

    Parameters
    ----------
    t0: float
        Time of the first delta.
    times: (n_deltas,)
        The time of the deltas in the sequence.
    response: (n_dirs, n_taps)
        The directional response of the receiver.
    hist: (n_dirs, n_bins)
        The histograms of direction of arrival probability.
    hist_bin_size: float
        The length in seconds of a bin of the histogram.
    fs: float
        The sampling frequency of the sequence.

    Returns
    -------
    A sequence built according to D. Schroeder, "Physically based real-time
    auralization of interactive virtual environments," Section 5.3.4, "Binaural
    Room Impulse Response".
    """
    rng = random.get_rng()

    # Compute the delta binary sequency first to compute local energy average
    # for each sequence bin.
    seq_bp = binary_sequence(times, fs)
    seq_power = seq_rolling_power(seq_bp, int(hist_bin_size * fs), filter_length_mult=2)
    nonzero = seq_power > 1e-10
    seq_power_norm = np.where(nonzero, 1.0 / np.where(nonzero, seq_power, 1.0), 0.0)

    # Rounded times of the deltas.
    indices = (times * fs).astype(np.int64)
    # Attribute each delta to a histogram bin.
    bins = np.floor((times - t0) / hist_bin_size).astype(np.int64)
    bins = np.minimum(bins, hist.shape[-1] - 1)

    # Compute the CDF over directions.
    # When all the bins are zero, we use a uniform distribution.
    cumul_hist = np.cumsum(hist, axis=0)
    cumul_uniform = np.cumsum(np.ones((cumul_hist.shape[0], 1)), axis=0)
    mask = cumul_hist[[-1], :] > 1e-10
    denom = np.where(mask, cumul_hist[[-1], :], cumul_uniform[[-1], :])
    cdf = np.where(mask, cumul_hist, cumul_uniform) / denom

    # CDF of directions for each delta.
    cdf_per_delta = cdf[:, bins]

    # Draw a response at random for each delta.
    z = rng.uniform(size=bins.shape[0])
    r_ind = np.sum(z[None, :] > cdf_per_delta, axis=0)

    # (n_times, n_taps)
    chosen_response = response[r_ind, :]

    signs = rng.choice([1, -1], size=len(indices))

    # Construct the sequence.
    # TODO: vectorize or port to C++.
    seq = np.zeros(indices[-1] + response.shape[1] + 1)
    for i, sign, taps in zip(indices, signs, chosen_response):
        seq[i : i + taps.shape[0]] += sign * taps * np.sqrt(seq_power_norm[i])

    return seq


def interp_hist(hist, N):
    """
    interpolate the histogram on N samples

    we use the  bin centers as anchors
    since we can't interpolate outside the anchors, we just repeat
    the first and last value to fill the array

    Parameters
    ----------
    hist: ndarray, (directions, bins)
        The histogram to interpolate.
    N: int
        The number samples.
    """
    n_input = hist.shape[-1]
    hbss = N // n_input
    pad = hbss // 2
    t = np.linspace(pad, N - 1 - pad, n_input)
    f = interp1d(t, hist, axis=-1)
    out = np.zeros((*hist.shape[:-1], N))
    out[..., pad:-pad] = f(np.arange(pad, N - pad))
    out[..., :pad] = out[..., [pad]]
    out[..., -pad:] = out[..., [-pad]]
    return out


def seq_rolling_power(seq, hbss, filter_length_mult=1):
    """Smooth local energy of the sequence."""
    filter = np.ones(filter_length_mult * hbss) / filter_length_mult
    return fftconvolve(seq**2, filter, mode="same")


def adjust_length_and_stack(histograms, n_bins):
    def adjust_length(h, n):
        if h.shape[-1] > n:
            return h[..., :n]
        elif h.shape[-1] < n:
            pad_width = [(0, 0)] * (h.ndim - 1) + [(0, n - h.shape[-1])]
            return np.pad(h, pad_width)
        else:
            return h

    return np.stack([adjust_length(h, n_bins) for h in histograms], axis=0)


def compute_rt_rir(
    t0,
    histograms,
    directional_responses,
    hist_bin_size,
    hist_bin_size_samples,
    volume_room,
    fdl,
    c,
    fs,
    octave_bands,
    air_abs_coeffs=None,
):

    # Get the maximum length from the histograms Sum vertically across octave
    # band for each value in histogram (7,2500) -> (2500). Set the number of
    # bins to one more than the location of the last non-zero element.
    def last_el(x):
        return 0 if len(x) == 0 else x[-1] + 1

    n_bins = max(last_el(np.nonzero(h.sum(axis=0))[0]) for h in histograms)

    if n_bins == 0:
        # The histogram is all zeros, there is no RIR to build
        # we return only an RIR that contains the default delay.
        return np.zeros(fdl // 2)

    t_max = n_bins * hist_bin_size

    # N changes here, the length of RIR changes if we apply RT method.
    # the number of samples needed
    # round up to multiple of the histogram bin size
    # add the lengths of the fractional delay filter

    hbss = int(hist_bin_size_samples)

    fdl2 = fdl // 2  # delay due to fractional delay filter
    # make the length a multiple of the bin size
    n_bins = math.ceil(t_max * fs / hbss)
    N = n_bins * hbss

    # Consolidate the histograms in a single array.
    # Shape: (directions, bands, bins)
    hist_all = adjust_length_and_stack(histograms, n_bins)
    n_dirs, n_bands, _ = hist_all.shape

    # Make a directional histogram aggregated over bands.
    hist_directions = np.sum(hist_all, axis=1)

    # Make a band histogram aggregated over directions.
    hist_bands = np.sum(hist_all, axis=0)

    # Directional gain.
    response = directional_responses["response"]
    if not directional_responses["is_impulse_response"]:
        if response.ndim == 1:
            response = np.broadcast_to(response[:, None], (n_dirs, 1))
        elif response.ndim == 2 and response.shape[1] == n_bands:
            response = octave_bands.synthesis(response)
        else:
            raise ValueError(
                f"Invalide response shape {response.shape} found "
                f"with n_bands != {response.shape[-1]}"
            )

    # This is the random sequence for the tail generation.
    t0, delta_times = poisson_sequence(volume_room, N / fs, c, t0=t0)
    seq = directional_sequence(
        t0, delta_times, response, hist_directions, hist_bin_size, fs
    )
    # Limit the length of the sequence to what is covered by the histogram.
    seq = seq[:N]

    rir_bands = np.zeros((n_bands, N))
    for b in range(n_bands):
        if n_bands > 1:
            seq_bp = octave_bands.analysis(seq, band=b)
        else:
            seq_bp = seq.copy()

        # The sequence local energy normalization is handled during the sequence
        # generation.
        # The weighting by the bandwidth is done implicitely when filtering the
        # sequence. Since the sequence has a flat spectrum, it will be divided
        # correctly by the octave band filter.

        # We linarly interpolate the histogram to smoothly cover all the samples.
        hist = interp_hist(hist_bands[b, :], N)

        # Impulse response for every octave band for each microphone.
        # Clamp to non-negative: interp_hist can produce tiny negatives
        # from floating-point noise in linear interpolation.  sqrt of a
        # negative yields NaN which downstream IIR filters (e.g. the
        # high-pass in compute_rir) then propagate to the entire RIR.
        rir_bands[b, :] = seq_bp * np.sqrt(np.maximum(hist, 0.0))

    if air_abs_coeffs is not None:
        if rir_bands.shape[0] == 1:
            # Do the octave band analysis if it was not done in the first step.
            rir_bands = np.array(
                [
                    octave_bands.analysis(rir_bands[0, :], band=b)
                    for b in range(octave_bands.n_bands)
                ]
            )

        air_abs = np.exp(
            -0.5 * air_abs_coeffs[:, None] * np.arange(N)[None, :] / fs * c
        )

        rir_bands *= air_abs

    # Pad half a fractional delay filter for compatibility with the ISM.
    return np.pad(np.sum(rir_bands, axis=0), (fdl2, 0))
