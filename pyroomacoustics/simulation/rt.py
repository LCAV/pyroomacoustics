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


def sequence_generation(volume, duration, c, fs, max_rate=10000):
    # repeated constant
    fpcv = 4 * np.pi * c**3 / volume

    # initial time
    t0 = ((2 * np.log(2)) / fpcv) ** (1.0 / 3.0)
    times = [t0]

    while times[-1] < t0 + duration:
        # uniform random variable
        z = np.random.rand()
        # rate of the point process at this time
        mu = np.minimum(fpcv * (t0 + times[-1]) ** 2, max_rate)
        # time interval to next point
        dt = np.log(1 / z) / mu

        times.append(times[-1] + dt)

    # convert from continuous to discrete time

    indices = (np.array(times) * fs).astype(np.int64)
    seq = np.zeros(indices[-1] + 1)
    seq[indices] = np.random.choice([1, -1], size=len(indices))

    return seq


def interp_hist(hist, N):
    """
    interpolate the histogram on N samples

    we use the  bin centers as anchors
    since we can't interpolate outside the anchors, we just repeat
    the first and last value to fill the array
    """
    hbss = N // hist.shape[0]
    pad = hbss // 2
    t = np.linspace(pad, N - 1 - pad, hist.shape[0])
    f = interp1d(t, hist)
    out = np.zeros(N)
    out[pad:-pad] = f(np.arange(pad, N - pad))
    out[:pad] = out[pad]
    out[-pad:] = out[-pad]
    return out


def seq_bin_power(seq, hbss):
    seq_rot = seq.reshape((-1, hbss))  # shape 72,64
    power = np.sum(seq_rot**2, axis=1)
    power[power <= 0.0] = 1.0
    return power


def compute_rt_rir(
    histograms,
    hist_bin_size,
    hist_bin_size_samples,
    volume_room,
    fdl,
    c,
    fs,
    octave_bands,
    air_abs_coeffs=None,
):
    # get the maximum length from the histograms
    # Sum vertically across octave band for each value in
    # histogram (7,2500) -> (2500) -> np .nonzero(
    nz_bins_loc = np.nonzero(histograms[0].sum(axis=0))[0]

    if len(nz_bins_loc) == 0:
        # the histogram is all zeros, there is no RIR to build
        # we return only an RIR that contains the default delay
        return np.zeros(fdl // 2)
    else:
        n_bins = nz_bins_loc[-1] + 1

    t_max = n_bins * hist_bin_size

    # N changes here , the length of RIR changes if we apply RT method.
    # the number of samples needed
    # round up to multiple of the histogram bin size
    # add the lengths of the fractional delay filter

    hbss = int(hist_bin_size_samples)

    fdl2 = fdl // 2  # delay due to fractional delay filter
    # make the length a multiple of the bin size
    n_bins = math.ceil(t_max * fs / hbss)
    N = n_bins * hbss

    # this is the random sequence for the tail generation
    seq = sequence_generation(volume_room, N / fs, c, fs)
    seq = seq[:N]  # take values according to N as seq is larger

    n_bands = histograms[0].shape[0]
    bws = octave_bands.get_bw() if n_bands > 1 else [fs / 2]

    rir_bands = np.zeros((n_bands, N))
    for b, bw in enumerate(bws):  # Loop through every band
        if n_bands > 1:
            seq_bp = octave_bands.analysis(seq, band=b)
        else:
            seq_bp = seq.copy()

        # Take only those bins which have some non-zero values for that specific octave bands.
        hist = histograms[0][b, :n_bins]

        # we normalize the histogram by the sequence power in that bin
        seq_power = seq_bin_power(seq_bp, hbss)

        # we linarly interpolate the histogram to smoothly cover all the samples
        hist_lin_interp = interp_hist(hist / seq_power, N)

        # Normalize the band power
        # the (bw / fs * 2.0) is necessary to normalize the band power
        # this is the contribution of the octave band to total energy
        seq_bp *= np.sqrt(bw / fs * 2.0 * hist_lin_interp)

        # Impulse response for every octave band for each microphone
        rir_bands[b] = seq_bp

    if air_abs_coeffs is not None:
        if rir_bands.shape[0] == 1:
            # do the octave band analysis if it was not done in the first step
            rir_bands = np.array(
                [
                    octave_bands.analysis(rir_bands[0], band=b)
                    for b in range(octave_bands.n_bands)
                ]
            )

        air_abs = np.exp(
            -0.5 * air_abs_coeffs[:, None] * np.arange(N)[None, :] / fs * c
        )

        rir_bands *= air_abs

    rir = np.zeros(fdl2 + N)
    rir[fdl2:] = np.sum(rir_bands, axis=0)

    return rir
