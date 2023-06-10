import math

import numpy as np


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
        n_bins = 0
    else:
        n_bins = nz_bins_loc[-1] + 1

    t_max = n_bins * hist_bin_size

    # N changes here , the length of RIR changes if we apply RT method.
    # the number of samples needed
    # round up to multiple of the histogram bin size
    # add the lengths of the fractional delay filter

    hbss = int(hist_bin_size_samples)

    fdl2 = fdl // 2  # delay due to fractional delay filter
    N = int(math.ceil(t_max * fs / hbss) * hbss)

    # this is the random sequence for the tail generation
    seq = sequence_generation(volume_room, N / fs, c, fs)
    seq = seq[:N]  # take values according to N as seq is larger

    n_bands = histograms[0].shape[0]
    bws = octave_bands.get_bw() if n_bands > 1 else [fs / 2]

    rir = np.zeros(fdl2 + N)
    for b, bw in enumerate(bws):  # Loop through every band
        if n_bands > 1:
            seq_bp = octave_bands.analysis(seq, band=b)
        else:
            seq_bp = seq.copy()

        # interpolate the histogram and multiply the sequence

        seq_bp_rot = seq_bp.reshape((-1, hbss))  # shape 72,64

        new_n_bins = seq_bp_rot.shape[0]

        # Take only those bins which have some non-zero values for that specific octave bands.

        hist = histograms[0][b, :new_n_bins]

        normalization = np.linalg.norm(
            seq_bp_rot, axis=1
        )  # Take normalize of the poisson distribution octave band filtered array on the axis 1 -> shape (72|71) if input is of size (72,64)

        # Only those indices which have normalization greater than 0.0
        indices = normalization > 0.0

        seq_bp_rot[indices, :] /= normalization[indices, None]

        seq_bp_rot *= np.sqrt(hist[:, None])

        # Normalize the band power
        # The bands should normally sum up to fs / 2

        seq_bp *= np.sqrt(bw / fs * 2.0)

        # Impulse response for every octave band for each microphone
        rir[fdl2:] += seq_bp

    return rir
