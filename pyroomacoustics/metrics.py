import os
import platform

import numpy as np
from scipy.stats import binom as _binom
from scipy.stats import norm as _norm
from scipy.signal import hann

from .transform import stft


def median(x, alpha=None, axis=-1, keepdims=False):
    """
    Computes 95% confidence interval for the median.

    Parameters
    ----------
    x: array_like
        the data array
    alpha: float, optional
        the confidence level of the interval, confidence intervals are only computed
        when this argument is provided
    axis: int, optional
        the axis of the data on which to operate, by default the last axis

    Returns
    -------
    :tuple ``(float, [float, float])``
        This function returns ``(m, [le, ue])`` and the confidence interval is ``[m-le, m+ue]``.
    """

    # place the axis on which to compute median in first position
    xsw = np.moveaxis(x, axis, 0)

    # sort the array
    xsw = np.sort(xsw, axis=0)
    n = xsw.shape[0]

    if n % 2 == 1:
        # if n is odd, take central element
        m = xsw[
            n // 2,
        ]
    else:
        # if n is even, average the two central elements
        m = 0.5 * (
            xsw[
                n // 2 - 1,
            ]
            + xsw[
                n // 2,
            ]
        )

    if alpha is None:
        if keepdims:
            m = np.moveaxis(
                m[
                    np.newaxis,
                ],
                0,
                axis,
            )
        return m

    else:
        # bound for using the large n approximation
        clt_bound = max(10 / alpha, 10 / (2 - alpha))

        if n < clt_bound:
            # Get the bounds of the CI from the binomial distribution
            b = _binom(n, 0.5)
            j, k = int(b.ppf(alpha / 2) - 1), int(b.ppf(1 - alpha / 2) - 1)

            if b.cdf(k) - b.cdf(j) < 1 - alpha:
                k += 1

            # sanity check
            assert b.cdf(k) - b.cdf(j) >= 1 - alpha

            if j < 0:
                raise ValueError(
                    "Warning: Sample size is too small. No confidence interval found."
                )
            else:
                ci = np.array(
                    [
                        xsw[
                            j,
                        ]
                        - m,
                        xsw[
                            k,
                        ]
                        - m,
                    ]
                )

        else:
            # we use the Normal approximation for large sets
            norm = _norm()
            eta = norm.ppf(1 - alpha / 2)
            j = int(np.floor(0.5 * n - 0.5 * eta * np.sqrt(n))) - 1
            k = int(np.ceil(0.5 * n + 0.5 * eta * np.sqrt(n)))
            ci = np.array(
                [
                    xsw[
                        j,
                    ]
                    - m,
                    xsw[
                        k,
                    ]
                    - m,
                ]
            )

        if keepdims:
            m = np.moveaxis(
                m[
                    np.newaxis,
                ],
                0,
                axis,
            )
            if axis < 0:
                ci = np.moveaxis(
                    ci[
                        :,
                        np.newaxis,
                    ],
                    1,
                    axis,
                )
            else:
                ci = np.moveaxis(
                    ci[
                        :,
                        np.newaxis,
                    ],
                    1,
                    axis + 1,
                )

        return m, ci


# Simple mean squared error function
def mse(x1, x2):
    r"""
    A short hand to compute the mean-squared error of two signals.

    .. math::

       MSE = \\frac{1}{n}\sum_{i=0}^{n-1} (x_i - y_i)^2


    :arg x1: (ndarray)
    :arg x2: (ndarray)
    :returns: (float) The mean of the squared differences of x1 and x2.
    """
    return (np.abs(x1 - x2) ** 2).sum() / len(x1)


# Itakura-Saito distance function
def itakura_saito(x1, x2, sigma2_n, stft_L=128, stft_hop=128):

    P1 = np.abs(stft.analysis(x1, stft_L, stft_hop)) ** 2
    P2 = np.abs(stft(x2, stft_L, stft_hop)) ** 2

    VAD1 = P1.mean(axis=1) > 2 * stft_L**2 * sigma2_n
    VAD2 = P2.mean(axis=1) > 2 * stft_L**2 * sigma2_n
    VAD = np.logical_or(VAD1, VAD2)

    if P1.shape[0] != P2.shape[0] or P1.shape[1] != P2.shape[1]:
        raise ValueError("Error: Itakura-Saito requires both array to have same length")

    R = P1[VAD, :] / P2[VAD, :]

    IS = (R - np.log(R) - 1.0).mean(axis=1)

    return np.median(IS)


def snr(ref, deg):

    return np.sum(ref**2) / np.sum((ref - deg) ** 2)


# Perceptual Evaluation of Speech Quality for multiple files using multiple threads
def pesq(ref_file, deg_files, Fs=8000, swap=False, wb=False, bin="./bin/pesq"):
    """
    pesq_vals = pesq(ref_file, deg_files, sample_rate=None, bin='./bin/pesq'):
    Computes the perceptual evaluation of speech quality (PESQ) metric of a degraded
    file with respect to a reference file.  Uses the utility obtained from ITU
    P.862 http://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en

    :arg ref_file:    The filename of the reference file.
    :arg deg_files:   A list of degraded sound files names.
    :arg sample_rate: Sample rates of the sound files [8kHz or 16kHz, default 8kHz].
    :arg swap:        Swap byte orders (whatever that does is not clear to me) [default: False].
    :arg wb:          Use wideband algorithm [default: False].
    :arg bin:         Location of pesq executable [default: ./bin/pesq].

    :returns: (ndarray size 2xN) ndarray containing Raw MOS and MOS LQO in rows 0 and 1,
        respectively, and has one column per degraded file name in deg_files.
    """

    if isinstance(deg_files, str):
        deg_files = [deg_files]

    if platform.system() == "Windows":
        bin = bin + ".exe"

    if not os.path.isfile(ref_file):
        raise ValueError("Some file did not exist")
    for f in deg_files:
        if not os.path.isfile(f):
            raise ValueError("Some file did not exist")

    if Fs not in (8000, 16000):
        raise ValueError("sample rate must be 8000 or 16000")

    args = [bin, "+%d" % int(Fs)]

    if swap is True:
        args.append("+swap")

    if wb is True:
        args.append("+wb")

    args.append(ref_file)

    # array to receive all output values
    pesq_vals = np.zeros((2, len(deg_files)))

    # launch pesq for each degraded file in a different process
    import subprocess

    pipes = [
        subprocess.Popen(args + [deg], stdout=subprocess.PIPE) for deg in deg_files
    ]
    states = np.ones(len(pipes), dtype=np.bool)

    # Recover output as the processes finish
    while states.any():

        for i, p in enumerate(pipes):
            if states[i] == True and p.poll() is not None:
                states[i] = False
                out = p.stdout.readlines()
                last_line = out[-1][:-2]

                if wb is True:
                    if not last_line.startswith("P.862.2 Prediction"):
                        raise ValueError(last_line)
                    pesq_vals[:, i] = np.array([0, float(last_line.split()[-1])])
                else:
                    if not last_line.startswith("P.862 Prediction"):
                        raise ValueError(last_line)
                    pesq_vals[:, i] = np.array(map(float, last_line.split()[-2:]))

    return pesq_vals


def sweeping_echo_measure(rir, fs, t_min=0, t_max=0.5, fb=400):
    """
    Measure of sweeping echo in RIR obtained from image-source method.
    A higher value indicates less sweeping echoes

    For details see : De Sena et al. "On the modeling of rectangular geometries in
    room acoustic simulations", IEEE TASLP, 2015


    Parameters
    ----------

    rir:    RIR signal from ISM (mono).
    fs:     sampling frequency.
    t_min:  TYPE, optional
            Minimum time window. The default is 0.
    t_max:  TYPE, optional
            Maximum time window. The default is 0.5.
    fb:     TYPE, optional
            Mask bandwidth. The default is 400.

    Returns
    -------
    sweeping spectrum flatness (ssf)

    """

    # some default values
    fmin = 50
    fmax = 0.9 * fs / 2

    # STFT parameters
    fft_size = 512  # fft size for analysis
    fft_hop = 256  # hop between analysis frame
    fft_zp = 2**12 - fft_size  # zero padding
    analysis_window = hann(fft_size)

    # calculate stft
    S = stft.analysis(rir, fft_size, fft_hop, win=analysis_window, zp_back=fft_zp)

    (nFrames, nFreqs) = np.shape(S)
    nFreqs = int(nFreqs / 2)

    # ignore negative frequencies
    S = S[:, :nFreqs]

    timeSlice = np.arange(0, nFrames) * fft_hop / fs
    assert nFrames == len(timeSlice)
    freqSlice = np.linspace(0, fs / 2, nFreqs)
    assert nFreqs == len(freqSlice)

    # get time-frequency grid
    (t_mesh, f_mesh) = np.meshgrid(timeSlice, freqSlice)

    bmin = int(np.floor(nFreqs / fs * fmin))
    bmax = int(np.ceil(nFreqs / fs * fmax))

    Phi = np.zeros(np.shape(S))

    # normalize spectrogram to make energy identical in each bin
    for k in range(nFrames):
        norm_factor = np.sum(np.power(np.abs(S[k, bmin:bmax]), 2))
        Phi[k, :] = np.abs(S[k, :]) / np.sqrt(norm_factor)

    # slope values
    nCoeffs = 500
    coeffs = np.linspace(5000, 150000, nCoeffs)
    ss = np.zeros(nCoeffs)

    # loop through different slope values
    for k in range(nCoeffs):

        # get masks
        a = coeffs[k]

        maskTime = np.logical_and(t_mesh > t_min, t_mesh < t_max)

        maskFreq = np.logical_and(
            f_mesh > t_mesh * a - fb / 2, f_mesh < t_mesh * a + fb / 2
        )

        boolMask = np.logical_and(maskTime, maskFreq)

        ss[k] = np.mean(Phi[boolMask.T])

    # calculate spectral flatness
    ssf = np.exp(np.mean(np.log(np.abs(ss)))) / np.mean(np.abs(ss))

    return ssf
