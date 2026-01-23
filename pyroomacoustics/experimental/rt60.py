# Copyright (c) 2018-2019 Robin Scheibler
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
"""
RT60 Measurement Routine
========================

Automatically determines the reverberation time of an impulse response
using the Schroeder method [1]_.

References
----------

.. [1] M. R. Schroeder, "New Method of Measuring Reverberation Time,"
    J. Acoust. Soc. Am., vol. 37, no. 3, pp. 409-412, Mar. 1968.
"""
import math
import numpy as np
from scipy.optimize import curve_fit


def _fit_exp_and_extrapolate(
    data_db, fs, extrapolate_value_db=-60.0, linear_domain_fit=False
):
    """Non-linear fit to an exponential f(t) = b * np.exp(a * t)."""
    # Fix the origin.
    data = data_db - data_db[0]

    N = data.shape[0]
    t = np.arange(N) / fs

    # We use a least-square fit in log-domain as initialization.
    X = np.column_stack((t, np.ones(N)))
    p, *_ = np.linalg.lstsq(X, data)

    if not linear_domain_fit:
        return extrapolate_value_db / p[0]

    # Non-linear fit using scipy.optimize.curve_fit.

    def model(x, a, b):
        return b * np.exp(a * x)

    def jac(x, a, b):
        u = np.exp(a * x)
        return np.column_stack((x * b * u, u))

    # Provide an initial guess [a, b]
    initial_guess = [p[0] / (10.0 * np.log10(np.e)), 10.0 ** (p[1] / 10.0)]

    # Perform the fit in the linear domain.
    # Empirically, this takes less than 10 iterations.
    popt, pcov = curve_fit(model, t, 10.0 ** (data / 10.0), p0=initial_guess, jac=jac)

    # This is the estimate of the T60 (or other value) based on the fit.
    return (extrapolate_value_db / 10.0) * np.log(10.0) / popt[0]


def measure_rt60(
    h,
    fs=1,
    decay_db=60,
    energy_thres=1.0,
    plot=False,
    rt60_tgt=None,
    label=None,
    linear_domain_fit=False,
):
    """
    Analyze the RT60 of an impulse response. Optionaly plots some useful information.

    Parameters
    ----------
    h: array_like
        The impulse response.
    fs: float or int, optional
        The sampling frequency of h (default to 1, i.e., samples).
    decay_db: float or int, optional
        The decay in decibels for which we actually estimate the time. Although
        we would like to estimate the RT60, it might not be practical. Instead,
        we measure the RT20 or RT30 and extrapolate to RT60.
    energy_thres: float
        This should be a value between 0.0 and 1.0.
        If provided, the fit will be done using a fraction energy_thres of the
        whole energy. This is useful when there is a long noisy tail for example.
    plot: bool, optional
        If set to ``True``, the power decay and different estimated values will
        be plotted (default False).
    rt60_tgt: float
        This parameter can be used to indicate a target RT60 to which we want
        to compare the estimated value.
    label: str, optional
        A label to use in a plot.
    linear_domain_fit: bool, optional
        When True, applies a direct fit of an exponential to the data in the linear domain.
        When False, a linear fit is done in the logarithm domain (default).
    """

    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h**2
    # Backward energy integration according to Schroeder
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    if energy_thres < 1.0:
        assert 0.0 < energy_thres < 1.0
        energy -= energy[0] * (1.0 - energy_thres)
        energy = np.maximum(energy, 0.0)

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    min_energy_db = -np.min(energy_db)
    if min_energy_db - 5 < decay_db:
        decay_db = min_energy_db

    # -5 dB headroom
    try:
        i_5db = np.min(np.where(energy_db < -5.0)[0])
    except ValueError:
        return 0.0
    e_5db = energy_db[i_5db]

    # after decay
    try:
        i_decay = np.min(np.where(energy_db < e_5db - decay_db)[0])
    except ValueError:
        i_decay = len(energy_db)

    # Compute the RT60 estimate using a fitting method.
    est_rt60 = _fit_exp_and_extrapolate(
        energy_db[i_5db:i_decay],
        fs,
        extrapolate_value_db=-60.0,
        linear_domain_fit=linear_domain_fit,
    )

    if plot:
        import matplotlib.pyplot as plt

        # Remove clip power below to minimum energy (for plotting purpose mostly)
        energy_min = energy[-1]
        energy_db_min = energy_db[-1]
        power[power < energy[-1]] = energy_min
        power_db = 10 * np.log10(power)
        power_db -= np.max(power_db)

        # time vector
        def get_time(x, fs):
            return np.arange(x.shape[0]) / fs - i_5db / fs

        T = get_time(power_db, fs)

        # Optional label for the legend.
        label = f" {label}" if label else ""

        # plot power and energy
        plt.plot(get_time(energy_db, fs), energy_db, label=f"Energy{label}")

        # now the linear fit
        plt.plot([0, est_rt60], [e_5db, -65], "--", label=f"Linear Fit{label}")
        plt.plot(T, np.ones_like(T) * -60, "--", label=f"-60 dB{label}")
        plt.plot(T, np.ones_like(T) * -5.0, "--", label=f"-5 dB{label}")
        plt.vlines(
            est_rt60,
            energy_db_min,
            0,
            linestyles="dashed",
            label=f"Estimated RT60{label}",
        )

        if rt60_tgt is not None:
            plt.vlines(rt60_tgt, energy_db_min, 0, label=f"Target RT60{label}")

        plt.legend()

    return est_rt60
