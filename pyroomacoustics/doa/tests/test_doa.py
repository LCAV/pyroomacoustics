# @version: 1.0  date: 2017/06/20 by Robin Scheibler
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2017

import unittest

import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve

# fix the RNG seed for repeatability
np.random.seed(0)

# Location of original source
azimuth = 61.0 / 180.0 * np.pi  # 60 degrees
tol = 0.3 / 180.0 * np.pi  # 0.3 degrees tolerance for the test

# algorithms parameters
c = 343.0
fs = 16000
nfft = 256
freq_bins = np.arange(5, 60)

# circular microphone array, 12 mics, radius 15 cm
R = pra.circular_2D_array([0, 0], 12, 0.0, 0.15)

# propagation filter bank
propagation_vector = -np.array([np.cos(azimuth), np.sin(azimuth)])
delays = np.dot(R.T, propagation_vector) / c * fs  # in fractional samples
filter_bank = pra.fractional_delay_filter_bank(delays)

# we use a white noise signal for the source
x = np.random.randn((nfft // 2 + 1) * nfft)

# convolve the source signal with the fractional delay filters
# to get the microphone input signals
mic_signals = np.array([fftconvolve(x, filter, mode="same") for filter in filter_bank])
X = pra.transform.stft.analysis(mic_signals.T, nfft, nfft // 2, win=np.hanning(nfft))
X = np.swapaxes(X, 2, 0)


class TestDOA(unittest.TestCase):
    def test_music(self):
        doa = pra.doa.algorithms["MUSIC"](R, fs, nfft, c=c)
        doa.locate_sources(X, freq_bins=freq_bins)
        print("distance:", pra.doa.circ_dist(azimuth, doa.azimuth_recon))
        self.assertTrue(pra.doa.circ_dist(azimuth, doa.azimuth_recon) < tol)

    def test_normmusic(self):
        doa = pra.doa.algorithms["NormMUSIC"](R, fs, nfft, c=c)
        doa.locate_sources(X, freq_bins=freq_bins)
        print("distance:", pra.doa.circ_dist(azimuth, doa.azimuth_recon))
        self.assertTrue(pra.doa.circ_dist(azimuth, doa.azimuth_recon) < tol)

    def test_srp_phat(self):
        doa = pra.doa.algorithms["SRP"](R, fs, nfft, c=c)
        doa.locate_sources(X, freq_bins=freq_bins)
        print("distance:", pra.doa.circ_dist(azimuth, doa.azimuth_recon))
        self.assertTrue(pra.doa.circ_dist(azimuth, doa.azimuth_recon) < tol)

    def test_srp_phat_2ch(self):
        """
        Tests that the cost function of SRP is properly computed (PR #197)

        This checks for the presence of a bug that was detected in PR #197.
        """
        doa = pra.doa.algorithms["SRP"](R[:, :2], fs, nfft, c=c)
        i_freq = 10
        doa.locate_sources(X[:2, :, :], freq_bins=freq_bins[i_freq : i_freq + 1])
        std_val = np.std(doa.grid.values)
        self.assertTrue(std_val > 1e-10)

    def test_cssm(self):
        doa = pra.doa.algorithms["CSSM"](R, fs, nfft, c=c)
        doa.locate_sources(X, freq_bins=freq_bins)
        print("distance:", pra.doa.circ_dist(azimuth, doa.azimuth_recon))
        self.assertTrue(pra.doa.circ_dist(azimuth, doa.azimuth_recon) < tol)

    def test_tops(self):
        doa = pra.doa.algorithms["TOPS"](R, fs, nfft, c=c)
        doa.locate_sources(X, freq_bins=freq_bins)
        print("distance:", pra.doa.circ_dist(azimuth, doa.azimuth_recon))
        self.assertTrue(pra.doa.circ_dist(azimuth, doa.azimuth_recon) < tol)

    def test_waves(self):
        doa = pra.doa.algorithms["WAVES"](R, fs, nfft, c=c)
        doa.locate_sources(X, freq_bins=freq_bins)
        print("distance:", pra.doa.circ_dist(azimuth, doa.azimuth_recon))
        self.assertTrue(pra.doa.circ_dist(azimuth, doa.azimuth_recon) < tol)

    def test_frida(self):
        doa = pra.doa.algorithms["FRIDA"](R, fs, nfft, c=c)
        doa.locate_sources(X, freq_bins=freq_bins)
        print("distance:", pra.doa.circ_dist(azimuth, doa.azimuth_recon))
        self.assertTrue(pra.doa.circ_dist(azimuth, doa.azimuth_recon) < tol)


if __name__ == "__main__":

    """
    algo_names = sorted(pra.doa.algorithms.keys())

    for algo_name in algo_names:
        doa = pra.doa.algorithms[algo_name](R, fs, nfft, c=c, max_four=4)
        doa.locate_sources(X, freq_bins=freq_bins)
        print(
            algo_name,
            doa.azimuth_recon / np.pi * 180.0,
            pra.doa.circ_dist(azimuth, doa.azimuth_recon) / np.pi * 180.0,
        )
    """

    unittest.main()
