import unittest

import numpy as np
import pyroomacoustics as pra


def make_filters(n_mics):
    # Location of original source
    azimuth = 61. / 180. * np.pi  # 60 degrees

    # algorithms parameters
    c = 343.
    fs = 16000

    # circular microphone array, 6 mics, radius 15 cm
    R = pra.circular_2D_array([0, 0], n_mics, 0., 0.15)

    # propagation filter bank
    propagation_vector = -np.array([np.cos(azimuth), np.sin(azimuth)])
    delays = np.dot(R.T, propagation_vector) / c * fs  # in fractional samples
    filter_bank = pra.fractional_delay_filter_bank(delays)

    return filter_bank


class TestIssue87(unittest.TestCase):

    def test_12_mics(self):
        # this was working
        make_filters(12)

    def test_6_mics(self):
        # but this failed
        make_filters(6)

if __name__ == '__main__':
    unittest.main()
