from __future__ import division, print_function
from unittest import TestCase
import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.transform import analysis, synthesis


# test parameters
tol = -100  # dB
np.random.seed(0)
D = 4
block_size = 512

# test signal (noise)
x = np.random.randn(block_size*100, D).astype(np.float32)


def no_overlap(D):

    if D == 1:
        x_local = x[:, 0]
    else:
        x_local = x[:, :D]

    hop = block_size

    # analysis
    X = analysis(x_local, L=block_size, hop=hop)

    # synthesis
    x_r = synthesis(X, L=block_size, hop=hop)

    return pra.dB(np.max(np.abs(x_local - x_r)))


def half_overlap(D):

    if D == 1:
        x_local = x[:, 0]
    else:
        x_local = x[:, :D]

    hop = block_size//2

    # analysis
    analysis_win = pra.hann(block_size)
    X = analysis(x_local, L=block_size, hop=hop, win=analysis_win)

    # synthesis
    x_r = synthesis(X, L=block_size, hop=hop)

    return pra.dB(np.max(np.abs(x_local[:-hop, ] - x_r[hop:, ])))


class TestSTFTOneShot(TestCase):

    def test_no_overlap(self):
        self.assertTrue(no_overlap(1) < tol)
        self.assertTrue(no_overlap(D) < tol)

    def test_half_overlap(self):
        self.assertTrue(half_overlap(1) < tol)
        self.assertTrue(half_overlap(D) < tol)


if __name__ == "__main__":

    print()
    print("TEST INFO")
    print("-------------------------------------------------------------")
    print("Max error in dB for randomly generated signal of %d samples." % len(x))
    print("Multichannel corresponds to %d channels." % D)
    print("-------------------------------------------------------------")
    print()

    err = no_overlap(1)
    print("No overlap, mono             : %d dB" % err)
    err = no_overlap(D)
    print("No overlap, multichannel     : %d dB" % err)

    err = half_overlap(1)
    print("No overlap, mono             : %d dB" % err)
    err = half_overlap(D)
    print("Half overlap, multichannel   : %d dB" % err)
