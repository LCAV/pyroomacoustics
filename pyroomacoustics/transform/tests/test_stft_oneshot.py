from __future__ import division, print_function
from unittest import TestCase
import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.transform import analysis, synthesis


# test parameters
tol = -100  # dB
np.random.seed(0)
D = 3
block_size = 512

# test signal (noise)
x = np.random.randn(block_size*20, D).astype(np.float32)


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

    return pra.dB(np.max(np.abs(x_local[:-block_size + hop, ] -
                                x_r[block_size - hop:, ])))


def append_one_sample(D):
    hop = block_size // 2
    n_samples = x.shape[0]
    n_frames = n_samples // hop
    x_local = x[:n_frames*hop-1, :]

    if D == 1:
        x_local = x_local[:, 0]
    else:
        x_local = x_local[:, :D]

    # analysis
    analysis_win = pra.hann(block_size)
    X = analysis(x_local, L=block_size, hop=hop, win=analysis_win)

    # synthesis
    x_r = synthesis(X, L=block_size, hop=hop)

    return pra.dB(np.max(np.abs(x_local[:-block_size + hop, ] -
                                x_r[block_size - hop:-1, ])))


def hop_one_sample(D):

    if D == 1:
        x_local = x[:, 0]
    else:
        x_local = x[:, :D]

    hop = 1

    # analysis
    analysis_win = pra.hann(block_size)
    X = analysis(x_local, L=block_size, hop=hop, win=analysis_win)

    # synthesis
    synthesis_win = pra.transform.compute_synthesis_window(analysis_win, hop)
    x_r = synthesis(X, L=block_size, hop=hop, win=synthesis_win)

    return pra.dB(np.max(np.abs(x_local[:-block_size+hop, ] -
                                x_r[block_size-hop:, ])))


class TestSTFTOneShot(TestCase):

    def test_no_overlap(self):
        self.assertTrue(no_overlap(1) < tol)
        self.assertTrue(no_overlap(D) < tol)

    def test_half_overlap(self):
        self.assertTrue(half_overlap(1) < tol)
        self.assertTrue(half_overlap(D) < tol)

    def test_append_one_sample(self):
        self.assertTrue(append_one_sample(1) < tol)
        self.assertTrue(append_one_sample(D) < tol)

    def test_hop_one_sample(self):
        self.assertTrue(hop_one_sample(1) < tol)
        self.assertTrue(hop_one_sample(D) < tol)


if __name__ == "__main__":

    print()
    print("TEST INFO")
    print("-------------------------------------------------------------")
    print("Max error in dB for randomly generated signal of %d samples." % len(x))
    print("Multichannel corresponds to %d channels." % D)
    print("-------------------------------------------------------------")
    print()

    err = no_overlap(1)
    print("No overlap, mono                : %d dB" % err)
    err = no_overlap(D)
    print("No overlap, multichannel        : %d dB" % err)

    err = half_overlap(1)
    print("No overlap, mono                : %d dB" % err)
    err = half_overlap(D)
    print("Half overlap, multichannel      : %d dB" % err)

    # check squeeze done properly
    err = append_one_sample(1)
    print("Append one zero, mono           : %d dB" % err)
    err = append_one_sample(D)
    print("Append one zero, multichannel   : %d dB" % err)

    err = hop_one_sample(1)
    print("Hop one sample, mono            : %d dB" % err)
    err = hop_one_sample(D)
    print("Hop one sample, multichannel    : %d dB" % err)
