
from __future__ import division, print_function
from unittest import TestCase
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# test parameters
tol = 1e-6
np.random.seed(0)

# filter to apply
h_len = 99
h = np.ones(h_len)
h /= np.linalg.norm(h)

# test signal (noise)
x = np.random.randn(100000)
fs = 8000  # dummy sampling frequency

# convolved signal
y = fftconvolve(x, h)

def test_no_overlap_no_filter():

    # parameters
    block_size = 512  # make sure the FFT size is a power of 2
    hop = block_size  # no overlap

    # Create the STFT object
    stft = pra.realtime.STFT(block_size, fs, hop=hop)

    # collect the processed blocks
    processed_x = np.zeros(x.shape)

    # process the signals while full blocks are available
    n = 0
    while  x.shape[0] - n > hop:

        # go to frequency domain
        stft.analysis(x[n:n+hop])

        # copy processed block in the output buffer
        processed_x[n:n+hop] = stft.synthesis()

        n += hop

    error = np.max(np.abs(x[:n] - processed_x[:n]))

    return error

def test_no_overlap_with_filter():

    # parameters
    block_size = 512 - h_len + 1  # make sure the FFT size is a power of 2
    hop = block_size  # no overlap

    # Create the STFT object
    stft = pra.realtime.STFT(block_size, fs, hop=hop)
    
    # setup the filter
    stft.set_filter(h, zb=h_len - 1)

    # collect the processed blocks
    processed_x = np.zeros(x.shape)

    # process the signals while full blocks are available
    n = 0
    while  x.shape[0] - n > hop:

        # go to frequency domain
        stft.analysis(x[n:n+hop])

        stft.process()  # apply the filter

        # copy processed block in the output buffer
        processed_x[n:n+hop] = stft.synthesis()

        n += hop

    error = np.max(np.abs(y[:n] - processed_x[:n]))

    return error

def test_with_overlap_no_filter():

    # parameters
    block_size = 512  # make sure the FFT size is a power of 2
    hop = block_size // 2  # half overlap
    window = pra.hann(block_size)  # the analysis window

    # Create the STFT object
    stft = pra.realtime.STFT(block_size, fs, hop=hop, analysis_window=window)

    # collect the processed blocks
    processed_x = np.zeros(x.shape)

    # process the signals while full blocks are available
    n = 0
    while  x.shape[0] - n > hop:

        # go to frequency domain
        stft.analysis(x[n:n+hop])

        # copy processed block in the output buffer
        processed_x[n:n+hop] = stft.synthesis()

        n += hop

    error = np.max(np.abs(x[:n] - processed_x[:n]))

    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(x[:n])
    plt.plot(processed_x[:n])
    
    plt.subplot(2,1,2)
    plt.plot(np.abs(x[:n] - processed_x[:n]))

    plt.show()

    return error


class TestSTFT(TestCase):

    def test_no_overlap_no_filter(self):
        error = test_no_overlap_no_filter()
        self.assertTrue(error < tol)

    def test_no_overlap_with_filter(self):
        error = test_no_overlap_with_filter()
        self.assertTrue(error < tol)

if __name__ == "__main__":

    error = test_no_overlap_no_filter()
    print('no overlap, no filter:', error)

    error = test_no_overlap_with_filter()
    print('no overlap, with filter:', error)

    error = test_with_overlap_no_filter()
    print('with overlap, no filter:', error)
