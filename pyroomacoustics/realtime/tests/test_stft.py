
from __future__ import division, print_function
from unittest import TestCase
import numpy as np
from scipy.signal import fftconvolve
import pyroomacoustics as pra

'''
We create a signal, a simple filter and compute their convolution.

Then we test STFT block procesing with and without overlap,
and with and without filtering.
'''

# test parameters
tol = 1e-6
np.random.seed(0)
D = 4
transform = 'numpy'   # 'numpy', 'pyfftw', or 'mkl'

# filter to apply
h_len = 99
h = np.ones((h_len, D))
h /= np.linalg.norm(h, axis=0)

# test signal (noise)
x = np.random.randn(100000, D)

# convolved signal
y = np.zeros((x.shape[0] + h_len - 1, x.shape[1]))
for i in range(x.shape[1]):
    y[:,i] = fftconvolve(x[:,i], h[:,i])

def no_overlap_no_filter(D):

    if D == 1:
        x_local = x[:,0]
    else:
        x_local = x[:,:D]

    # parameters
    block_size = 512  # make sure the FFT size is a power of 2
    hop = block_size  # no overlap

    # Create the STFT object
    stft = pra.realtime.STFT(block_size, hop=hop, channels=D, 
        transform=transform)

    # collect the processed blocks
    processed_x = np.zeros(x_local.shape)

    # process the signals while full blocks are available
    n = 0
    while  x_local.shape[0] - n > hop:

        # go to frequency domain
        stft.analysis(x_local[n:n+hop,])

        # copy processed block in the output buffer
        processed_x[n:n+hop,] = stft.synthesis()

        n += hop

    error = np.max(np.abs(x_local[:n,] - processed_x[:n,]))

    return error

def no_overlap_with_filter(D):
    if D == 1:
        x_local = x[:,0]
        y_local = y[:,0]
        h_local = h[:,0]
    else:
        x_local = x[:,:D]
        y_local = y[:,:D]
        h_local = h[:,:D]


    # parameters
    block_size = 512 - h_len + 1  # make sure the FFT size is a power of 2
    hop = block_size  # no overlap

    # Create the STFT object
    stft = pra.realtime.STFT(block_size, hop=hop, channels=D, 
        transform=transform)
    
    # setup the filter
    stft.set_filter(h_local, zb=h_len - 1)

    # collect the processed blocks
    processed_x = np.zeros(x_local.shape)

    # process the signals while full blocks are available
    n = 0
    while  x_local.shape[0] - n > hop:

        # go to frequency domain
        stft.analysis(x_local[n:n+hop,])

        stft.process()  # apply the filter

        # copy processed block in the output buffer
        processed_x[n:n+hop,] = stft.synthesis()

        n += hop

    error = np.max(np.abs(y_local[:n,] - processed_x[:n,]))

    return error

def with_half_overlap_no_filter(D):

    if D == 1:
        x_local = x[:,0]
    else:
        x_local = x[:,:D]

    # parameters
    block_size = 512  # make sure the FFT size is a power of 2
    hop = block_size // 2  # half overlap
    window = pra.hann(block_size)  # the analysis window

    # Create the STFT object
    stft = pra.realtime.STFT(block_size, hop=hop, analysis_window=window, 
        channels=D, transform=transform)

    # collect the processed blocks
    processed_x = np.zeros(x_local.shape)

    # process the signals while full blocks are available
    n = 0
    while  x_local.shape[0] - n > hop:

        # go to frequency domain
        stft.analysis(x_local[n:n+hop,])

        # copy processed block in the output buffer
        processed_x[n:n+hop,] = stft.synthesis()

        n += hop

    error = np.max(np.abs(x_local[:n-hop,] - processed_x[hop:n,]))

    return error

def with_half_overlap_with_filter(D):

    if D == 1:
        x_local = x[:,0]
        y_local = y[:,0]
        h_local = h[:,0]
    else:
        x_local = x[:,:D]
        y_local = y[:,:D]
        h_local = h[:,:D]

    # parameters
    block_size = 512 - h_len + 1  # make sure the FFT size is a power of 2
    hop = block_size // 2  # half overlap
    window = pra.hann(block_size)  # the analysis window

    # Create the STFT object
    stft = pra.realtime.STFT(block_size, hop=hop, analysis_window=window, 
        channels=D, transform=transform)

    # setup the filter
    stft.set_filter(h_local, zb=h_len - 1)

    # collect the processed blocks
    processed_x = np.zeros(x_local.shape)

    # process the signals while full blocks are available
    n = 0
    while  x.shape[0] - n > hop:

        # go to frequency domain
        stft.analysis(x_local[n:n+hop,])

        stft.process()  # filtering

        # copy processed block in the output buffer
        processed_x[n:n+hop,] = stft.synthesis()

        n += hop

    error = np.max(np.abs(y_local[:n-hop,] - processed_x[hop:n,]))

    return error


class TestSTFT(TestCase):

    def test_no_overlap_no_filter_mono(self):
        error = no_overlap_no_filter(1)
        self.assertTrue(error < tol)

    def test_no_overlap_no_filter_multichannel(self):
        error = no_overlap_no_filter(D)
        self.assertTrue(error < tol)

    def test_no_overlap_with_filter_mono(self):
        error = no_overlap_with_filter(1)
        self.assertTrue(error < tol)

    def test_no_overlap_with_filter_multichannel(self):
        error = no_overlap_with_filter(D)
        self.assertTrue(error < tol)

    def test_with_half_overlap_no_filter_mono(self):
        error = with_half_overlap_no_filter(1)
        self.assertTrue(error < tol)

    def test_with_half_overlap_no_filter_multichannel(self):
        error = with_half_overlap_no_filter(D)
        self.assertTrue(error < tol)

    def test_with_half_overlap_with_filter_mono(self):
        error = with_half_overlap_with_filter(1)
        self.assertTrue(error < tol)

    def test_with_half_overlap_with_filter_multichannel(self):
        error = with_half_overlap_with_filter(D)
        self.assertTrue(error < tol)

if __name__ == "__main__":

    error = no_overlap_no_filter(1)
    print('no overlap, no filter, mono:', error)

    error = no_overlap_no_filter(D)
    print('no overlap, no filter, multichannel:', error)

    error = no_overlap_with_filter(1)
    print('no overlap, with filter, mono:', error)

    error = no_overlap_with_filter(D)
    print('no overlap, with filter, multichannel:', error)

    error = with_half_overlap_no_filter(1)
    print('with half overlap, no filter, mono:', error)

    error = with_half_overlap_no_filter(D)
    print('with half overlap, no filter, multichannel:', error)

    error = with_half_overlap_with_filter(1)
    print('with half overlap, with filter, mono:', error)

    error = with_half_overlap_with_filter(D)
    print('with half overlap, with filter, multichannel:', error)
