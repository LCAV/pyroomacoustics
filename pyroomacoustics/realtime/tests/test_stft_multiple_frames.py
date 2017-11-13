from __future__ import division, print_function
from unittest import TestCase
import numpy as np
from scipy.signal import fftconvolve
import pyroomacoustics as pra

'''
We create a signal, a simple filter and compute their convolution.

Then we test STFT block procesing on multiple frames with and without overlap,
and with and without filtering.
'''

# test parameters
tol = 1e-6
np.random.seed(0)
D = 4
block_size = 512
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
    hop = block_size

    # create STFT object
    stft = pra.realtime.STFT(block_size, hop=hop,
        channels=D, 
        transform=transform)

    # multiple frames all at once
    mX = stft.analysis_multiple(x_local)
    x_r = stft.synthesis_multiple(mX)

    n = x_r.shape[0]
    return np.max(np.abs(x_local[:n,] - x_r[:n,]))


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

    # multiple frames all at once
    mX = stft.analysis_multiple(x_local)
    mX = stft.process_multiple(mX)
    y_r = stft.synthesis_multiple(mX)

    # compute error
    n = y_r.shape[0]
    return np.max(np.abs(y_local[:n,] - y_r[:n,]))



def half_overlap_no_filter(D):

    if D == 1:
        x_local = x[:,0]
    else:
        x_local = x[:,:D]

    # parameters
    hop = block_size//2
    window = pra.hann(block_size)  # the analysis window

    # create STFT object
    stft = pra.realtime.STFT(block_size, hop=hop, 
        analysis_window=window, 
        channels=D, 
        transform=transform)

    # multiple frames all at once
    mX = stft.analysis_multiple(x_local)
    x_r = stft.synthesis_multiple(mX)

    n = x_r.shape[0]
    return np.max(np.abs(x_local[block_size-hop:n,] 
        - x_r[block_size-hop:n,]))


def half_overlap_with_filter(D):


    if D == 1:
        x_local = x[:,0]
        y_local = y[:,0]
        h_local = h[:,0]
    else:
        x_local = x[:,:D]
        y_local = y[:,:D]
        h_local = h[:,:D]


    block_size = 512 - h_len + 1  # make sure the FFT size is a power of 2
    hop = block_size // 2  # half overlap
    window = pra.hann(block_size)  # the analysis window

    # Create the STFT object
    stft = pra.realtime.STFT(block_size, hop=hop, analysis_window=window, 
        channels=D, transform=transform)

    # setup the filter
    stft.set_filter(h_local, zb=h_len - 1)

    # multiple frames all at once
    mX = stft.analysis_multiple(x_local)
    mX = stft.process_multiple(mX)
    y_r = stft.synthesis_multiple(mX)

    # compute error
    n = y_r.shape[0]
    return np.max(np.abs(y_local[stft.nfft-hop:n,] 
        - y_r[stft.nfft-hop:n,]))



class TestSTFTMultple(TestCase):

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
        error = half_overlap_no_filter(1)
        self.assertTrue(error < tol)

    def test_with_half_overlap_no_filter_multichannel(self):
        error = half_overlap_no_filter(D)
        self.assertTrue(error < tol)

    def test_with_half_overlap_with_filter_mono(self):
        error = half_overlap_with_filter(1)
        self.assertTrue(error < tol)

    def test_with_half_overlap_with_filter_multichannel(self):
        error = half_overlap_with_filter(D)
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

    error = half_overlap_no_filter(1)
    print('half overlap, no filter, mono:', error)

    error = half_overlap_no_filter(D)
    print('half overlap, no filter, multichannel:', error)

    error = half_overlap_with_filter(1)
    print('half overlap, with filter, mono:', error)

    error = half_overlap_with_filter(D)
    print('half overlap, with filter, multichannel:', error)



    

