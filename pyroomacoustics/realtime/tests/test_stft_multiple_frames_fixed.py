from __future__ import division, print_function
from unittest import TestCase
import numpy as np
from scipy.signal import fftconvolve
import pyroomacoustics as pra

import gc

'''
We create a signal, a simple filter and compute their convolution.

Then we test STFT block procesing on multiple frames with and without overlap,
and with and without filtering.

The optional parameter 'num_frames' is given to the STFT constructor so that 
memory is allocated specifically for a particular input size [num_frames*hop].
Otherwise an error will be raised.
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


def incorrect_input_size(D):

    if D == 1:
        x_local = x[:,0]
    else:
        x_local = x[:,:D]

    # parameters
    block_size = 512
    hop = block_size

    # create STFT object
    num_frames = 100
    stft = pra.realtime.STFT(block_size, hop=hop,
        channels=D, 
        transform=transform,
        num_frames=num_frames)

    try:  # passing more frames than 'num_frames'
        stft.analysis(x_local)
        computed = False
    except:
        computed = True

    return computed


def no_overlap_no_filter(D):

    if D == 1:
        x_local = x[:,0]
    else:
        x_local = x[:,:D]

    # parameters
    block_size = 512
    hop = block_size

    # create STFT object
    num_frames = 100
    stft = pra.realtime.STFT(block_size, hop=hop,
        channels=D, 
        transform=transform,
        num_frames=num_frames)

    # multiple frames all at once
    stft.analysis(x_local[:num_frames*hop,])
    x_r = stft.synthesis()

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
    num_frames = 100
    stft = pra.realtime.STFT(block_size, hop=hop,
        channels=D, 
        transform=transform,
        num_frames=num_frames)
    
    # setup the filter
    stft.set_filter(h_local, zb=h_len - 1)

    # multiple frames all at once
    stft.analysis(x_local[:num_frames*hop,])
    stft.process()
    y_r = stft.synthesis()

    # compute error
    n = y_r.shape[0]
    return np.max(np.abs(y_local[:n,] - y_r[:n,]))



def half_overlap_no_filter(D):

    if D == 1:
        x_local = x[:,0]
    else:
        x_local = x[:,:D]

    # parameters
    block_size = 512
    hop = block_size//2
    window = pra.hann(block_size)  # the analysis window

    # create STFT object
    num_frames = 100
    stft = pra.realtime.STFT(block_size, hop=hop, 
        analysis_window=window, 
        channels=D, 
        transform=transform,
        num_frames=num_frames)

    # multiple frames all at once
    stft.analysis(x_local[:num_frames*hop,])
    x_r = stft.synthesis()

    # if D==1:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.plot(x_local[:num_frames*hop,])
    #     plt.plot(x_r)
    #     # plt.plot(np.abs(x_local[:num_frames*hop,]-x_r))
    #     plt.show()

    n = x_r.shape[0]
    error = np.max(np.abs(x_local[block_size-hop:n-hop,] 
        - x_r[block_size-hop:n-hop,]))

    return error 


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
    num_frames = 100
    stft = pra.realtime.STFT(block_size, hop=hop, 
        analysis_window=window, 
        channels=D, 
        transform=transform,
        num_frames=num_frames)

    # setup the filter
    stft.set_filter(h_local, zb=h_len - 1)

    # multiple frames all at once
    stft.analysis(x_local[:num_frames*hop,])
    stft.process()
    y_r = stft.synthesis()

    # if D==1:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.plot(x_local)
    #     plt.plot(y_r)
    #     plt.show()

    # compute error
    n = y_r.shape[0]
    return np.max(np.abs(y_local[block_size:n-block_size,] 
        - y_r[block_size:n-block_size,]))



class TestSTFTMultpleFixed(TestCase):

    def test_incorrect_input_check_mono(self):
        result = incorrect_input_size(1)
        self.assertTrue(result)

    def test_incorrect_input_check_stereo(self):
        result = incorrect_input_size(D)
        self.assertTrue(result)

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

    result = incorrect_input_size(1)
    print('incorrect input size, mono:', result)

    result = incorrect_input_size(D)
    print('incorrect input size, multichannel:', result)

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

    gc.collect()



    

