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
memory is allocated specifically for a particular input size. Otherwise an 
error will be raised.
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
num_frames = 0    # only passing [hop] samples (real-time case), set to 0

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
    stft = pra.realtime.STFT(block_size, hop=hop,
        channels=D, 
        transform=transform,
        num_frames=num_frames)

    try:  # passing more frames than 'hop'
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
    stft = pra.realtime.STFT(block_size, hop=hop,
        channels=D, 
        transform=transform,
        num_frames=num_frames)

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

    # if D==1:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.plot(x_local)
    #     plt.plot(processed_x)
    #     plt.show()

    # last hop samples not processed since didn't get full frame
    processed_x = processed_x[(block_size-hop):-hop]
    n = processed_x.shape[0]
    return np.max(np.abs(x_local[:n,] - processed_x[:n,]))


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
    stft = pra.realtime.STFT(block_size, hop=hop,
        channels=D, 
        transform=transform,
        num_frames=num_frames)

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

    # if D==1:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.plot(y_local)
    #     plt.plot(processed_x)
    #     plt.show()

    # last hop samples not processed since didn't get full frame
    processed_x = processed_x[(block_size-hop):-hop]
    n = processed_x.shape[0]
    return np.max(np.abs(y_local[:n,] - processed_x))



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
    stft = pra.realtime.STFT(block_size, hop=hop, 
        analysis_window=window, 
        channels=D, 
        transform=transform,
        num_frames=num_frames)

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

    # if D==1:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.plot(x_local)
    #     plt.plot(processed_x[(block_size-hop):-hop])
    #     plt.show()

    # first [block_size-hop] samples not processed since need to accumulate full frame
    processed_x = processed_x[(block_size-hop):-hop]
    n = processed_x.shape[0]
    return np.max(np.abs(x_local[:n,] - processed_x[:n,]))


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
    stft = pra.realtime.STFT(block_size, hop=hop, 
        analysis_window=window, 
        channels=D, 
        transform=transform,
        num_frames=num_frames)

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

    # if D==1:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.plot(y_local)
    #     plt.plot(processed_x[(block_size-hop):-hop])
    #     plt.show()

    # last hop samples not processed since didn't get full frame
    processed_x = processed_x[(block_size-hop):-hop]
    n = processed_x.shape[0]
    return np.max(np.abs(y_local[:n,] - processed_x))



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



    

