
from __future__ import division, print_function
from unittest import TestCase
import numpy as np
from scipy.signal import fftconvolve
import pyroomacoustics as pra

'''
We create a signal, a simple filter and compute their convolution.

Then we test STFT block procesing with and without overlap,
and with and without filtering. Simulating a case of real-time
block processing.
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


def incorrect_input_size(D, num_frames):

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



def no_overlap_no_filter(D, num_frames=None, fixed_memory=False):
    """
    D             - number of channels
    num_frames    - how many frames to process, None will process one frame at 
                    a time 
    fixed_memory  - whether to enforce checks for size (real-time consideration)
    """

    if D == 1:
        x_local = x[:,0]
    else:
        x_local = x[:,:D]

    # parameters
    block_size = 512  # make sure the FFT size is a power of 2
    hop = block_size  # no overlap
    if num_frames is not None:
        corr_num_samples = (num_frames-1)*hop + block_size
        x_local = x_local[:corr_num_samples,]

    # Create the STFT object
    if fixed_memory:
        if num_frames is not None:
            stft = pra.realtime.STFT(block_size, hop=hop, channels=D, 
                transform=transform, num_frames=num_frames)
        else:
            stft = pra.realtime.STFT(block_size, hop=hop, channels=D, 
                transform=transform, num_frames=0)
    else:
        stft = pra.realtime.STFT(block_size, hop=hop, channels=D, 
            transform=transform)

    # collect the processed blocks
    processed_x = np.zeros(x_local.shape)

    if num_frames is not None:

        stft.analysis(x_local)
        processed_x = stft.synthesis()
        n = processed_x.shape[0]

    else:

        n = 0
        # process the signals while full blocks are available
        while  x_local.shape[0] - n > hop:
            stft.analysis(x_local[n:n+hop,])
            processed_x[n:n+hop,] = stft.synthesis()
            n += hop

    error = np.max(np.abs(x_local[:n,] - processed_x[:n,]))

    return error

def no_overlap_with_filter(D, num_frames=None, fixed_memory=False):
    """
    D             - number of channels
    num_frames    - how many frames to process, None will process one frame at 
                    a time 
    fixed_memory  - whether to enforce checks for size (real-time consideration)
    """

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
    if num_frames is not None:
        corr_num_samples = (num_frames-1)*hop + block_size
        x_local = x_local[:corr_num_samples,]

    # Create the STFT object
    if fixed_memory:
        if num_frames is not None:
            stft = pra.realtime.STFT(block_size, hop=hop, channels=D, 
                transform=transform, num_frames=num_frames)
        else:
            stft = pra.realtime.STFT(block_size, hop=hop, channels=D, 
                transform=transform, num_frames=0)
    else:
        stft = pra.realtime.STFT(block_size, hop=hop, channels=D, 
            transform=transform)
    
    # setup the filter
    stft.set_filter(h_local, zb=h_len - 1)

    # collect the processed blocks
    processed_x = np.zeros(x_local.shape)

    if num_frames is not None:

        stft.analysis(x_local)
        stft.process()
        processed_x = stft.synthesis()
        n = processed_x.shape[0]

    else:

        n = 0
        # process the signals while full blocks are available
        while  x_local.shape[0] - n > hop:
            stft.analysis(x_local[n:n+hop,])
            stft.process()  # apply the filter
            processed_x[n:n+hop,] = stft.synthesis()
            n += hop

    error = np.max(np.abs(y_local[:n,] - processed_x[:n,]))

    return error


def with_half_overlap_no_filter(D, num_frames=None, fixed_memory=False):
    """
    D             - number of channels
    num_frames    - how many frames to process, None will process one frame at 
                    a time 
    fixed_memory  - whether to enforce checks for size (real-time consideration)
    """

    if D == 1:
        x_local = x[:,0]
    else:
        x_local = x[:,:D]

    # parameters
    block_size = 512  # make sure the FFT size is a power of 2
    hop = block_size // 2  # half overlap
    window = pra.hann(block_size)  # the analysis window
    if num_frames is not None:
        corr_num_samples = (num_frames-1)*hop + block_size
        x_local = x_local[:corr_num_samples,]

    # Create the STFT object
    if fixed_memory:
        if num_frames is not None:
            stft = pra.realtime.STFT(block_size, hop=hop, 
                analysis_window=window, channels=D, 
                transform=transform, num_frames=num_frames)
        else:
            stft = pra.realtime.STFT(block_size, hop=hop, 
                analysis_window=window, channels=D, 
                transform=transform, num_frames=0)
    else:
        stft = pra.realtime.STFT(block_size, hop=hop, 
            analysis_window=window, channels=D, 
            transform=transform)

    # collect the processed blocks
    processed_x = np.zeros(x_local.shape)

    if num_frames is not None:

        stft.analysis(x_local)
        processed_x = stft.synthesis()
        n = processed_x.shape[0]

        error = np.max(np.abs(x_local[block_size-hop:n-hop,] 
            - processed_x[block_size-hop:n-hop,]))

    else:

        n = 0
        # process the signals while full blocks are available
        while  x_local.shape[0] - n > hop:
            stft.analysis(x_local[n:n+hop,])
            processed_x[n:n+hop,] = stft.synthesis()
            n += hop

        error = np.max(np.abs(x_local[:n-hop,] - processed_x[hop:n,]))

    return error

def with_half_overlap_with_filter(D, num_frames=None, fixed_memory=False):
    """
    D             - number of channels
    num_frames    - how many frames to process, None will process one frame at 
                    a time 
    fixed_memory  - whether to enforce checks for size (real-time consideration)
    """

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
    if num_frames is not None:
        corr_num_samples = (num_frames-1)*hop + block_size
        x_local = x_local[:corr_num_samples,]

    # Create the STFT object
    if fixed_memory:
        if num_frames is not None:
            stft = pra.realtime.STFT(block_size, hop=hop, 
                analysis_window=window, channels=D, 
                transform=transform, num_frames=num_frames)
        else:
            stft = pra.realtime.STFT(block_size, hop=hop, 
                analysis_window=window, channels=D, 
                transform=transform, num_frames=0)
    else:
        stft = pra.realtime.STFT(block_size, hop=hop, 
            analysis_window=window, channels=D, 
            transform=transform)

    # setup the filter
    stft.set_filter(h_local, zb=h_len - 1)

    # collect the processed blocks
    processed_x = np.zeros(x_local.shape)

    if num_frames is not None:

        stft.analysis(x_local)
        stft.process()
        processed_x = stft.synthesis()
        n = processed_x.shape[0]

        error = np.max(np.abs(y_local[block_size:n-block_size,] 
            - processed_x[block_size:n-block_size,]))

    else:

        n = 0
        # process the signals while full blocks are available
        while  x_local.shape[0] - n > hop:
            stft.analysis(x_local[n:n+hop,])
            stft.process()  # apply the filter
            processed_x[n:n+hop,] = stft.synthesis()
            n += hop

        error = np.max(np.abs(y_local[:n-hop,] - processed_x[hop:n,]))

    return error


class TestSTFT(TestCase):

    # --- ONE FRAME, KEEP STATE, NOT FIXED MEMORY
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

    # --- ONE FRAME, KEEP STATE, FIXED MEMORY
    def test_incorrect_input_check_mono(self):
        result = incorrect_input_size(1, num_frames)
        self.assertTrue(result)

    def test_incorrect_input_check_stereo(self):
        result = incorrect_input_size(D, num_frames)
        self.assertTrue(result)

    def test_no_overlap_no_filter_mono_fixed(self):
        error = no_overlap_no_filter(1, num_frames)
        self.assertTrue(error < tol)

    def test_no_overlap_no_filter_multichannel_fixed(self):
        error = no_overlap_no_filter(D, num_frames)
        self.assertTrue(error < tol)

    def test_no_overlap_with_filter_mono(self):
        error = no_overlap_with_filter(1, num_frames)
        self.assertTrue(error < tol)

    def test_no_overlap_with_filter_multichannel_fixed(self):
        error = no_overlap_with_filter(D, num_frames)
        self.assertTrue(error < tol)

    def test_with_half_overlap_no_filter_mono_fixed(self):
        error = with_half_overlap_no_filter(1, num_frames)
        self.assertTrue(error < tol)

    def test_with_half_overlap_no_filter_multichannel_fixed(self):
        error = with_half_overlap_no_filter(D, num_frames)
        self.assertTrue(error < tol)

    def test_with_half_overlap_with_filter_mono_fixed(self):
        error = with_half_overlap_with_filter(1, num_frames)
        self.assertTrue(error < tol)

    def test_with_half_overlap_with_filter_multichannel_fixed(self):
        error = with_half_overlap_with_filter(D, num_frames)
        self.assertTrue(error < tol)

if __name__ == "__main__":

    print()

    print("---ONE FRAME, STREAMING, NOT FIXED MEMORY")
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
    print()

    print("---MULTIPLE FRAMES, STREAMING, NOT FIXED MEMORY")
    num_frames = 100
    error = no_overlap_no_filter(1, num_frames)
    print('no overlap, no filter, mono:', error)
    error = no_overlap_no_filter(D, num_frames)
    print('no overlap, no filter, multichannel:', error)
    error = no_overlap_with_filter(1, num_frames)
    print('no overlap, with filter, mono:', error)
    error = no_overlap_with_filter(D, num_frames)
    print('no overlap, with filter, multichannel:', error)
    error = with_half_overlap_no_filter(1, num_frames)
    print('with half overlap, no filter, mono:', error)
    error = with_half_overlap_no_filter(D, num_frames)
    print('with half overlap, no filter, multichannel:', error)
    error = with_half_overlap_with_filter(1, num_frames)
    print('with half overlap, with filter, mono:', error)
    error = with_half_overlap_with_filter(D, num_frames)
    print('with half overlap, with filter, multichannel:', error)
    print()

    print("---ONE FRAME, STREAMING, FIXED MEMORY")
    result = incorrect_input_size(1, 0)
    print('incorrect input size, mono:', result)
    result = incorrect_input_size(D, 0)
    print('incorrect input size, multichannel:', result)
    error = no_overlap_no_filter(1, None, True)
    print('no overlap, no filter, mono:', error)
    error = no_overlap_no_filter(D, None, True)
    print('no overlap, no filter, multichannel:', error)
    error = no_overlap_with_filter(1, None, True)
    print('no overlap, with filter, mono:', error)
    error = no_overlap_with_filter(D, None, True)
    print('no overlap, with filter, multichannel:', error)
    error = with_half_overlap_no_filter(1, None, True)
    print('with half overlap, no filter, mono:', error)
    error = with_half_overlap_no_filter(D, None, True)
    print('with half overlap, no filter, multichannel:', error)
    error = with_half_overlap_with_filter(1, None, True)
    print('with half overlap, with filter, mono:', error)
    error = with_half_overlap_with_filter(D, None, True)
    print('with half overlap, with filter, multichannel:', error)
    print()

    print("---MULTIPLE FRAME, STREAMING, FIXED MEMORY")
    num_frames = 100
    result = incorrect_input_size(1, num_frames)
    print('incorrect input size, mono:', result)
    result = incorrect_input_size(D, num_frames)
    print('incorrect input size, multichannel:', result)
    error = no_overlap_no_filter(1, num_frames, True)
    print('no overlap, no filter, mono:', error)
    error = no_overlap_no_filter(D, num_frames, True)
    print('no overlap, no filter, multichannel:', error)
    error = no_overlap_with_filter(1, num_frames, True)
    print('no overlap, with filter, mono:', error)
    error = no_overlap_with_filter(D, num_frames, True)
    print('no overlap, with filter, multichannel:', error)
    error = with_half_overlap_no_filter(1, num_frames, True)
    print('with half overlap, no filter, mono:', error)
    error = with_half_overlap_no_filter(D, num_frames, True)
    print('with half overlap, no filter, multichannel:', error)
    error = with_half_overlap_with_filter(1, num_frames, True)
    print('with half overlap, with filter, mono:', error)
    error = with_half_overlap_with_filter(D, num_frames, True)
    print('with half overlap, with filter, multichannel:', error)
    print()




