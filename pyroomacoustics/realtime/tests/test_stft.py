
from __future__ import division, print_function
from unittest import TestCase
import numpy as np
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from pyroomacoustics.realtime import STFT

'''
We create a signal, a simple filter and compute their convolution.

Then we test STFT block procesing with and without overlap,
and with and without filtering. Simulating a case of real-time
block processing.
'''

# test parameters
tol = 3e-6
np.random.seed(0)
D = 4
transform = 'numpy'   # 'numpy', 'pyfftw', or 'mkl'

# filter to apply
h_len = 99
h = np.ones((h_len, D))
h /= np.linalg.norm(h, axis=0)

# test signal (noise)
x = np.random.randn(100000, D).astype(np.float32)

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
    stft = STFT(block_size, hop=hop,
        channels=D, 
        transform=transform,
        num_frames=num_frames)

    try:  # passing more frames than 'hop'
        stft.analysis(x_local)
        computed = False
    except:
        computed = True

    return computed



def no_overlap_no_filter(D, num_frames=1, fixed_memory=False,
                        streaming=True):
    """
    D             - number of channels
    num_frames    - how many frames to process, None will process one frame at 
                    a time 
    fixed_memory  - whether to enforce checks for size (real-time consideration)
    streaming     - whether or not to stitch between frames
    """

    if D == 1:
        x_local = x[:,0]
    else:
        x_local = x[:,:D]

    # parameters
    block_size = 512  # make sure the FFT size is a power of 2
    hop = block_size  # no overlap
    if not streaming:
        num_samples = (num_frames-1)*hop+block_size
        x_local = x_local[:num_samples,]

    # Create the STFT object
    if fixed_memory:
        stft = STFT(block_size, hop=hop, channels=D, 
                transform=transform, num_frames=num_frames, streaming=streaming)
    else:
        stft = STFT(block_size, hop=hop, channels=D, 
            transform=transform, streaming=streaming)

    # collect the processed blocks
    processed_x = np.zeros(x_local.shape)

    if streaming:

        n = 0
        hop_frames = hop*num_frames
        # process the signals while full blocks are available
        while  x_local.shape[0] - n > hop_frames:
            stft.analysis(x_local[n:n+hop_frames,])
            processed_x[n:n+hop_frames,] = stft.synthesis()
            n += hop_frames

    else:

        stft.analysis(x_local)
        processed_x = stft.synthesis()
        n = processed_x.shape[0]

    error = np.max(np.abs(x_local[:n,] - processed_x[:n,]))

    return error

def no_overlap_with_filter(D, num_frames=1, fixed_memory=False,
                        streaming=True):
    """
    D             - number of channels
    num_frames    - how many frames to process, None will process one frame at 
                    a time 
    fixed_memory  - whether to enforce checks for size (real-time consideration)
    streaming     - whether or not to stitch between frames
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
    if not streaming:
        num_samples = (num_frames-1)*hop+block_size
        x_local = x_local[:num_samples,]

    # Create the STFT object
    if fixed_memory:
        stft = STFT(block_size, hop=hop, channels=D, 
                transform=transform, num_frames=num_frames, streaming=streaming)
    else:
        stft = STFT(block_size, hop=hop, channels=D, 
            transform=transform, streaming=streaming)
    
    # setup the filter
    stft.set_filter(h_local, zb=h_len - 1)

    # collect the processed blocks
    processed_x = np.zeros(x_local.shape)

    if not streaming:

        stft.analysis(x_local)
        stft.process()
        processed_x = stft.synthesis()
        n = processed_x.shape[0]

    else:

        n = 0
        hop_frames = hop*num_frames
        # process the signals while full blocks are available
        while  x_local.shape[0] - n > hop_frames:
            stft.analysis(x_local[n:n+hop_frames,])
            stft.process()  # apply the filter
            processed_x[n:n+hop_frames,] = stft.synthesis()
            n += hop_frames

    error = np.max(np.abs(y_local[:n,] - processed_x[:n,]))

    return error


def with_half_overlap_no_filter(D, num_frames=1, fixed_memory=False,
                        streaming=True):
    """
    D             - number of channels
    num_frames    - how many frames to process, None will process one frame at 
                    a time 
    fixed_memory  - whether to enforce checks for size (real-time consideration)
    streaming     - whether or not to stitch between frames
    """

    if D == 1:
        x_local = x[:,0]
    else:
        x_local = x[:,:D]

    # parameters
    block_size = 512  # make sure the FFT size is a power of 2
    hop = block_size // 2  # half overlap
    window = pra.hann(block_size)  # the analysis window
    if not streaming:
        num_samples = (num_frames-1)*hop+block_size
        x_local = x_local[:num_samples,]

    # Create the STFT object
    if fixed_memory:
        stft = STFT(block_size, hop=hop, channels=D, 
                transform=transform, num_frames=num_frames, 
                analysis_window=window, streaming=streaming)
    else:
        stft = STFT(block_size, hop=hop, channels=D, 
            transform=transform, analysis_window=window, streaming=streaming)

    # collect the processed blocks
    processed_x = np.zeros(x_local.shape)

    if not streaming:

        stft.analysis(x_local)
        processed_x = stft.synthesis()
        n = processed_x.shape[0]

        error = np.max(np.abs(x_local[block_size-hop:n-hop,] 
            - processed_x[block_size-hop:n-hop,]))

    else:

        n = 0
        hop_frames = hop*num_frames
        # process the signals while full blocks are available
        while  x_local.shape[0] - n > hop_frames:
            stft.analysis(x_local[n:n+hop_frames,])
            processed_x[n:n+hop_frames,] = stft.synthesis()
            n += hop_frames

        error = np.max(np.abs(x_local[:n-hop,] - processed_x[hop:n,]))

    return error

def with_half_overlap_with_filter(D, num_frames=1, fixed_memory=False,
                        streaming=True):
    """
    D             - number of channels
    num_frames    - how many frames to process, None will process one frame at 
                    a time 
    fixed_memory  - whether to enforce checks for size (real-time consideration)
    streaming     - whether or not to stitch between frames
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
    if not streaming:
        num_samples = (num_frames-1)*hop+block_size
        x_local = x_local[:num_samples,]

    # Create the STFT object
    if fixed_memory:
        stft = STFT(block_size, hop=hop, channels=D, 
                transform=transform, num_frames=num_frames, 
                analysis_window=window, streaming=streaming)
    else:
        stft = STFT(block_size, hop=hop, channels=D, 
            transform=transform, analysis_window=window, streaming=streaming)

    # setup the filter
    stft.set_filter(h_local, zb=h_len - 1)

    # collect the processed blocks
    processed_x = np.zeros(x_local.shape)

    if not streaming:

        stft.analysis(x_local)
        stft.process()
        processed_x = stft.synthesis()
        n = processed_x.shape[0]

        error = np.max(np.abs(y_local[block_size:n-block_size,] 
            - processed_x[block_size:n-block_size,]))

    else:

        n = 0
        hop_frames = hop*num_frames
        # process the signals while full blocks are available
        while  x_local.shape[0] - n > hop_frames:
            stft.analysis(x_local[n:n+hop_frames,])
            stft.process()  # apply the filter
            processed_x[n:n+hop_frames,] = stft.synthesis()
            n += hop_frames

        error = np.max(np.abs(y_local[:n-hop,] - processed_x[hop:n,]))

        # if D==1:
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     plt.plot(y_local)
        #     plt.plot(processed_x)
        #     plt.show()

    return error


def call_all_stft_tests(num_frames=1, fixed_memory=False, streaming=True, 
    overlap=True):

    error = no_overlap_no_filter(1, num_frames, fixed_memory,
        streaming)
    print('no overlap, no filter, mono             : %0.0f dB' 
        % (20*np.log10(error)))
    error = no_overlap_no_filter(D, num_frames, fixed_memory, streaming)
    print('no overlap, no filter, multichannel     : %0.0f dB' 
        % (20*np.log10(error)))
    error = no_overlap_with_filter(1, num_frames, fixed_memory,
        streaming)
    print('no overlap, with filter, mono           : %0.0f dB' 
        % (20*np.log10(error)))
    error = no_overlap_with_filter(D, num_frames, fixed_memory,
        streaming)
    print('no overlap, with filter, multichannel   : %0.0f dB' 
        % (20*np.log10(error)))
    if overlap:
        error = with_half_overlap_no_filter(1, num_frames, fixed_memory,
            streaming)
        print('half overlap, no filter, mono           : %0.0f dB' 
            % (20*np.log10(error)))
        error = with_half_overlap_no_filter(D, num_frames, fixed_memory,
            streaming)
        print('half overlap, no filter, multichannel   : %0.0f dB' 
            % (20*np.log10(error)))
        error = with_half_overlap_with_filter(1, num_frames, 
            fixed_memory, streaming)
        print('half overlap, with filter, mono         : %0.0f dB' 
            % (20*np.log10(error)))
        error = with_half_overlap_with_filter(D, num_frames, 
            fixed_memory, streaming)
        print('half overlap, with filter, multichannel : %0.0f dB' 
            % (20*np.log10(error)))
    print()



class TestSTFT(TestCase):

    def test_incorrect_input_check(self):
        result = incorrect_input_size(1, 100)
        self.assertTrue(result)
        result = incorrect_input_size(D, 100)
        self.assertTrue(result)

    def test_no_overlap_no_filter_mono(self):
        error = no_overlap_no_filter(D=1, num_frames=1, fixed_memory=False, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=1, num_frames=50, fixed_memory=False, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=1, num_frames=1, fixed_memory=True, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=1, num_frames=50, fixed_memory=True, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=1, num_frames=1, fixed_memory=False, 
            streaming=False)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=1, num_frames=50, fixed_memory=False, 
            streaming=False)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=1, num_frames=1, fixed_memory=True, 
            streaming=False)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=1, num_frames=50, fixed_memory=True, 
            streaming=False)
        self.assertTrue(error < tol)


    def test_no_overlap_no_filter_multichannel(self):
        error = no_overlap_no_filter(D=D, num_frames=1, fixed_memory=False, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=D, num_frames=50, fixed_memory=False, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=D, num_frames=1, fixed_memory=True, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=D, num_frames=50, fixed_memory=True, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=D, num_frames=1, fixed_memory=False, 
            streaming=False)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=D, num_frames=50, fixed_memory=False, 
            streaming=False)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=D, num_frames=1, fixed_memory=True, 
            streaming=False)
        self.assertTrue(error < tol)
        error = no_overlap_no_filter(D=D, num_frames=50, fixed_memory=True, 
            streaming=False)
        self.assertTrue(error < tol)

    def test_no_overlap_with_filter_mono(self):
        error = no_overlap_with_filter(D=1, num_frames=1, fixed_memory=False, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=1, num_frames=50, fixed_memory=False, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=1, num_frames=1, fixed_memory=True, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=1, num_frames=50, fixed_memory=True, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=1, num_frames=1, fixed_memory=False, 
            streaming=False)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=1, num_frames=50, fixed_memory=False, 
            streaming=False)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=1, num_frames=1, fixed_memory=True, 
            streaming=False)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=1, num_frames=50, fixed_memory=True, 
            streaming=False)
        self.assertTrue(error < tol)

    def test_no_overlap_with_filter_multichannel(self):
        error = no_overlap_with_filter(D=D, num_frames=1, fixed_memory=False, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=D, num_frames=50, fixed_memory=False, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=D, num_frames=1, fixed_memory=True, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=D, num_frames=50, fixed_memory=True, 
            streaming=True)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=D, num_frames=1, fixed_memory=False, 
            streaming=False)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=D, num_frames=50, fixed_memory=False, 
            streaming=False)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=D, num_frames=1, fixed_memory=True, 
            streaming=False)
        self.assertTrue(error < tol)
        error = no_overlap_with_filter(D=D, num_frames=50, fixed_memory=True, 
            streaming=False)
        self.assertTrue(error < tol)


    def test_with_half_overlap_no_filter_mono(self):
        error = with_half_overlap_no_filter(D=1, num_frames=1, 
            fixed_memory=False, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_no_filter(D=1, num_frames=50, 
            fixed_memory=False, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_no_filter(D=1, num_frames=1, 
            fixed_memory=True, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_no_filter(D=1, num_frames=50, 
            fixed_memory=True, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_no_filter(D=1, num_frames=50, 
            fixed_memory=False, streaming=False)
        self.assertTrue(error < tol)
        error = with_half_overlap_no_filter(D=1, num_frames=50, 
            fixed_memory=True, streaming=False)
        self.assertTrue(error < tol)


    def test_with_half_overlap_no_filter_multichannel(self):
        error = with_half_overlap_no_filter(D=D, num_frames=1, 
            fixed_memory=False, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_no_filter(D=D, num_frames=50, 
            fixed_memory=False, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_no_filter(D=D, num_frames=1, 
            fixed_memory=True, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_no_filter(D=D, num_frames=50, 
            fixed_memory=True, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_no_filter(D=D, num_frames=50, 
            fixed_memory=False, streaming=False)
        self.assertTrue(error < tol)
        error = with_half_overlap_no_filter(D=D, num_frames=50, 
            fixed_memory=True, streaming=False)
        self.assertTrue(error < tol)

    def test_with_half_overlap_with_filter_mono(self):
        error = with_half_overlap_with_filter(D=1, num_frames=1, 
            fixed_memory=False, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_with_filter(D=1, num_frames=50, 
            fixed_memory=False, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_with_filter(D=1, num_frames=1, 
            fixed_memory=True, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_with_filter(D=1, num_frames=50, 
            fixed_memory=True, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_with_filter(D=1, num_frames=50, 
            fixed_memory=False, streaming=False)
        self.assertTrue(error < tol)
        error = with_half_overlap_with_filter(D=1, num_frames=50, 
            fixed_memory=True, streaming=False)
        self.assertTrue(error < tol)

    def test_with_half_overlap_with_filter_multichannel(self):
        error = with_half_overlap_with_filter(D=D, num_frames=1, 
            fixed_memory=False, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_with_filter(D=D, num_frames=50, 
            fixed_memory=False, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_with_filter(D=D, num_frames=1, 
            fixed_memory=True, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_with_filter(D=D, num_frames=50, 
            fixed_memory=True, streaming=True)
        self.assertTrue(error < tol)
        error = with_half_overlap_with_filter(D=D, num_frames=50, 
            fixed_memory=False, streaming=False)
        self.assertTrue(error < tol)
        error = with_half_overlap_with_filter(D=D, num_frames=50, 
            fixed_memory=True, streaming=False)
        self.assertTrue(error < tol)


if __name__ == "__main__":

    print()
    print("TEST INFO")
    print("-------------------------------------------------------------")
    print("Max error in dB for randomnly generated signal of %d samples."
        % len(x))
    print("Multichannel corresponds to %d channels." % D)
    print("-------------------------------------------------------------")
    print()

    print("---ONE FRAME, STREAMING, NOT FIXED MEMORY")
    call_all_stft_tests(num_frames=1, fixed_memory=False, streaming=True)

    print("---MULTIPLE FRAMES, STREAMING, NOT FIXED MEMORY")
    call_all_stft_tests(num_frames=50, fixed_memory=False, streaming=True)

    print("---ONE FRAME, STREAMING, FIXED MEMORY")
    num_frames = 1
    result = incorrect_input_size(1, num_frames)
    print('incorrect input size, mono              :', result)
    result = incorrect_input_size(D, num_frames)
    print('incorrect input size, multichannel      :', result)
    call_all_stft_tests(num_frames=num_frames, fixed_memory=True, 
        streaming=True)

    print("---MULTIPLE FRAME, STREAMING, FIXED MEMORY")
    num_frames=50
    result = incorrect_input_size(1, num_frames)
    print('incorrect input size, mono              :', result)
    result = incorrect_input_size(D, num_frames)
    print('incorrect input size, multichannel      :', result)
    call_all_stft_tests(num_frames=num_frames, fixed_memory=True, 
        streaming=True)

    print("---ONE FRAME, NON-STREAMING, NOT FIXED MEMORY")
    call_all_stft_tests(num_frames=1, fixed_memory=False, streaming=False, 
        overlap=False)

    print("---MULTIPLE FRAMES, NON-STREAMING, NOT FIXED MEMORY")
    call_all_stft_tests(num_frames=50, fixed_memory=False, streaming=False)

    print("---ONE FRAME, NON-STREAMING, FIXED MEMORY")
    call_all_stft_tests(num_frames=1, fixed_memory=True, streaming=False, 
        overlap=False)

    print("---MULTIPLE FRAMES, NON-STREAMING, FIXED MEMORY")
    call_all_stft_tests(num_frames=50, fixed_memory=True, streaming=False)




