# Author: Eric Bezzam
# Date: Feb 1, 2016

"""Class for real-time STFT analysis and processing."""
from __future__ import division

import numpy as np
from .dft import DFT


class STFT(object):
    """
    A class STFT processing.

    Parameters
    -----------
    N : int
        number of samples per frame
    hop : int
        hop size
    analysis_window : numpy array
        window applied to block before analysis
    synthesis : numpy array
        window applied to the block before synthesis
    channels : int
        number of signals
    transform: str, optional
        which FFT package to use: 'numpy', 'pyfftw', or 'mkl'
    """
    def __init__(self, N, hop=None, analysis_window=None, 
        synthesis_window=None, channels=1, transform='numpy'):
        # initialize parameters
        self.N = N          # number of samples per frame
        self.D = channels         # number of channels
        if hop is not None: # hop size
            self.hop = hop  
        else:
            self.hop = self.N/2
        self.hop = int(np.floor(self.hop))

        # analysis window
        if analysis_window is not None:
            self.analysis_window = analysis_window
        elif analysis_window is None and self.hop ==self.N/2:
            self.analysis_window = np.hanning(self.N)
        else:
            self.analysis_window = None
        # synthesis window
        if synthesis_window is not None:  
            self.synthesis_window = synthesis_window
        elif synthesis_window is None and self.hop ==self.N/2:
            self.synthesis_window = None # rectangular window
        else:
            self.synthesis_window = None

        # create DFT object
        self.transform = transform
        self.nfft = self.N   # differ when there is zero-padding
        self.nbin = self.nfft // 2 + 1
        self.dft = DFT(nfft=self.nfft,D=self.D,
                analysis_window=self.analysis_window,
                synthesis_window=self.synthesis_window,
                transform=self.transform)

        self.fft_out_buffer = np.zeros(self.nbin, dtype=np.complex64)

        # initialize filter + zero padding --> use set_filter
        self.zf = 0; self.zb = 0
        self.H = None       # filter frequency spectrum

        # state variables
        self.num_frames = 0            # number of frames processed so far
        self.n_state = self.N - self.hop

        # allocate all the required buffers
        self._make_buffers()

    def _make_buffers(self):

        if self.D==1:  # need this distinction for fftw

            # The input buffer, float32 for speed!
            self.fft_in_buffer = np.zeros(self.nfft, dtype=np.float32)
            #  a number of useful views on the input buffer
            self.fft_in_state = self.fft_in_buffer[self.zf:self.zf+self.n_state]  # Location of state
            self.fresh_samples = self.fft_in_buffer[self.zf+self.n_state:self.zf+self.n_state+self.hop]
            self.old_samples = self.fft_in_buffer[self.zf+self.hop:self.zf+self.hop+self.n_state]

            self.x_p = np.zeros(self.n_state, dtype=np.float32)  # State buffer
            self.y_p = np.zeros(self.nfft - self.hop, dtype=np.float32)  # prev reconstructed samples
            self.X = np.zeros(self.nbin, dtype=np.complex64)       # current frame in STFT domain
            self.out = np.zeros(self.hop, dtype=np.float32)

        else:

            # The input buffer, float32 for speed!
            self.fft_in_buffer = np.zeros((self.nfft, self.D), dtype=np.float32)
            #  a number of useful views on the input buffer
            self.fft_in_state = self.fft_in_buffer[self.zf:self.zf+self.n_state,:]  # Location of state
            self.fresh_samples = self.fft_in_buffer[self.zf+self.n_state:self.zf+self.n_state+self.hop,:]
            self.old_samples = self.fft_in_buffer[self.zf+self.hop:self.zf+self.hop+self.n_state,:]

            self.x_p = np.zeros((self.n_state, self.D), dtype=np.float32)  # State buffer
            self.y_p = np.zeros((self.nfft - self.hop, self.D), dtype=np.float32)  # prev reconstructed samples
            self.X = np.zeros((self.nbin, self.D), dtype=np.complex64)       # current frame in STFT domain
            self.out = np.zeros((self.hop,self.D), dtype=np.float32)


    def reset(self):
        """
        Reset state variables. Necesary after changing or setting the filter or zero padding.
        """
        self.num_frames = 0
        self.nbin = self.nfft // 2 + 1

        if self.D==1:
            self.fft_in_buffer[:] = 0.
            self.X[:] = 0.
            self.y_p[:] = 0.
        else:
            self.fft_in_buffer[:,:] = 0.
            self.X[:,:] = 0.
            self.y_p[:,:] = 0.

        self.dft = DFT(nfft=self.nfft,D=self.D,
            analysis_window=self.analysis_window,
            synthesis_window=self.synthesis_window,
            transform=self.transform)

    def zero_pad_front(self, zf):
        """
        Set zero-padding at beginning of frame.
        """
        self.zf = zf
        self.nfft = self.N+self.zb+self.zf
        if self.analysis_window is not None:
            self.analysis_window = np.concatenate((np.zeros(zf), self.analysis_window))
        if self.synthesis_window is not None:
            self.synthesis_window = np.concatenate((np.zeros(zf), self.synthesis_window))

    def zero_pad_back(self, zb):
        """
        Set zero-padding at end of frame.
        """
        self.zb = zb
        self.nfft = self.N+self.zb+self.zf
        if self.analysis_window is not None:
            self.analysis_window = np.concatenate((self.analysis_window, np.zeros(zb)))
        if self.synthesis_window is not None:
            self.synthesis_window = np.concatenate((self.synthesis_window, np.zeros(zb)))

    def set_filter(self, coeff, zb=None, zf=None, freq=False):
        """
        Set time-domain filter with appropriate zero-padding.
        Frequency spectrum of the filter is computed and set for the object. 
        There is also a check for sufficient zero-padding.

        Parameters
        -----------
        coeff : numpy array 
            Filter in time domain.
        zb : int
            Amount of zero-padding added to back/end of frame.
        zf : int
            Amount of zero-padding added to front/beginning of frame.
        """
        # apply zero-padding
        if zb is not None:
            self.zero_pad_back(zb)
        if zf is not None:
            self.zero_pad_front(zf)  
        self.reset()      
        if not freq:
            # compute filter magnitude and phase spectrum
            self.H = np.complex64(np.fft.rfft(coeff, self.nfft, axis=0))
            # check for sufficient zero-padding
            if self.nfft < (self.N+len(coeff)-1):
                raise ValueError('Insufficient zero-padding for chosen number of samples per frame (L) and filter length (h). Require zero-padding such that new length is at least (L+h-1).')
        else:
            if len(coeff)!=self.nbin:
                raise ValueError('Invalid length for frequency domain coefficients.')
            self.H = coeff

        # We need to reallocate buffers after changing zero padding
        self._make_buffers()


    def analysis(self, x_n):
        """
        Transform new samples to STFT domain for analysis.
        Parameters
        -----------
        x_n : numpy array
            [self.hop] new samples.
        Returns
        -----------
        self.X : numpy array 
            Frequency spectrum of given frame.
        """

        # check for valid input - already done by self.dft
        # if x_n.shape[0]!=self.hop:
        #     raise ValueError('Invalid input dimensions.')
        # if self.D > 1 and x_n.shape[1]!=self.D:
        #     raise ValueError('Invalid input dimensions.')

        self.fresh_samples[:,] = x_n[:,]  # introduce new samples

        self.x_p[:,] = self.old_samples   # save next state

        # apply DFT to current frame
        self.X[:] = self.dft.analysis(self.fft_in_buffer)

        # shift backwards in the buffer the state
        self.fft_in_state[:,] = self.x_p[:,]

        # self.num_frames += 1

    def process(self):
        """
        Apply filtering in STFT domain.
        Returns
        -----------
        self.X : numpy array 
            Frequency spectrum of given frame.
        """
        if self.H is not None:
            np.multiply(self.X, self.H, self.X)


    def synthesis(self, X=None):
        """
        Transform to time domain and reconstruct output with overlap-and-add.

        Returns
        -------
        numpy array
            Reconstructed array of samples of length <self.hop> (Optional)
        """

        if X is not None:
            self.X[:] = X

        # apply IDFT to current frame
        self.dft.synthesis(self.X)

        # reconstruct output
        L = self.y_p.shape[0]  # length of output state vector

        self.out[:,] = self.dft.x[0:self.hop,]  # fresh output samples

        # add state from previous frames when overlap is used
        if L > 0:
            m = np.minimum(self.hop, L)
            self.out[:m,] += self.y_p[:m,]
            # update state variables
            self.y_p[:-self.hop,] = self.y_p[self.hop:,]  # shift out left
            self.y_p[-self.hop:,] = 0.
            self.y_p[:,] += self.dft.x[-L:,]

        return self.out


    def get_prev_samples(self):
        """
        Get reconstructed previous samples.
        """
        return self.y_p
