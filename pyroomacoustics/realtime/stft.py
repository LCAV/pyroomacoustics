# Author: Eric Bezzam
# Date: Feb 1, 2016

"""Class for real-time STFT analysis and processing."""
from __future__ import division

import numpy as np
from .dft import DFT


class STFT(object):
    """
    A class for STFT processing.

    Parameters
    -----------
    N : int
        number of samples per frame
    hop : int
        hop size (number of new samples)
    analysis_window : numpy array
        window applied to block before analysis
    synthesis_window : numpy array
        window applied to the block after synthesis
    channels : int
        number of signals
    transform: str, optional
        which FFT package to use: 'numpy', 'pyfftw', or 'mkl'
    """
    def __init__(self, N, hop=None, analysis_window=None, 
        synthesis_window=None, channels=1, transform='numpy'):

        # initialize parameters
        self.nsamples = N
        self.nchannels = channels
        if hop is not None:
            self.hop = hop  
        else:
            self.hop = N//2
        self.hop = int(np.floor(self.hop))

        # analysis window
        self.analysis_window = analysis_window
        if analysis_window is None and self.hop==N/2:
            # default to hanning (analysis) and rectangular (synthesis) window
            self.analysis_window = np.hanning(N)

        # synthesis window
        self.synthesis_window = synthesis_window

        # create DFT object
        self.transform = transform
        self.nfft = N
        self.nbin = self.nfft//2+1
        self.dft = DFT(nfft=self.nfft,
                       D=channels,
                       analysis_window=self.analysis_window,
                       synthesis_window=self.synthesis_window,
                       transform=self.transform)
        self.fft_out_buffer = np.zeros(self.nbin, dtype=np.complex64)

        # initialize filter + zero padding --> use set_filter
        self.zf = 0
        self.zb = 0
        self.H = None       # filter frequency spectrum

        # allocate all the required buffers
        self._make_buffers()


    def _make_buffers(self):

        self.n_state = self.nsamples - self.hop
        self.n_state_out = self.nfft - self.hop

        if self.nchannels==1:  # need this distinction for fftw

            # The input buffer, float32 for speed!
            self.fft_in_buffer = np.zeros(self.nfft, dtype=np.float32)
            #  a number of useful views on the input buffer
            self.fft_in_state = self.fft_in_buffer[self.zf:self.zf+self.n_state]  # Location of state
            self.fresh_samples = self.fft_in_buffer[self.zf+self.n_state:self.zf+self.n_state+self.hop]
            self.old_samples = self.fft_in_buffer[self.zf+self.hop:self.zf+self.hop+self.n_state]

            self.x_p = np.zeros(self.n_state, dtype=np.float32)  # State buffer
            self.y_p = np.zeros(self.n_state_out, dtype=np.float32)  # prev reconstructed samples
            self.X = np.zeros(self.nbin, dtype=np.complex64)       # current frame in STFT domain
            self.out = np.zeros(self.hop, dtype=np.float32)

        else:

            # The input buffer, float32 for speed!
            self.fft_in_buffer = np.zeros((self.nfft, self.nchannels), dtype=np.float32)
            #  a number of useful views on the input buffer
            self.fft_in_state = self.fft_in_buffer[self.zf:self.zf+self.n_state,:]  # Location of state
            self.fresh_samples = self.fft_in_buffer[self.zf+self.n_state:self.zf+self.n_state+self.hop,:]
            self.old_samples = self.fft_in_buffer[self.zf+self.hop:self.zf+self.hop+self.n_state,:]

            self.x_p = np.zeros((self.n_state, self.nchannels), dtype=np.float32)  # State buffer
            self.y_p = np.zeros((self.nfft - self.hop, self.nchannels), dtype=np.float32)  # prev reconstructed samples
            self.X = np.zeros((self.nbin, self.nchannels), dtype=np.complex64)       # current frame in STFT domain
            self.out = np.zeros((self.hop,self.nchannels), dtype=np.float32)


    def reset(self):
        """
        Reset state variables.
        """

        if self.nchannels==1:
            self.fft_in_buffer[:] = 0.
            self.x_p[:] = 0.
            self.y_p[:] = 0.
            self.X[:] = 0.
            self.out[:] = 0.
        else:
            self.fft_in_buffer[:,:] = 0.
            self.x_p[:,:] = 0.
            self.y_p[:,:] = 0.
            self.X[:,:] = 0.
            self.out[:,:] = 0.


    def zero_pad_front(self, zf):
        """
        Set zero-padding at beginning of frame.
        """
        self.zf = zf
        self.nfft = self.nsamples+self.zb+self.zf
        self.nbin = self.nfft//2+1
        if self.analysis_window is not None:
            self.analysis_window = np.concatenate((np.zeros(zf), self.analysis_window))
        if self.synthesis_window is not None:
            self.synthesis_window = np.concatenate((np.zeros(zf), self.synthesis_window))

        self.dft = DFT(nfft=self.nfft,
                       D=self.nchannels,
                       analysis_window=self.analysis_window,
                       synthesis_window=self.synthesis_window,
                       transform=self.transform)

        # We need to reallocate buffers after changing zero padding
        self._make_buffers()


    def zero_pad_back(self, zb):
        """
        Set zero-padding at end of frame.
        """
        self.zb = zb
        self.nfft = self.nsamples+self.zb+self.zf
        self.nbin = self.nfft//2+1
        if self.analysis_window is not None:
            self.analysis_window = np.concatenate((self.analysis_window, np.zeros(zb)))
        if self.synthesis_window is not None:
            self.synthesis_window = np.concatenate((self.synthesis_window, np.zeros(zb)))

        self.dft = DFT(nfft=self.nfft,
                       D=self.nchannels,
                       analysis_window=self.analysis_window,
                       synthesis_window=self.synthesis_window,
                       transform=self.transform)

        # We need to reallocate buffers after changing zero padding
        self._make_buffers()


    def set_filter(self, coeff, zb=None, zf=None, freq=False):
        """
        Set time-domain filter with appropriate zero-padding.
        Frequency spectrum of the filter is computed and set for the object. 
        There is also a check for sufficient zero-padding.

        Parameters
        -----------
        coeff : numpy array 
            Filter in time domain or frequency domain (if freq=True).
        zb : int
            Amount of zero-padding added to back/end of frame.
        zf : int
            Amount of zero-padding added to front/beginning of frame.
        freq : bool
            Whether or not given coefficients (coeff) are in the frequency domain.
        """

        # apply zero-padding
        if zb is not None:
            self.zero_pad_back(zb)
        if zf is not None:
            self.zero_pad_front(zf)       
        if not freq:
            # compute filter magnitude and phase spectrum
            self.H = np.complex64(np.fft.rfft(coeff, self.nfft, axis=0))
            # check for sufficient zero-padding
            if self.nfft < (self.nsamples+len(coeff)-1):
                raise ValueError('Insufficient zero-padding for chosen number of samples per frame (L) and filter length (h). Require zero-padding such that new length is at least (L+h-1).')
        else:
            if len(coeff)!=self.nbin:
                raise ValueError('Invalid length for frequency domain coefficients.')
            self.H = coeff


    def analysis(self, x_n):
        """
        Transform new samples to STFT domain for analysis.

        Parameters
        -----------
        x_n : numpy array
            [self.hop] new samples.
        """

        self.fresh_samples[:,] = x_n[:,]  # introduce new samples
        self.x_p[:,] = self.old_samples   # save next state

        # apply DFT to current frame
        self.X[:] = self.dft.analysis(self.fft_in_buffer)

        # shift back old samples to make way for new
        self.fft_in_state[:,] = self.x_p[:,]


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

        Parameters
        -----------
        X : numpy array (optional)
            Frequency domain vector of length <self.nbin> and type np.complex64

        Returns
        -------
        numpy array
            Reconstructed array of samples of length <self.hop> (Optional)
        """

        if X is not None:
            if X.shape != self.X.shape:
                raise ValueError('Input does not have the correct shape!')
            if X.dtype != self.X.dtype:
                raise ValueError('Input must be of type np.complex64!')
            self.X[:] = X

        # apply IDFT to current frame
        self.dft.synthesis(self.X)

        # reconstruct output
        self.out[:,] = self.dft.x[:self.hop,]  # fresh output samples

        # add state from previous frames when overlap is used
        if self.n_state_out > 0:
            m = np.minimum(self.hop, self.n_state_out)
            self.out[:m,] += self.y_p[:m,]
            # update state variables
            self.y_p[:-self.hop,] = self.y_p[self.hop:,]  # shift out left
            self.y_p[-self.hop:,] = 0.
            self.y_p[:,] += self.dft.x[-self.n_state_out:,]

        # if self.n_state_out > 0:
        #     if self.hop >= self.n_state_out:
        #         self.out[:self.n_state_out,] += self.y_p
        #         self.y_p[:,] = self.dft.x[-self.n_state_out:,]
        #     else:
        #         self.out[:,] += self.y_p[:self.hop,]
        #         self.y_p[:-self.hop,] = self.y_p[self.hop:,]  # shift unused part to left
        #         self.y_p[-self.hop:,] = 0.
        #         self.y_p[:,] += self.dft.x[-self.n_state_out:,]

        return self.out


    def get_prev_samples(self):
        """
        Get previous reconstructed samples.
        """
        return self.y_p
