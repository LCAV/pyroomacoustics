# Author: Eric Bezzam
# Date: Feb 1, 2016

"""Class for real-time STFT analysis and processing."""
from __future__ import division

import numpy as np
from numpy.lib.stride_tricks import as_strided as _as_strided
import warnings
from .dft import DFT


class STFT(object):
    """
    A class for STFT processing.

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

    num_frames (optional) : int
        Number of frames to be processed. If set, this will be strictly enforced
        as the STFT block will allocate memory accordingly. If not set, there
        will be no check on the number of frames sent to 
        analysis/process/synthesis
    """
    def __init__(self, N, hop=None, analysis_window=None, 
        synthesis_window=None, channels=1, transform='numpy',
        **kwargs):

        # initialize parameters
        self.num_samples = N            # number of samples per frame
        self.num_channels = channels    # number of channels
        if hop is not None:             # hop size --> number of input samples
            self.hop = hop  
        else:
            self.hop = self.N/2
        self.hop = int(np.floor(self.hop))

        # analysis and synthesis window
        self.analysis_window = analysis_window
        self.synthesis_window = synthesis_window

        if analysis_window is None and synthesis_window is None and self.hop==N/2:
            # default to hanning (analysis) and rectangular (synthesis) window
            self.analysis_window = np.hanning(self.num_samples)

        # create DFT object
        self.transform = transform
        self.nfft = self.num_samples   # differ when there is zero-padding
        self.nbin = self.nfft // 2 + 1
        self.dft = DFT(nfft=self.nfft,
                       D=self.num_channels,
                       analysis_window=self.analysis_window,
                       synthesis_window=self.synthesis_window,
                       transform=self.transform)

        self.fft_out_buffer = np.zeros(self.nbin, dtype=np.complex64)

        # initialize filter + zero padding --> use set_filter
        self.zf = 0
        self.zb = 0
        self.H = None           # filter frequency spectrum
        self.H_multi = None     # for multiple frames

        # check keywords
        if 'num_frames' in kwargs.keys():
            self.fixed_input = True
            num_frames = kwargs['num_frames']
            if num_frames <= 0:
                raise ValueError('num_frames must be positive!')
            self.num_frames = num_frames
        else:
            self.fixed_input = False
            self.num_frames = 1

        # allocate all the required buffers
        self._make_buffers()



    def _make_buffers(self):
        """
        Allocate memory for internal buffers according to FFT size, number of
        channels, and number of frames.
        """

        # state variables
        self.n_state = self.num_samples - self.hop
        self.n_state_out = self.nfft - self.hop

        # need this distinction for mono and multi for FFTW package
        if self.num_channels==1:  

            # The input buffer, float32 for speed!
            self.fft_in_buffer = np.zeros(self.nfft, dtype=np.float32)
            #  a number of useful views on the input buffer
            self.fft_in_state = self.fft_in_buffer[self.zf:self.zf+self.n_state]  # Location of state
            self.fresh_samples = self.fft_in_buffer[self.zf+self.n_state:self.zf+self.n_state+self.hop]
            self.old_samples = self.fft_in_buffer[self.zf+self.hop:self.zf+self.hop+self.n_state]

            self.x_p = np.zeros(self.n_state, dtype=np.float32)  # State buffer
            self.y_p = np.zeros(self.n_state_out, dtype=np.float32)  # prev reconstructed samples
            self.out = np.zeros(self.hop, dtype=np.float32)

            if self.fixed_input:
                if self.num_frames==1:
                    self.X = np.zeros((self.nbin), dtype=np.complex64) 
                else:
                    self.X = np.zeros((self.nbin,self.num_frames), dtype=np.complex64)
                    self.dft_multi = DFT(nfft=self.nfft,D=self.num_frames,
                                        analysis_window=self.analysis_window,
                                        synthesis_window=self.synthesis_window,
                                        transform=self.transform)

            else:
                self.X = None
                self.dft_multi = None
            

        else:

            # The input buffer, float32 for speed!
            self.fft_in_buffer = np.zeros((self.nfft, self.num_channels), dtype=np.float32)
            #  a number of useful views on the input buffer
            self.fft_in_state = self.fft_in_buffer[self.zf:self.zf+self.n_state,:]  # Location of state
            self.fresh_samples = self.fft_in_buffer[self.zf+self.n_state:self.zf+self.n_state+self.hop,:]
            self.old_samples = self.fft_in_buffer[self.zf+self.hop:self.zf+self.hop+self.n_state,:]

            self.x_p = np.zeros((self.n_state, self.num_channels), dtype=np.float32)  # State buffer
            self.y_p = np.zeros((self.n_state_out, self.num_channels), dtype=np.float32)  # prev reconstructed samples
            self.out = np.zeros((self.hop,self.num_channels), dtype=np.float32)


            if self.fixed_input:
                if self.num_frames==1:
                    self.X = np.zeros((self.nbin,self.num_channels), dtype=np.complex64)
                else:
                    self.X = np.zeros((self.nbin,self.num_frames,self.num_channels), 
                        dtype=np.complex64)
                    self.dft_multi = DFT(nfft=self.nfft,D=self.num_frames,
                                        analysis_window=self.analysis_window,
                                        synthesis_window=self.synthesis_window,
                                        transform=self.transform)
            else:
                self.X = None
                self.dft_multi = None


    def reset(self):
        """
        Reset state variables. Necesary after changing or setting the filter or zero padding.
        """

        if self.num_channels==1:
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
        self.nfft = self.N+self.zb+self.zf
        self.nbin = self.nfft//2+1
        if self.analysis_window is not None:
            self.analysis_window = np.concatenate((np.zeros(zf), self.analysis_window))
        if self.synthesis_window is not None:
            self.synthesis_window = np.concatenate((np.zeros(zf), self.synthesis_window))

        self.dft = DFT(nfft=self.nfft,
                       D=self.num_channels,
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
        self.nfft = self.num_samples+self.zb+self.zf
        self.nbin = self.nfft//2+1
        if self.analysis_window is not None:
            self.analysis_window = np.concatenate((self.analysis_window, np.zeros(zb)))
        if self.synthesis_window is not None:
            self.synthesis_window = np.concatenate((self.synthesis_window, np.zeros(zb)))

        self.dft = DFT(nfft=self.nfft,
                       D=self.num_channels,
                       analysis_window=self.analysis_window,
                       synthesis_window=self.synthesis_window,
                       transform=self.transform)

        # We need to reallocate buffers after changing zero padding
        self._make_buffers()


    def set_filter(self, coeff, zb=None, zf=None, freq=False):
        """
        Set time-domain FIR filter with appropriate zero-padding.
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
            if self.nfft < (self.num_samples+len(coeff)-1):
                raise ValueError('Insufficient zero-padding for chosen number of samples per frame (L) and filter length (h). Require zero-padding such that new length is at least (L+h-1).')
        else:
            if len(coeff)!=self.nbin:
                raise ValueError('Invalid length for frequency domain coefficients.')
            self.H = coeff

        # prepare filter if fixed input case
        if self.fixed_input:
            if self.num_channels == 1:
                self.H_multi = np.tile(self.H,(self.num_frames,1)).T
            else:
                self.H_multi = np.tile(self.H,(self.num_frames,1,1))
                self.H_multi = np.swapaxes(self.H_multi,0,1)




    def analysis(self, x):

        """
        Parameters
        -----------
        x  : numpy array
            Time-domain signal. If 'fixed_input' is set to True during object
            construction, the number of samples in each channel of x must be 
            [hop*num_frames]. Otherwise, num_frames will be set as the number of
            samples in x (num_samples) divided by 'hop': num_samples//hop (for 
            an integer value).

        """

        # ----check correct number of channels
        x_shape = x.shape
        if self.num_channels > 1:
            if len(x_shape) < 1:   # received mono
                raise ValueError("Incorrect number of channels.")
            if x_shape[1] != self.num_channels:
                raise ValueError("Incorrect number of channels.")
        else:
            if len(x.shape) > 1:    # received multi-channel, expecting mono
                raise ValueError("Incorrect number of channels.")

        # ----check number of frames
        if self.fixed_input:
            if x_shape[0]!=self.hop*self.num_frames:
                raise ValueError('Input must be of length %d; received %d samples.' %
                    (self.hop*self.num_frames, x_shape[0]))
            num_frames = self.num_frames

        else:
            num_frames = x_shape[0]//self.hop
            self.num_frames = num_frames
            x_spill = x[num_frames*self.hop:]

            # re-allocate memory for self.X
            if num_frames > 1:
                if self.num_channels == 1:
                    self.X = np.zeros((self.nbin,num_frames), 
                        dtype=np.complex64)
                else:
                    self.X = np.zeros((self.nbin,num_frames,self.num_channels), 
                        dtype=np.complex64)
                self.dft_multi = DFT(nfft=self.nfft,D=self.num_frames,
                    analysis_window=self.analysis_window,
                    synthesis_window=self.synthesis_window,
                    transform=self.transform)
            elif num_frames==1:
                if self.num_channels == 1:
                    self.X = np.zeros((self.nbin), dtype=np.complex64) 
                else:
                    self.X = np.zeros((self.nbin,self.num_channels), dtype=np.complex64)

        # use appropriate function
        if num_frames > 1:
            self._analysis_multiple(x)
        elif num_frames==1:
            self._analysis_single(x)
        else:
            warnings.warn("Not enough samples for 'analysis'. Received %d samples, need at least %d." % (x.shape[0], self.hop))


    def _analysis_single(self, x_n):
        """
        Transform new samples to STFT domain for analysis.

        Parameters
        -----------
        x_n : numpy array
            [self.hop] new samples
        """

        # correct input size check in: dft.analysis()

        self.fresh_samples[:,] = x_n[:,]  # introduce new samples
        self.x_p[:,] = self.old_samples   # save next state

        # apply DFT to current frame
        self.X[:] = self.dft.analysis(self.fft_in_buffer)

        # shift backwards in the buffer the state
        self.fft_in_state[:,] = self.x_p[:,]


    def _analysis_multiple(self, x):
        """
        Apply STFT analysis to multiple frames.

        Parameters
        -----------
        x : numpy array
            New samples.
        Returns
        -----------
        mX : numpy array
            Multple frames in the STFT domain.

        """ 

        ## ----- STRIDED WAY
        new_strides = (self.hop * x.strides[0], x.strides[0])
        # new_shape = (self.num_samples, self.num_frames)  # this way leads to segmentation fault
        new_shape = (self.num_frames, self.num_samples)

        if self.num_channels > 1:
            for c in range(self.num_channels):
                y = _as_strided(x[:,c], shape=new_shape, strides=new_strides)
                # y = np.concatenate((np.zeros((self.zf,y.shape[1])), y, 
                #     np.zeros((self.zb,y.shape[1]))), axis=0)
                y = np.concatenate((np.zeros((y.shape[0],self.zf)), y, 
                    np.zeros((y.shape[0],self.zb))), axis=1)

                self.X[:,:,c] = self.dft_multi.analysis(y.T)
        else:
            y = _as_strided(x, shape=new_shape, strides=new_strides)
            # y = np.concatenate((np.zeros((self.zf,y.shape[1])), y, 
            #         np.zeros((self.zb,y.shape[1]))), axis=0)
            y = np.concatenate((np.zeros((y.shape[0],self.zf)), y, 
                    np.zeros((y.shape[0],self.zb))), axis=1)

            self.X[:] = self.dft_multi.analysis(y.T)


        ## ----- "BRUTE" FORCE WAY
        # if self.num_channels > 1:

        #     for c in range(self.num_channels):

        #         if self.num_samples == self.hop:
        #             y = np.reshape(x[:self.num_frames*self.hop,c], (self.num_frames, self.hop))
        #         else:
        #             n = 0
        #             y = np.zeros((self.num_frames,self.num_samples),dtype=np.float32)
        #             for f in range(self.num_frames-1):
        #                 y[f:,] = x[n:n+self.num_samples,c]
        #                 n += self.hop

        #         y = np.concatenate((np.zeros((y.shape[0],self.zf)), y, 
        #             np.zeros((y.shape[0],self.zb))), axis=1)

        #         self.X[:,:,c] = dft.analysis(y.T).T

        # else:

        #     if self.num_samples == self.hop:
        #         y = np.reshape(x[:self.num_frames*self.hop], (self.num_frames, self.hop))
        #     else:
        #         n = 0
        #         y = np.zeros((self.num_frames,self.num_samples),dtype=np.float32)
        #         for f in range(self.num_frames-1):
        #             y[f:,] = x[n:n+self.num_samples]
        #             n += self.hop

        #     y = np.concatenate((np.zeros((y.shape[0],self.zf)), y, 
        #         np.zeros((y.shape[0],self.zb))), axis=1)

        #     self.X[:] = dft.analysis(y.T).T



    def process(self, X=None):

        """
        Parameters
        -----------
        X  : numpy array
            X can take on multiple shapes:
            1) (N,) if it is single channel and only one frame
            2) (N,D) if it is multi-channel and only one frame
            3) (F,N) if it is single channel but multiple frames
            4) (F,N,D) if it is multi-channel and multiple frames

        Returns
        -----------
        x_r : numpy array
            Reconstructed time-domain signal.

        """

        # check that there is filter
        if self.H is None:
            warnings.warn("No filter is set! Exiting...")
            return

        # check number of frames and correct number of bins
        if X is not None:
            X_shape = X.shape
            if len(X_shape)==1:  # single channel, one frame
                num_frames = 1
            elif len(X_shape)==2 and self.num_channels>1: # multi-channel, one frame
                num_frames = 1
            elif len(X_shape)==2 and self.num_channels==1: # single channel, multiple frames
                num_frames = X_shape[1]
            elif len(X_shape)==3 and self.num_channels>1: # multi-channel, multiple frames
                num_frames = X_shape[1]
            else:
                raise ValueError("Invalid input shape.")

            # check number of bins
            if X_shape[0]!=self.nbin:
                raise ValueError('Invalid number of frequency bins! Expecting %d, got %d'
                    % (self.nbin,X_shape[0]))

            # check number of frames, if fixed input size
            if self.fixed_input:
                if num_frames != self.num_frames:
                    raise ValueError('Input must have %d frames!', 
                        self.num_frames)
                self.X[:] = X
            else:
                self.X = X
                self.num_frames = num_frames


        else:
            num_frames = self.num_frames


        # use appropriate function
        if num_frames > 1:
            self._process_multiple()
        elif num_frames==1:
            self._process_single()


    def _process_single(self):

        np.multiply(self.X, self.H, self.X)


    def _process_multiple(self):

        if not self.fixed_input:
            if self.num_channels == 1:
                self.H_multi = np.tile(self.H,(self.num_frames,1)).T
            else:

                self.H_multi = np.tile(self.H,(self.num_frames,1,1))
                self.H_multi = np.swapaxes(self.H_multi,0,1)

        np.multiply(self.X, self.H_multi, self.X)


    def synthesis(self, X=None):

        """
        Parameters
        -----------
        X  : numpy array
            X can take on multiple shapes:
            1) (N,) if it is single channel and only one frame
            2) (N,D) if it is multi-channel and only one frame
            3) (F,N) if it is single channel but multiple frames
            4) (F,N,D) if it is multi-channel and multiple frames

        Returns
        -----------
        x_r : numpy array
            Reconstructed time-domain signal.

        """

        # check number of frames
        if X is not None:
            X_shape = X.shape
            if len(X_shape)==1:  # single channel, one frame
                num_frames = 1
            elif len(X_shape)==2 and self.num_channels>1: # multi-channel, one frame
                num_frames = 1
            elif len(X_shape)==2 and self.num_channels==1: # single channel, multiple frames
                num_frames = X_shape[1]
            elif len(X_shape)==3 and self.num_channels>1: # multi-channel, multiple frames
                num_frames = X_shape[1]
            else:
                raise ValueError("Invalid input shape.")

            if self.fixed_input:
                if num_frames != self.num_frames:
                    raise ValueError('Input must have %d frames!', self.num_frames)
            else:
                self.num_frames = num_frames
        else:
            num_frames = self.num_frames

        # use appropriate function
        if num_frames > 1:
            return self._synthesis_multiple(X)
        elif num_frames==1:
            return self._synthesis_single(X)



    def _synthesis_single(self, X=None):
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

        return self._overlap_and_add()



    def _synthesis_multiple(self, X=None):
        """
        Apply STFT analysis to multiple frames.

        Parameters
        -----------
        mX : numpy array
            Multiple STFT domain frames. Array of shape FxBxC where
            F is the number of frames, B is the number of frequency bins,
            and C is the number of channels.
        Returns
        -----------
        x_r : numpy array
            Recovered signal.

        """

        if X is not None:
            self.X[:] = X

        # synthesis + overlap and add
        if self.num_channels > 1:

            x_r = np.zeros((self.num_frames*self.hop,self.num_channels), dtype=np.float32)

            n = 0
            for f in range(self.num_frames):

                # appy IDFT to current frame and reconstruct output
                x_r[n:n+self.hop,] = self._overlap_and_add(self.dft.synthesis(self.X[:,f,:]))
                n += self.hop

        else:

            x_r = np.zeros(self.num_frames*self.hop, dtype=np.float32)

            # treat number of frames as the multiple channels for DFT
            if not self.fixed_input:
                self.dft_multi = DFT(nfft=self.nfft,D=self.num_frames,
                    analysis_window=self.analysis_window,
                    synthesis_window=self.synthesis_window,
                    transform=self.transform)

            # back to time domain
            mx = self.dft_multi.synthesis(self.X)

            # overlap and add
            n = 0
            for f in range(self.num_frames):
                x_r[n:n+self.hop,] = self._overlap_and_add(mx[:,f])
                n += self.hop

        return x_r


    def _overlap_and_add(self, x=None):

        if x is None:
            x = self.dft.x

        self.out[:,] = x[0:self.hop,]  # fresh output samples

        # add state from previous frames when overlap is used
        if self.n_state_out > 0:
            m = np.minimum(self.hop, self.n_state_out)
            self.out[:m,] += self.y_p[:m,]
            # update state variables
            self.y_p[:-self.hop,] = self.y_p[self.hop:,]  # shift out left
            self.y_p[-self.hop:,] = 0.
            self.y_p[:,] += x[-self.n_state_out:,]

        return self.out


    def get_prev_samples(self):
        """
        Get reconstructed previous samples.
        """
        return self.y_p
