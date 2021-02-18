# Short Time Fourier Transform
# Copyright (C) 2019  Eric Bezzam, Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# You should have received a copy of the MIT License along with this program. If
# not, see <https://opensource.org/licenses/MIT>.

"""Class for real-time STFT analysis and processing."""
from __future__ import division

import numpy as np
from numpy.lib.stride_tricks import as_strided as _as_strided
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
    synthesis_window : numpy array
        window applied to the block before synthesis
    channels : int
        number of signals

    transform : str, optional
        which FFT package to use: 'numpy' (default), 'pyfftw', or 'mkl'
    streaming : bool, optional
        whether (True, default) or not (False) to "stitch" samples between
        repeated calls of 'analysis' and 'synthesis' if we are receiving a 
        continuous stream of samples.
    num_frames : int, optional
        Number of frames to be processed. If set, this will be strictly enforced
        as the STFT block will allocate memory accordingly. If not set, there
        will be no check on the number of frames sent to 
        analysis/process/synthesis

        NOTE:
            1) num_frames = 0, corresponds to a "real-time" case in which each
            input block corresponds to [hop] samples.
            2) num_frames > 0, requires [(num_frames-1)*hop + N] samples as the
            last frame must contain [N] samples.

    precision : string, np.float32, np.float64, np.complex64, np.complex128, optional
        How many precision bits to use for the input.
        If 'single'/np.float32/np.complex64, 32 bits for real inputs or 64 for complex spectrum.
        Otherwise, cast to 64 bits for real inputs or 128 for complex spectrum (default).

    """

    def __init__(self, N, hop=None, analysis_window=None,
                 synthesis_window=None, channels=1, transform='numpy',
                 streaming=True, precision='double', **kwargs):

        # initialize parameters
        self.num_samples = N            # number of samples per frame
        self.num_channels = channels    # number of channels
        self.mono = True if self.num_channels == 1 else False
        if hop is not None:             # hop size --> number of input samples
            self.hop = hop  
        else:
            self.hop = self.num_samples

        if precision == np.float32 or precision == np.complex64 or precision == 'single':
            self.time_dtype = np.float32
            self.freq_dtype = np.complex64
        else:
            self.time_dtype = np.float64
            self.freq_dtype = np.complex128

        # analysis and synthesis window
        self.analysis_window = analysis_window
        self.synthesis_window = synthesis_window

        # prepare variables for DFT object
        self.transform = transform
        self.nfft = self.num_samples   # differ when there is zero-padding
        self.nbin = self.nfft // 2 + 1

        # initialize filter + zero padding --> use set_filter
        self.zf = 0
        self.zb = 0
        self.H = None           # filter frequency spectrum
        self.H_multi = None     # for multiple frames

        # check keywords
        if 'num_frames' in kwargs.keys():
            self.fixed_input = True
            num_frames = kwargs['num_frames']
            if num_frames < 0:
                raise ValueError('num_frames must be non-negative!')
            self.num_frames = num_frames
        else:
            self.fixed_input = False
            self.num_frames = 0

        # allocate all the required buffers
        self.streaming = streaming
        self._make_buffers()

    def _make_buffers(self):
        """
        Allocate memory for internal buffers according to FFT size, number of
        channels, and number of frames.
        """

        # state variables
        self.n_state = self.num_samples - self.hop
        self.n_state_out = self.nfft - self.hop

        # make DFT object
        self.dft = DFT(nfft=self.nfft,
                       D=self.num_channels,
                       analysis_window=self.analysis_window,
                       synthesis_window=self.synthesis_window,
                       transform=self.transform)
        """
        1D array for num_channels=1 as the FFTW package can only take 1D array 
        for 1D DFT.
        """
        if self.mono:
            # input buffer
            self.fft_in_buffer = np.zeros(self.nfft, dtype=self.time_dtype)
            # state buffer
            self.x_p = np.zeros(self.n_state, dtype=self.time_dtype)
            # prev reconstructed samples
            self.y_p = np.zeros(self.n_state_out, dtype=self.time_dtype)
            # output samples
            self.out = np.zeros(self.hop, dtype=self.time_dtype)
        else:
            # input buffer
            self.fft_in_buffer = np.zeros((self.nfft, self.num_channels),
                                          dtype=self.time_dtype)
            # state buffer
            self.x_p = np.zeros((self.n_state, self.num_channels),
                                dtype=self.time_dtype)
            # prev reconstructed samples
            self.y_p = np.zeros((self.n_state_out, self.num_channels),
                                dtype=self.time_dtype)
            # output samples
            self.out = np.zeros((self.hop, self.num_channels),
                                dtype=self.time_dtype)

        # useful views on the input buffer
        self.fft_in_state = self.fft_in_buffer[self.zf:self.zf + self.n_state, ]
        self.fresh_samples = self.fft_in_buffer[self.zf + self.n_state:
                                                self.zf + self.n_state +
                                                self.hop, ]
        self.old_samples = self.fft_in_buffer[self.zf + self.hop:
                                              self.zf + self.hop +
                                              self.n_state, ]

        # if fixed number of frames to process
        if self.fixed_input:
            if self.num_frames == 0:
                if self.mono:
                    self.X = np.zeros(self.nbin, dtype=self.freq_dtype)
                else:
                    self.X = np.zeros((self.nbin, self.num_channels),
                                      dtype=self.freq_dtype)
            else:
                self.X = np.squeeze(np.zeros((self.num_frames,
                                              self.nbin,
                                              self.num_channels),
                                             dtype=self.freq_dtype))
                # DFT object for multiple frames
                self.dft_frames = DFT(nfft=self.nfft, D=self.num_frames,
                                      analysis_window=self.analysis_window,
                                      synthesis_window=self.synthesis_window,
                                      transform=self.transform)
        else: # we will allocate these on-the-fly
            self.X = None
            self.dft_frames = None

    def reset(self):
        """
        Reset state variables. Necessary after changing or setting the filter
        or zero padding.
        """

        if self.mono:
            self.fft_in_buffer[:] = 0.
            self.x_p[:] = 0.
            self.y_p[:] = 0.
            self.X[:] = 0.
            self.out[:] = 0.
        else:
            self.fft_in_buffer[:, :] = 0.
            self.x_p[:, :] = 0.
            self.y_p[:, :] = 0.
            self.X[:, :] = 0.
            self.out[:, :] = 0.

    def zero_pad_front(self, zf):
        """
        Set zero-padding at beginning of frame.
        """
        self.zf = zf
        self.nfft = self.num_samples+self.zb+self.zf
        self.nbin = self.nfft//2+1
        if self.analysis_window is not None:
            self.analysis_window = np.concatenate((np.zeros(zf),
                                                   self.analysis_window))
        if self.synthesis_window is not None:
            self.synthesis_window = np.concatenate((np.zeros(zf),
                                                    self.synthesis_window))

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
            self.analysis_window = np.concatenate((self.analysis_window,
                                                   np.zeros(zb)))
        if self.synthesis_window is not None:
            self.synthesis_window = np.concatenate((self.synthesis_window,
                                                    np.zeros(zb)))

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
            Whether or not given coefficients (coeff) are in the frequency
            domain.
        """
        # apply zero-padding
        if zb is not None:
            self.zero_pad_back(zb)
        if zf is not None:
            self.zero_pad_front(zf)
        if not freq:
            # compute filter magnitude and phase spectrum
            self.H = self.freq_dtype(np.fft.rfft(coeff, self.nfft, axis=0))

            # check for sufficient zero-padding
            if self.nfft < (self.num_samples+len(coeff)-1):
                raise ValueError('Insufficient zero-padding for chosen number '
                                 'of samples per frame (L) and filter length '
                                 '(h). Require zero-padding such that new '
                                 'length is at least (L+h-1).')
        else:
            if len(coeff) != self.nbin:
                raise ValueError('Invalid length for frequency domain '
                                 'coefficients.')
            self.H = coeff

        # prepare filter if fixed input case
        if self.fixed_input:
            if self.num_channels == 1:
                self.H_multi = np.tile(self.H, (self.num_frames, 1))
            else:
                self.H_multi = np.tile(self.H, (self.num_frames, 1, 1))

    def analysis(self, x):
        """
        Parameters
        -----------
        x  : 2D numpy array, [samples, channels]
            Time-domain signal.
        """

        # ----check correct number of channels
        x_shape = x.shape
        if not self.mono:
            if len(x_shape) < 1:   # received mono
                raise ValueError('Received 1-channel signal. Expecting %d '
                                 'channels.' % self.num_channels)
            if x_shape[1] != self.num_channels:
                raise ValueError('Incorrect number of channels. Received %d, '
                                 'expecting %d.' % (x_shape[1],
                                                    self.num_channels))
        else:   # expecting mono
            if len(x_shape) > 1:    # received multi-channel
                raise ValueError('Received %d channels; expecting 1D mono '
                                 'signal.' % x_shape[1])

        # ----check number of frames
        if self.streaming:  # need integer multiple of hops

            if self.fixed_input:
                if x_shape[0] != self.num_frames*self.hop:
                    raise ValueError('Input must be of length %d; received %d '
                                     'samples.' % (self.num_frames*self.hop,
                                                   x_shape[0]))
            else:
                self.num_frames = int(np.ceil(x_shape[0]/self.hop))
                extra_samples = (self.num_frames*self.hop)-x_shape[0]
                if extra_samples:
                    if self.mono:
                        x = np.concatenate((x, np.zeros(extra_samples)))
                    else:
                        x = np.concatenate(
                            (x, np.zeros((extra_samples, self.num_channels))))

        # non-streaming
        # need at least num_samples for last frame
        # e.g.[hop|hop|...|hop|num_samples]
        else:

            if self.fixed_input:
                if x_shape[0] != (self.hop*(self.num_frames-1)+
                                  self.num_samples):
                    raise ValueError('Input must be of length %d; received %d '
                                     'samples.' % ((self.hop*(self.num_frames-1)
                                                    + self.num_samples),
                                                   x_shape[0]))
            else:
                if x_shape[0] < self.num_samples:
                    # raise ValueError('Not enough samples. Received %d; need \
                    #     at least %d.' % (x_shape[0],self.num_samples))
                    extra_samples = self.num_samples - x_shape[0]
                    if self.mono:
                        x = np.concatenate((x, np.zeros(extra_samples)))
                    else:
                        x = np.concatenate(
                            (x, np.zeros((extra_samples, self.num_channels))))
                    self.num_frames = 1
                else:

                    # calculate num_frames and append zeros if necessary
                    self.num_frames = \
                        int(np.ceil((x_shape[0]-self.num_samples)/self.hop) + 1)
                    extra_samples = ((self.num_frames-1)*self.hop +
                                     self.num_samples)-x_shape[0]
                    if extra_samples:
                        if self.mono:
                            x = np.concatenate((x, np.zeros(extra_samples)))
                        else:
                            x = np.concatenate(
                                (x,
                                 np.zeros((extra_samples, self.num_channels))))

        # ----allocate memory if necessary
        if not self.fixed_input:
            self.X = np.squeeze(np.zeros((self.num_frames,
                                          self.nbin,
                                          self.num_channels),
                                         dtype=self.freq_dtype))
            self.dft_frames = DFT(nfft=self.nfft,
                                  D=self.num_frames,
                                  analysis_window=self.analysis_window,
                                  synthesis_window=self.synthesis_window,
                                  transform=self.transform)

        # ----use appropriate function
        if self.streaming:
            self._analysis_streaming(x)
        else:
            self.reset()
            self._analysis_non_streaming(x)
                
        return self.X

    def _analysis_single(self, x_n):
        """
        Transform new samples to STFT domain for analysis.

        Parameters
        -----------
        x_n : numpy array
            [self.hop] new samples
        """

        # correct input size check in: dft.analysis()
        self.fresh_samples[:, ] = x_n[:, ]  # introduce new samples
        self.x_p[:, ] = self.old_samples   # save next state

        # apply DFT to current frame
        self.X[:] = self.dft.analysis(self.fft_in_buffer)

        # shift backwards in the buffer the state
        self.fft_in_state[:, ] = self.x_p[:, ]

    def _analysis_streaming(self, x):
        """
        STFT analysis for streaming case in which we expect
        [num_frames*hop] samples
        """

        if self.num_frames == 1:
            self._analysis_single(x)
        else:
            n = 0
            for k in range(self.num_frames):
                # introduce new samples
                self.fresh_samples[:, ] = x[n:n+self.hop, ]
                # save next state
                self.x_p[:, ] = self.old_samples

                # apply DFT to current frame
                self.X[k, ] = self.dft.analysis(self.fft_in_buffer)

                # shift backwards in the buffer the state
                self.fft_in_state[:, ] = self.x_p[:, ]

                n += self.hop

    def _analysis_non_streaming(self, x):
        """
        STFT analysis for non-streaming case in which we expect
        [(num_frames-1)*hop+num_samples] samples
        """

        ## ----- STRIDED WAY
        new_strides = (x.strides[0], self.hop * x.strides[0])
        new_shape = (self.num_samples, self.num_frames)

        if not self.mono:
            for c in range(self.num_channels):

                y = _as_strided(x[:, c], shape=new_shape, strides=new_strides)
                y = np.concatenate((np.zeros((self.zf, self.num_frames)), y,
                                    np.zeros((self.zb, self.num_frames))))

                if self.num_frames == 1:
                    self.X[:, c] = self.dft_frames.analysis(y[:, 0]).T
                else:
                    self.X[:, :, c] = self.dft_frames.analysis(y).T
        else:

            y = _as_strided(x, shape=new_shape, strides=new_strides)
            y = np.concatenate((np.zeros((self.zf, self.num_frames)), y,
                                np.zeros((self.zb, self.num_frames))))

            if self.num_frames == 1:
                self.X[:] = self.dft_frames.analysis(y[:, 0]).T
            else:
                self.X[:] = self.dft_frames.analysis(y).T

    def _check_input_frequency_dimensions(self, X):
        """
        Ensure that given frequency data is valid, i.e. number of channels and
        number of frequency bins.

        If fixed_input=True, ensure expected number of frames. Otherwise, infer 
        from given data.

        Axis order of X should be : [frames, frequencies, channels]
        """

        # check number of frames and correct number of bins
        X_shape = X.shape
        if len(X_shape) == 1:  # single channel, one frame
            num_frames = 1
        elif len(X_shape) == 2 and not self.mono: # multi-channel, one frame
            num_frames = 1
        elif len(X_shape) == 2 and self.mono: # single channel, multiple frames
            num_frames = X_shape[0]
        elif len(X_shape) == 3 and not self.mono: # multi-channel, multiple frames
            num_frames = X_shape[0]
        else:
            raise ValueError('Invalid input shape.')

        # check number of bins
        if num_frames == 1:
            if X_shape[0] != self.nbin:
                raise ValueError('Invalid number of frequency bins! Expecting '
                                 '%d, got %d.' % (self.nbin, X_shape[0]))
        else:
            if X_shape[1] != self.nbin:
                raise ValueError('Invalid number of frequency bins! Expecting'
                                 ' %d, got %d.' % (self.nbin, X_shape[1]))

        # check number of frames, if fixed input size
        if self.fixed_input:
            if num_frames != self.num_frames:
                raise ValueError('Input must have %d frames!' % self.num_frames)
            self.X[:] = X  # reset if size is alright
        else:
            self.X = X
            self.num_frames = num_frames

        return self.X

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
            return

        if X is not None:
            self._check_input_frequency_dimensions(X)

        # use appropriate function
        if self.num_frames == 1:
            self._process_single()
        elif self.num_frames > 1:
            self._process_multiple()

        return self.X

    def _process_single(self):

        np.multiply(self.X, self.H, self.X)

    def _process_multiple(self):

        if not self.fixed_input:
            if self.mono:
                self.H_multi = np.tile(self.H, (self.num_frames, 1))
            else:
                self.H_multi = np.tile(self.H, (self.num_frames, 1, 1))

        np.multiply(self.X, self.H_multi, self.X)

    def synthesis(self, X=None):
        """
        Parameters
        -----------
        X  : numpy array of frequency content
            X can take on multiple shapes:
            1) (N,) if it is single channel and only one frame
            2) (N,D) if it is multi-channel and only one frame
            3) (F,N) if it is single channel but multiple frames
            4) (F,N,D) if it is multi-channel and multiple frames
            where:
            - F is the number of frames
            - N is the number of frequency bins
            - D is the number of channels


        Returns
        -----------
        x_r : numpy array
            Reconstructed time-domain signal.

        """

        if X is not None:
            self._check_input_frequency_dimensions(X)

        # use appropriate function
        if self.num_frames == 1:
            return self._synthesis_single()
        elif self.num_frames > 1:
            return self._synthesis_multiple()

    def _synthesis_single(self):
        """
        Transform to time domain and reconstruct output with overlap-and-add.

        Returns
        -------
        numpy array
            Reconstructed array of samples of length <self.hop>.
        """

        # apply IDFT to current frame
        self.dft.synthesis(self.X)

        return self._overlap_and_add()

    def _synthesis_multiple(self):
        """
        Apply STFT analysis to multiple frames.

        Returns
        -----------
        x_r : numpy array
            Recovered signal.

        """

        # synthesis + overlap and add
        if not self.mono:

            x_r = np.zeros((self.num_frames*self.hop, self.num_channels),
                           dtype=self.time_dtype)

            n = 0
            for f in range(self.num_frames):

                # apply IDFT to current frame and reconstruct output
                x_r[n:n+self.hop, ] = self._overlap_and_add(
                    self.dft.synthesis(self.X[f, :, :]))
                n += self.hop

        else:

            x_r = np.zeros(self.num_frames*self.hop, dtype=self.time_dtype)

            # treat number of frames as the multiple channels for DFT
            if not self.fixed_input:
                self.dft_frames = DFT(nfft=self.nfft,
                                      D=self.num_frames,
                                      analysis_window=self.analysis_window,
                                      synthesis_window=self.synthesis_window,
                                      transform=self.transform)

            # back to time domain
            mx = self.dft_frames.synthesis(self.X.T)

            # overlap and add
            n = 0
            for f in range(self.num_frames):
                x_r[n:n+self.hop, ] = self._overlap_and_add(mx[:, f])
                n += self.hop

        return x_r

    def _overlap_and_add(self, x=None):

        if x is None:
            x = self.dft.x

        self.out[:, ] = x[0:self.hop, ]  # fresh output samples

        # add state from previous frames when overlap is used
        if self.n_state_out > 0:
            m = np.minimum(self.hop, self.n_state_out)
            self.out[:m, ] += self.y_p[:m, ]
            # update state variables
            self.y_p[:-self.hop, ] = self.y_p[self.hop:, ]  # shift out left
            self.y_p[-self.hop:, ] = 0.
            self.y_p[:, ] += x[-self.n_state_out:, ]

        return self.out


" ---------------------------------------------------------------------------- "
" --------------- One-shot functions to avoid creating object. --------------- "
" ---------------------------------------------------------------------------- "
# Authors: Robin Scheibler, Ivan Dokmanic, Sidney Barthe

def analysis(x, L, hop, win=None, zp_back=0, zp_front=0):
    """
    Convenience function for one-shot STFT

    Parameters
    ----------
    x: array_like, (n_samples) or (n_samples, n_channels)
        input signal 
    L: int
        frame size
    hop: int
        shift size between frames
    win: array_like
        the window to apply (default None)
    zp_back: int
        zero padding to apply at the end of the frame
    zp_front: int
        zero padding to apply at the beginning of the frame

    Returns
    -------
    X: ndarray, (n_frames, n_frequencies) or (n_frames, n_frequencies, n_channels)
        The STFT of x
    """

    if x.ndim == 2:
        channels = x.shape[1]
    else:
        channels = 1

    the_stft = STFT(L, hop=hop, analysis_window=win, channels=channels, precision=x.dtype)

    if zp_back > 0:
        the_stft.zero_pad_back(zp_back)

    if zp_front > 0:
        the_stft.zero_pad_front(zp_front)

    # apply transform
    return the_stft.analysis(x)


# inverse STFT
def synthesis(X, L, hop, win=None, zp_back=0, zp_front=0):
    """
    Convenience function for one-shot inverse STFT

    Parameters
    ----------
    X: array_like (n_frames, n_frequencies) or (n_frames, n_frequencies, n_channels)
        The data
    L: int
        frame size
    hop: int
        shift size between frames
    win: array_like
        the window to apply (default None)
    zp_back: int
        zero padding to apply at the end of the frame
    zp_front: int
        zero padding to apply at the beginning of the frame

    Returns
    -------
    x: ndarray, (n_samples) or (n_samples, n_channels)
        The inverse STFT of X
    """

    if X.ndim == 3:
        channels = X.shape[2]
    else:
        channels = 1

    the_stft = STFT(L, hop=hop, synthesis_window=win, channels=channels, precision=X.dtype)

    if zp_back > 0:
        the_stft.zero_pad_back(zp_back)

    if zp_front > 0:
        the_stft.zero_pad_front(zp_front)

    # apply transform
    return the_stft.synthesis(X)


def compute_synthesis_window(analysis_window, hop):
    """
    Computes the optimal synthesis window given an analysis window
    and hop (frame shift). The procedure is described in

    D. Griffin and J. Lim, *Signal estimation from modified short-time Fourier transform,*
    IEEE Trans. Acoustics, Speech, and Signal Process.,
    vol. 32, no. 2, pp. 236-243, 1984.

    Parameters
    ----------
    analysis_window: array_like
        The analysis window
    hop: int
        The frame shift
    """

    norm = np.zeros_like(analysis_window)
    L = analysis_window.shape[0]

    # move the window back as far as possible while still overlapping
    n = 0
    while n - hop > -L:
        n -= hop

    # now move the window and sum all the contributions
    while n < L:
        if n == 0:
            norm += analysis_window ** 2
        elif n < 0:
            norm[:n+L] += analysis_window[-n-L:] ** 2
        else:
            norm[n:] += analysis_window[:-n] ** 2

        n += hop

    return analysis_window / norm
