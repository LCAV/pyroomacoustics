# Author: Eric Bezzam
# Last modification: February 22, 2019

"""
Class for performing the Discrete Fourier Transform (DFT) and inverse DFT for
real signals, including multichannel. It is also possible to specific an 
analysis or synthesis window.

When available, it is possible to use the ``pyfftw`` or ``mkl_fft`` packages.
Otherwise the default is to use ``numpy.fft.rfft``/``numpy.fft.irfft``.

More on ``pyfftw`` can be read here: 
https://pyfftw.readthedocs.io/en/latest/index.html

More on ``mkl_fft`` can be read here: 
https://github.com/IntelPython/mkl_fft
"""

import numpy as np
from numpy.fft import rfft, irfft
import warnings

try:
    import pyfftw
    pyfftw_available = True
except ImportError:
    pyfftw_available = False

try:
    import mkl_fft    # https://github.com/IntelPython/mkl_fft
    mkl_available = True
except ImportError:
    mkl_available = False


class DFT(object):
    """
    Class for performing the Discrete Fourier Transform (DFT) of real signals.

    Attributes
    ----------
    X: numpy array
        Real DFT computed by ``analysis``.
    x: numpy array
        IDFT computed by ``synthesis``.
    nfft: int
        FFT size.
    D: int
        Number of channels.
    transform: str
        Which FFT package will be used.
    analysis_window: numpy array
        Window to be applied before DFT.
    synthesis_window: numpy array
        Window to be applied after inverse DFT.
    axis : int
        Axis over which to compute the FFT.
    precision, bits : string, np.float32, np.float64, np.complex64, np.complex128
        How many precision bits to use for the input. Twice the amount will be used
        for complex spectrum.

    Parameters
    ----------
    nfft: int
        FFT size.
    D: int, optional
        Number of channels. Default is 1.
    analysis_window: numpy array, optional
        Window to be applied before DFT. Default is no window.
    synthesis_window: numpy array, optional
        Window to be applied after inverse DFT. Default is no window.
    transform: str, optional
        which FFT package to use: ``numpy``, ``pyfftw``, or ``mkl``. Default is 
        ``numpy``.
    axis : int, optional
        Axis over which to compute the FFT. Default is first axis.
    precision : string, np.float32, np.float64, np.complex64, np.complex128, optional
        How many precision bits to use for the input.
        If 'single'/np.float32/np.complex64, 32 bits for real inputs or 64 for complex spectrum.
        Otherwise, cast to 64 bits for real inputs or 128 for complex spectrum (default).

    """

    def __init__(self, nfft, D=1, analysis_window=None, synthesis_window=None, 
        transform='numpy', axis=0, precision='double', bits=None):

        self.nfft = nfft
        self.D = D
        self.axis=axis

        if bits is not None and precision is not None:
            warnings.warn('Deprecated keyword "bits" ignored in favor of new keyword "precision"', DeprecationWarning)
        elif bits is not None and precision is None:
            warnings.warn('Keyword "bits" is deprecated and has been replaced by "precision"')
            if bits == 32:
                precision = 'single'
            elif bits == 64:
                precision = 'double'

        if precision == np.float32 or precision == np.complex64 or precision == 'single':
            time_dtype = np.float32
            freq_dtype = np.complex64
        else:
            time_dtype = np.float64
            freq_dtype = np.complex128

        if axis==0:
            self.x = np.squeeze(np.zeros((self.nfft, self.D), 
                dtype=time_dtype))
            self.X = np.squeeze(np.zeros((self.nfft//2+1, self.D),
                dtype=freq_dtype))
        elif axis==1:
            self.x = np.squeeze(np.zeros((self.D, self.nfft),
                dtype=time_dtype))
            self.X = np.squeeze(np.zeros((self.D, self.nfft//2+1),
                dtype=freq_dtype))
        else:
            raise ValueError("Invalid 'axis' option. Must be 0 or 1.")

        if analysis_window is not None:
            if axis==0 and D>1:
                self.analysis_window = analysis_window[:,np.newaxis].astype(time_dtype)
            else:
                self.analysis_window = analysis_window.astype(time_dtype)
        else:
            self.analysis_window = None

        if synthesis_window is not None:
            if axis==0 and D>1:
                self.synthesis_window = synthesis_window[:,np.newaxis].astype(time_dtype)
            else:
                self.synthesis_window = synthesis_window.astype(time_dtype)
        else:
            self.synthesis_window=None


        if transform == 'fftw':
            if pyfftw_available:
                from pyfftw import empty_aligned, FFTW
                self.transform = transform
                # allocate input (real) and output for pyfftw
                if self.D==1:
                    self.a = empty_aligned(self.nfft, dtype=time_dtype)
                    self.b = empty_aligned(self.nfft//2+1, dtype=freq_dtype)
                    self._forward = FFTW(self.a, self.b)
                    self._backward = FFTW(self.b, self.a, 
                        direction='FFTW_BACKWARD')
                else:
                    if axis==0:
                        self.a = empty_aligned([self.nfft, self.D], 
                            dtype=time_dtype)
                        self.b = empty_aligned([self.nfft//2+1, self.D], 
                            dtype=freq_dtype)
                    elif axis==1:
                        self.a = empty_aligned([self.D, self.nfft], 
                            dtype=time_dtype)
                        self.b = empty_aligned([self.D, self.nfft//2+1], 
                            dtype=freq_dtype)
                    self._forward = FFTW(self.a, self.b, 
                        axes=(self.axis, ))
                    self._backward = FFTW(self.b, self.a, axes=(self.axis, ), 
                        direction='FFTW_BACKWARD')
            else: 
                warnings.warn("Could not import pyfftw wrapper for fftw functions. Using numpy's rfft instead.")
                self.transform = 'numpy'
        elif transform == 'mkl':
            if mkl_available:
                import mkl_fft
                self.transform = 'mkl'
            else:
                warnings.warn("Could not import mkl wrapper. Using numpy's rfft instead.")
                self.transform = 'numpy'
        else:
            self.transform = 'numpy'


    def analysis(self, x):
        """
        Perform frequency analysis of a real input using DFT.

        Parameters
        ----------
        x : numpy array
            Real signal in time domain.

        Returns
        -------
        numpy array
            DFT of input.
        """

        # check for valid input
        if x.shape != self.x.shape:
            raise ValueError('Invalid input dimensions! Got (%d, %d), expecting (%d, %d).' 
                % (x.shape[0], x.shape[1],
                self.x.shape[0], self.x.shape[1]))

        # apply window if needed
        if self.analysis_window is not None:
            np.multiply(self.analysis_window, x, x)

        # apply DFT
        if self.transform == 'fftw':
            self.a[:,] = x
            self.X[:,] = self._forward()
        elif self.transform == 'mkl':
            self.X[:,] = mkl_fft.rfft_numpy(x, self.nfft, axis=self.axis)
        else:
            self.X[:,] = rfft(x, self.nfft, axis=self.axis)

        return self.X


    def synthesis(self, X=None):
        """
        Perform time synthesis of frequency domain to real signal using the 
        inverse DFT.

        Parameters
        ----------
        X : numpy array, optional
            Complex signal in frequency domain. Default is to use DFT computed
            from ``analysis``.

        Returns
        -------
        numpy array
            IDFT of ``self.X`` or input if given.
        """

        # check for valid input
        if X is not None:
            if X.shape != self.X.shape:
                raise ValueError('Invalid input dimensions! Got (%d, %d), expecting (%d, %d).' 
                    % (X.shape[0], X.shape[1],
                    self.X.shape[0], self.X.shape[1]))

            self.X[:,] = X

        # inverse DFT
        if self.transform == 'fftw':
            self.b[:] = self.X
            self.x[:,] = self._backward()
        elif self.transform == 'mkl':
            self.x[:,] = mkl_fft.irfft_numpy(self.X, self.nfft, axis=self.axis)
        else:
            self.x[:,] = irfft(self.X, self.nfft, axis=self.axis)

        # apply window if needed
        if self.synthesis_window is not None:
            np.multiply(self.synthesis_window, self.x, self.x)

        return self.x
