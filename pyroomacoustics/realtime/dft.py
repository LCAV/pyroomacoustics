# Author: Eric Bezzam
# Date: Jan 31, 2016

"""
Methods for using the Discrete Fourier Transform (DFT) for the analysis and
synthesis of real signals.

When available, it is able to use pyfftw package or mkl fft.
"""

import numpy as np
from numpy.fft import rfft
from numpy.fft import irfft
import warnings


try:
    import pyfftw
    pyfftw_available = True
except ImportError:
    pyfftw_available = False

try:
    import mkl_fft  # https://github.com/LCAV/mkl_fft
    mkl_available = True
except ImportError:
    mkl_available = False


class DFT(object):
    """
    Parent class for performing Fourier Analysis of real signals through the
    Discrete Fourier Transform (DFT).

    Parameters
    ----------
    nfft: int
        FFT size.
    D: int
        Number of channels. Default is 1.
    analysis_window: numpy array
        Window to be applied before DFT.
    synthesis_window: numpy array
        Window to be applied after inverse DFT.
    transform: str
        which FFT package to use: 'numpy', 'pyfftw', or 'mkl'

    """

    def __init__(self, nfft, D=1, analysis_window=None, synthesis_window=None, transform='numpy'):

        self.nfft = nfft
        self.D = D
        
        self.nbin = self.nfft//2+1
        self.X = np.squeeze(np.zeros((self.nbin,self.D),dtype='complex64'))
        self.x = np.squeeze(np.zeros((self.nfft,self.D),dtype='float32'))

        if analysis_window is not None:
            if self.D==1:
                self.analysis_window = analysis_window.astype('float32')
            else:
                self.analysis_window = np.tile(analysis_window, (self.D,1)).astype('float32').T
        else:
            self.analysis_window = None
        if synthesis_window is not None:
            if self.D==1:
                self.synthesis_window = synthesis_window.astype('float32')
            else:
                self.synthesis_window = np.tile(synthesis_window, (self.D,1)).astype('float32').T
        else:
            self.synthesis_window=None


        if transform == 'fftw':
            if pyfftw_available:
                import pyfftw
                self.transform = transform
                # allocate input and output (assuming real input) for pyfftw
                if self.D==1:
                    self.a = pyfftw.empty_aligned(self.nfft, dtype='float32')
                    self.b = pyfftw.empty_aligned(self.nbin, dtype='complex64')
                    self.c = pyfftw.empty_aligned(self.nfft, dtype='float32')
                    self.forward = pyfftw.FFTW(self.a, self.b)
                    self.backward = pyfftw.FFTW(self.b, self.c, 
                        direction='FFTW_BACKWARD')
                else:
                    self.a = pyfftw.empty_aligned([self.nfft,self.D], dtype='float32')
                    self.b = pyfftw.empty_aligned([self.nbin,self.D], dtype='complex64')
                    self.c = pyfftw.empty_aligned([self.nfft,self.D], dtype='float32')
                    self.forward = pyfftw.FFTW(self.a, self.b, axes=(0, ))
                    self.backward = pyfftw.FFTW(self.b, self.c, axes=(0, ), 
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
        x: numpy array (float32)
            Real input signal in time domain. Must be of size (N,D).
        plot_spec: bool
            Whether or not to plot frequency magnitude spectrum.

        Returns
        -------
        Frequency spectrum, numpy array of size (N/2+1,D) and of type complex64.
        """

        # check for valid input
        if self.D!=1:
            if x.shape!=(self.nfft,self.D):
                raise ValueError('Invalid input dimensions.')
        elif self.D==1:
            if x.ndim!=1 and x.shape[0]!=self.nfft:
                raise ValueError('Invalid input dimensions.')
        # apply window if needed
        if self.analysis_window is not None:
            try:
                np.multiply(self.analysis_window, x, x)
            except:
                self.analysis_window = self.analysis_window.astype(x.dtype, 
                    copy=False)
                np.multiply(self.analysis_window, x, x)
        # apply DFT
        if self.transform == 'fftw':
            self.a[:,] = x
            self.X[:,] = self.forward()
        elif self.transform == 'mkl':
            self.X[:,] = mkl_fft.rfft(x,axis=0)
        else:
            self.X[:,] = rfft(x,axis=0)

        return self.X


    def synthesis(self, X=None):
        """
        Perform time synthesis of frequency domain to real signal using the 
        inverse DFT.

        Parameters
        ----------
        X: numpy array (complex64)
            Real input signal in time domain. Must be of size (N/2+1,D).
            Default is previously computed DFT.
        plot_time: bool
            Whether or not to plot time waveform.

        Returns
        -------
        Time domain signal, numpy array of size (N,D) and of type float32.
        """

        # check for valid input
        if X is not None:
            if self.D!=1:
                if X.shape!=(self.nbin,self.D):
                    raise ValueError('Invalid input dimensions.')
            elif self.D==1:
                if X.ndim!=1 and X.shape[0]!=self.nbin:
                    raise ValueError('Invalid input dimensions.')
            self.X[:,] = X
        # inverse DFT
        if self.transform == 'fftw':
            self.b[:] = self.X
            self.x[:,] = self.backward()
        elif self.transform == 'mkl':
            self.x[:,] = mkl_fft.irfft(self.X,axis=0)
        else:
            self.x[:,] = irfft(self.X, axis=0)
        # apply window if needed
        if self.synthesis_window is not None:
            np.multiply(self.synthesis_window, self.x[:,], self.x[:,])

        return self.x

