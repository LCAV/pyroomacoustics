# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

'''Collection of spectral estimation methods.

**This module is deprecated.** It is replaced by the methods of :py:obj:`pyroomacoustics.transform`
'''
from __future__ import division

import sys, warnings
import numpy as np
from numpy.lib.stride_tricks import as_strided as _as_strided

# a routine for long convolutions using overlap add method
def overlap_add(in1, in2, L):

    warnings.warn("The function pyroomacoustics.overlap is deprecated and will disappear soon.",
            DeprecationWarning)

    # set the shortest sequence as the filter
    if (len(in1) > len(in2)):
        x = in1
        h = in2
    else:
        h = in1
        x = in2

    # filter length
    M = len(h)

    # FFT size
    N = L + M - 1

    # frequency domain filter (zero-padded)
    H = np.fft.rfft(h, N)

    # prepare output signal
    ylen = int(np.ceil(len(x) / float(L)) * L + M - 1)
    y = np.zeros(ylen)

    # overlap add
    i = 0
    while (i < len(x)):
        y[i:i + N] += np.fft.irfft(np.fft.rfft(x[i:i + L], N) * H, N)
        i += L

    return y[:len(x) + M - 1]


# Nicely plot the spectrogram
def spectroplot(Z, N, hop, fs, fdiv=None, tdiv=None,
                vmin=None, vmax=None, cmap=None, interpolation='none', colorbar=True):

    warnings.warn("The function pyroomacoustics.spectroplot is deprecated and will disappear soon.",
            DeprecationWarning)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn('Matplotlib is required for plotting')
        return

    plt.imshow(
        20 * np.log10(np.abs(Z[:N // 2 + 1, :])),
        aspect='auto',
        origin='lower',
        vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)

    # label y axis correctly
    plt.ylabel('Freq [Hz]')
    yticks = plt.getp(plt.gca(), 'yticks')
    plt.setp(plt.gca(), 'yticklabels', np.round(yticks / float(N) * fs))
    if (fdiv is not None):
        tick_lbls = np.arange(0, fs / 2, fdiv)
        tick_locs = tick_lbls * N / fs
        plt.yticks(tick_locs, tick_lbls)

    # label x axis correctly
    plt.xlabel('Time [s]')
    xticks = plt.getp(plt.gca(), 'xticks')
    plt.setp(plt.gca(), 'xticklabels', xticks / float(fs) * hop)
    if (tdiv is not None):
        unit = float(hop) / fs
        length = unit * Z.shape[1]
        tick_lbls = np.arange(0, int(length), tdiv)
        tick_locs = tick_lbls * fs / hop
        plt.xticks(tick_locs, tick_lbls)

    if colorbar is True:
        plt.colorbar(orientation='horizontal')

# A more general implementation of STFT


def stft(x, L, hop, transform=np.fft.fft, win=None, zp_back=0, zp_front=0):
    '''
    Parameters
    ----------
    x: 
        input signal
    L: 
        frame size
    hop: 
        shift size between frames
    transform: 
        the transform routine to apply (default FFT)
    win: 
        the window to apply (default None)
    zp_back: 
        zero padding to apply at the end of the frame
    zp_front: 
        zero padding to apply at the beginning of the frame

    Returns
    -------
    The STFT of x
    '''

    warnings.warn("The function pyroomacoustics.stft is deprecated and will disappear soon. It will be replaced by the pyroomacoustics.transform.STFT object.",
            DeprecationWarning)

    # the transform size
    N = L + zp_back + zp_front

    # window needs to be same size as transform
    if (win is not None and len(win) != N):
        print('Window length need to be equal to frame length + zero padding.')
        sys.exit(-1)

    # reshape
    new_strides = (hop * x.strides[0], x.strides[0])
    new_shape = ((len(x) - L) // hop + 1, L)
    y = _as_strided(x, shape=new_shape, strides=new_strides)

    # add the zero-padding
    y = np.concatenate(
            (
                np.zeros( (y.shape[0], zp_front), dtype=x.dtype), 
                y, 
                np.zeros( (y.shape[0], zp_back), dtype=x.dtype)
            ), axis=1)

    # apply window if needed
    if (win is not None):
        y = win.astype(x.dtype) * y

    # transform along rows
    Z = transform(y, axis=1)

    # apply transform
    return Z


# inverse STFT
def istft(X, L, hop, transform=np.fft.ifft, win=None, zp_back=0, zp_front=0):

    warnings.warn("The function pyroomacoustics.istft is deprecated and will disappear soon. It will be replaced by the pyroomacoustics.transform.STFT object.",
            DeprecationWarning)

    # the transform size
    N = L + zp_back + zp_front

    # window needs to be same size as transform
    if (win is not None and len(win) != N):
        print('Window length need to be equal to frame length + zero padding.')
        sys.exit(-1)

    # inverse transform
    iX = transform(X, axis=1)
    if (iX.dtype == 'complex128'):
        iX = np.real(iX)

    # apply synthesis window if necessary
    if (win is not None):
        iX *= win

    # create output signal
    x = np.zeros(X.shape[0] * hop + (L - hop) + zp_back + zp_front)

    # overlap add
    for i in range(X.shape[0]):
        x[i * hop:i * hop + N] += iX[i]

    return x


# FreqVec: given FFT size and sampling rate, returns a vector of real
# frequencies
def freqvec(N, fs, centered=False):
    '''
    Compute the vector of frequencies corresponding to DFT bins.
    Parameters
    ----------
    N: int
        FFT length
    fs: int
        sampling rate of the signal
    shift: int
        False if the DC is at the beginning, True if the DC is centered
    '''

    # Create a centered vector. The (1-N%2) is to correct for even/odd length
    vec = np.arange(-N // 2 + (1 - N % 2), N // 2 + 1) * float(fs) / float(N)

    # Shift positive/negative frequencies if needed. Again (1-N%2) for
    # even/odd length
    if centered:
        return vec
    else:
        return np.concatenate((vec[N // 2 - (1 - N % 2):], vec[0:N // 2 - 1]))
