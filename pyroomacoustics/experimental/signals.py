'''
A few test signals like sweeps and stuff.
'''
from __future__ import division, print_function
import numpy as np

def window(signal, n_win):
    ''' window the signal at beginning and end with window of size n_win/2 '''

    win = np.hanning(2*n_win)

    sig_copy = signal.copy()

    sig_copy[:n_win] *= win[:n_win]
    sig_copy[-n_win:] *= win[-n_win:]

    return sig_copy

def exponential_sweep(T, fs, f_lo=0., f_hi=1., fade=None, ascending=False):
    '''
    Exponential sine sweep

    Parameters
    ----------
    T: float
        length in seconds
    fs: 
        sampling frequency
    f_lo: float
        lowest frequency in fraction of fs (default 0)
    f_hi: float
        lowest frequency in fraction of fs (default 1)
    fade: float, optional
        length of fade in and out in seconds (default 0)
    ascending: bool, optional
    '''

    Ts = 1./fs   # Sampling period in [s]
    N = np.floor(T/Ts)  # number of samples
    n  = np.arange(0, N, dtype='float64')  # Sample index

    om1 = 2 * np.pi * (f_lo * fs)
    om2 = 2 * np.pi * (f_hi * fs)

    sweep = 0.95 * np.sin(om1*N*Ts / np.log(om2/om1) * (np.exp(n/N*np.log(om2/om1)) - 1))

    if not ascending:
        sweep = sweep[::-1]

    if fade:
        sweep = window(sweep, int(fs * fade))

    return sweep

def linear_sweep(T, fs, f_lo=0., f_hi=1., fade=None, ascending=False):
    '''
    Linear sine sweep

    Parameters
    ----------
    T: float
        length in seconds
    fs: 
        sampling frequency
    f_lo: float
        lowest frequency in fraction of fs (default 0)
    f_hi: float
        lowest frequency in fraction of fs (default 1)
    fade: float, optional
        length of fade in and out in seconds (default 0)
    ascending: bool, optional
    '''

    Ts = 1./fs   # Sampling period in [s]

    N = np.floor(T/Ts)
    n  = np.arange(0, N, dtype='float64')  # Sample index

    om1 = 2 * np.pi * f_lo * fs
    om2 = 2 * np.pi * f_hi * fs

    sweep = 0.95 * np.sin(2 * np.pi * 0.5 * (f1 + (f2 - f1) * n / N) * n / fs)

    if not ascending:
        sweep = sweep[::-1]

    if fade:
        sweep = window(sweep, int(fs * fade))

    return sweep

