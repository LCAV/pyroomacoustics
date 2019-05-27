from __future__ import division

import numpy as np
from scipy import linalg as la

try:
    import mkl_fft._numpy_fft as fft
except ImportError:
    import numpy.fft as fft

def deconvolve(y, s, length=None, thresh=0.0):
    '''
    Deconvolve an excitation signal from an impulse response

    Parameters
    ----------

    y : ndarray
        The recording
    s : ndarray
        The excitation signal
    length: int, optional
        the length of the impulse response to deconvolve
    thresh : float, optional
        ignore frequency bins with power lower than this
    '''

    # FFT length including zero padding
    n = y.shape[0] + s.shape[0] - 1

    # let FFT size be even for convenience
    if n % 2 != 0:
        n += 1

    # when unknown, pick the filter size as size of test signal
    if length is None:
        length = n

    # Forward transforms
    Y  = fft.rfft(np.array(y, dtype=np.float32), n=n) / np.sqrt(n)
    S = fft.rfft(np.array(s, dtype=np.float32), n=n) / np.sqrt(n)

    # Only do the division where S is large enough
    H = np.zeros(*Y.shape, dtype=Y.dtype)
    I = np.where(np.abs(S) > thresh)
    H[I] = Y[I] / S[I]

    # Inverse transform
    h = fft.irfft(H, n=n)

    return h[:length]

def wiener_deconvolve(y, x, length=None, noise_variance=1., let_n_points=15, let_div_base=2):
    '''
    Deconvolve an excitation signal from an impulse response

    We use Wiener filter

    Parameters
    ----------

    y : ndarray
        The recording
    x : ndarray
        The excitation signal
    length: int, optional
        the length of the impulse response to deconvolve
    noise_variance : float, optional
        estimate of the noise variance
    let_n_points: int
        number of points to use in the LET approximation
    let_div_base: float
        the divider used for the LET grid
    '''

    # FFT length including zero padding
    n = y.shape[0] + x.shape[0] - 1

    # let FFT size be even for convenience
    if n % 2 != 0:
        n += 1

    # when unknown, pick the filter size as size of test signal
    if length is None:
        length = n

    # Forward transforms
    Y  = fft.rfft(np.array(y, dtype=np.float32), n=n) / np.sqrt(n)  # recording
    X = fft.rfft(np.array(x, dtype=np.float32), n=n) / np.sqrt(n)   # test signal

    # Squared amplitude of test signal
    X_sqm = np.abs(X)**2

    # approximate SNR
    SNR_hat = np.maximum(1e-7, ((np.linalg.norm(Y)**2 / np.linalg.norm(X)**2) - noise_variance))  / noise_variance
    dividers = let_div_base**np.linspace(-let_n_points/2, let_n_points, let_n_points)
    SNR_grid = SNR_hat / dividers

    # compute candidate points
    G = (X_sqm[:,None] / (X_sqm[:,None] + 1./SNR_grid[None,:])) * Y[:,None]
    H_candidates = G / X[:,None]

    # find the best linear combination of the candidates
    weights = np.linalg.lstsq(G, Y, rcond=None)[0]

    # compute the estimated filter
    H = np.squeeze(np.dot(H_candidates, weights))
    h = fft.irfft(H, n=n)

    return h[:length]

