from __future__ import division

import numpy as np

try:
    import mkl_fft as fft
except:
    from numpy import fft

def tdoa(signal, reference, interp=1, phat=False, fs=1, t_max=None):
    '''
    Estimates the shift of array signal with respect to reference
    using generalized cross-correlation

    Parameters
    ----------
    signal: array_like
        The array whose tdoa is measured
    reference: array_like
        The reference array
    interp: int, optional
        The interpolation factor for the output array, default 1.
    phat: bool, optional
        Apply the PHAT weighting (default False)
    fs: int or float, optional
        The sampling frequency of the input arrays, default=1

    Returns
    -------
    The estimated delay between the two arrays
    '''

    signal = np.array(signal)
    reference = np.array(reference)

    N1 = signal.shape[0]
    N2 = reference.shape[0]

    r_12 = correlate(signal, reference, interp=interp, phat=phat)

    delay = (np.argmax(np.abs(r_12)) / interp  - (N2 - 1) ) / fs

    return delay

def correlate(x1, x2, interp=1, phat=False):
    '''
    Compute the cross-correlation between x1 and x2

    Parameters
    ----------
    x1,x2: array_like
        The data arrays
    interp: int, optional
        The interpolation factor for the output array, default 1.
    phat: bool, optional
        Apply the PHAT weighting (default False)

    Returns
    -------
    The cross-correlation between the two arrays
    '''

    N1 = x1.shape[0]
    N2 = x2.shape[0]

    N = N1 + N2 - 1

    X1 = fft.rfft(x1, n=N)
    X2 = fft.rfft(x2, n=N)

    if phat:
        eps1 = np.mean(np.abs(X1)) * 1e-10
        X1 /= (np.abs(X1) + eps1)
        eps2 = np.mean(np.abs(X2)) * 1e-10
        X2 /= (np.abs(X2) + eps2)

    m = np.minimum(N1, N2)

    out = fft.irfft(X1*np.conj(X2), n=int(N*interp))

    return np.concatenate([out[-interp*(N2-1):], out[:(interp*N1)]])

def delay_estimation(x1, x2, L):
    '''
    Estimate the delay between x1 and x2.
    L is the block length used for phat
    '''

    K = int(np.minimum(x1.shape[0], x2.shape[0]) / L)

    delays = np.zeros(K)
    for k in range(K):
        delays[k] = tdoa(x1[k*L:(k+1)*L], x2[k*L:(k+1)*L], phat=True)

    return int(np.median(delays))


def time_align(ref, deg, L=4096):
    '''
    return a copy of deg time-aligned and of same-length as ref.
    L is the block length used for correlations.
    '''

    # estimate delay of signal
    from numpy import zeros, minimum
    delay = delay_estimation(ref, deg, L)

    # time-align with reference segment for error metric computation
    sig = zeros(ref.shape[0])
    if (delay >= 0):
        length = minimum(deg.shape[0], ref.shape[0]-delay)
        sig[delay:length+delay] = deg[:length]
    else:
        length = minimum(deg.shape[0]+delay, ref.shape[0])
        sig = zeros(ref.shape)
        sig[:length] = deg[-delay:-delay+length]

    return sig

