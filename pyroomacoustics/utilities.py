# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

import numpy as np
from scipy import signal

from .parameters import constants, eps


def to_16b(signal):
    '''
    converts float 32 bit signal (-1 to 1) to a signed 16 bits representation
    No clipping in performed, you are responsible to ensure signal is within
    the correct interval.
    '''
    return ((2**15-1)*signal).astype(np.int16)

def clip(signal, high, low):
    '''Clip a signal from above at high and from below at low.'''
    s = signal.copy()

    s[np.where(s > high)] = high
    s[np.where(s < low)] = low

    return s


def normalize(signal, bits=None):
    '''
    normalize to be in a given range. The default is to normalize the maximum
    amplitude to be one. An optional argument allows to normalize the signal
    to be within the range of a given signed integer representation of bits.
    '''

    s = signal.copy()

    s /= np.abs(s).max()

    # if one wants to scale for bits allocated
    if bits is not None:
        s *= 2 ** (bits - 1)
        s = clip(signal, 2 ** (bits - 1) - 1, -2 ** (bits - 1))

    return s


def normalize_pwr(sig1, sig2):
    '''Normalize sig1 to have the same power as sig2.'''

    # average power per sample
    p1 = np.mean(sig1 ** 2)
    p2 = np.mean(sig2 ** 2)

    # normalize
    return sig1.copy() * np.sqrt(p2 / p1)


def highpass(signal, Fs, fc=None, plot=False):
    ''' Filter out the really low frequencies, default is below 50Hz '''

    if fc is None:
        fc = constants.get('fc_hp')

    # have some predefined parameters
    rp = 5  # minimum ripple in dB in pass-band
    rs = 60   # minimum attenuation in dB in stop-band
    n = 4    # order of the filter
    type = 'butter'

    # normalized cut-off frequency
    wc = 2. * fc / Fs

    # design the filter
    from scipy.signal import iirfilter, lfilter, freqz
    b, a = iirfilter(n, Wn=wc, rp=rp, rs=rs, btype='highpass', ftype=type)

    # plot frequency response of filter if requested
    if (plot):
        import matplotlib.pyplot as plt
        w, h = freqz(b, a)

        plt.figure()
        plt.title('Digital filter frequency response')
        plt.plot(w, 20 * np.log10(np.abs(h)))
        plt.title('Digital filter frequency response')
        plt.ylabel('Amplitude Response [dB]')
        plt.xlabel('Frequency (rad/sample)')
        plt.grid()

    # apply the filter
    signal = lfilter(b, a, signal.copy())

    return signal


def time_dB(signal, Fs, bits=16):
    '''
    Compute the signed dB amplitude of the oscillating signal
    normalized wrt the number of bits used for the signal.
    '''

    import matplotlib.pyplot as plt

    # min dB (least significant bit in dB)
    lsb = -20 * np.log10(2.) * (bits - 1)

    # magnitude in dB (clipped)
    pos = clip(signal, 2. ** (bits - 1) - 1, 1.) / 2. ** (bits - 1)
    neg = -clip(signal, -1., -2. ** (bits - 1)) / 2. ** (bits - 1)

    mag_pos = np.zeros(signal.shape)
    Ip = np.where(pos > 0)
    mag_pos[Ip] = 20 * np.log10(pos[Ip]) + lsb + 1

    mag_neg = np.zeros(signal.shape)
    In = np.where(neg > 0)
    mag_neg[In] = 20 * np.log10(neg[In]) + lsb + 1

    plt.plot(np.arange(len(signal)) / float(Fs), mag_pos - mag_neg)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [dB]')
    plt.axis('tight')
    plt.ylim(lsb-1, -lsb+1)

    # draw ticks corresponding to decibels
    div = 20
    n = int(-lsb/div)+1
    yticks = np.zeros(2*n)
    yticks[:n] = lsb - 1 + np.arange(0, n*div, div)
    yticks[n:] = -lsb + 1 - np.arange((n-1)*div, -1, -div)
    yticklabels = np.zeros(2*n)
    yticklabels = range(0, -n*div, -div) + range(-(n-1)*div, 1, div)
    plt.setp(plt.gca(), 'yticks', yticks)
    plt.setp(plt.gca(), 'yticklabels', yticklabels)

    plt.setp(plt.getp(plt.gca(), 'ygridlines'), 'ls', '--')


def spectrum(signal, Fs, N):

    from .stft import stft, spectroplot
    from .windows import hann

    F = stft(signal, N, N / 2, win=hann(N))
    spectroplot(F.T, N, N / 2, Fs)


def dB(signal, power=False):
    if power is True:
        return 10*np.log10(np.abs(signal))
    else:
        return 20*np.log10(np.abs(signal))


def compare_plot(signal1, signal2, Fs, fft_size=512, norm=False, equal=False, title1=None, title2=None):

    import matplotlib.pyplot as plt

    td_amp = np.maximum(np.abs(signal1).max(), np.abs(signal2).max())

    if norm:
        if equal:
            signal1 /= np.abs(signal1).max()
            signal2 /= np.abs(signal2).max()
        else:
            signal1 /= td_amp
            signal2 /= td_amp
        td_amp = 1.

    plt.subplot(2, 2, 1)
    plt.plot(np.arange(len(signal1))/float(Fs), signal1)
    plt.axis('tight')
    plt.ylim(-td_amp, td_amp)
    if title1 is not None:
        plt.title(title1)

    plt.subplot(2, 2, 2)
    plt.plot(np.arange(len(signal2))/float(Fs), signal2)
    plt.axis('tight')
    plt.ylim(-td_amp, td_amp)
    if title2 is not None:
        plt.title(title2)

    from .stft import stft, spectroplot
    from .windows import hann

    F1 = stft.stft(signal1, fft_size, fft_size / 2, win=windows.hann(fft_size))
    F2 = stft.stft(signal2, fft_size, fft_size / 2, win=windows.hann(fft_size))

    # try a fancy way to set the scale to avoid having the spectrum
    # dominated by a few outliers
    p_min = 1
    p_max = 99.5
    all_vals = np.concatenate((dB(F1+eps), dB(F2+eps))).flatten()
    vmin, vmax = np.percentile(all_vals, [p_min, p_max])

    cmap = 'jet'
    interpolation = 'sinc'

    plt.subplot(2, 2, 3)
    stft.spectroplot(F1.T, fft_size, fft_size / 2, Fs, vmin=vmin, vmax=vmax,
                    cmap=plt.get_cmap(cmap), interpolation=interpolation)

    plt.subplot(2, 2, 4)
    stft.spectroplot(F2.T, fft_size, fft_size / 2, Fs, vmin=vmin, vmax=vmax,
            cmap=plt.get_cmap(cmap), interpolation=interpolation)


def real_spectrum(signal, axis=-1, **kwargs):

    import matplotlib.pyplot as plt

    S = np.fft.rfft(signal, axis=axis)
    f = np.arange(S.shape[axis])/float(2*S.shape[axis])

    plt.subplot(2,1,1)
    P = dB(S)
    plt.plot(f, P, **kwargs)

    plt.subplot(2,1,2)
    phi = np.unwrap(np.angle(S))
    plt.plot(f, phi, **kwargs)



def convmtx(x, n):
    '''
    Create a convolution matrix H for the vector x of size len(x) times n.
    Then, the result of np.dot(H,v) where v is a vector of length n is the same
    as np.convolve(x, v).
    '''

    import scipy as s

    c = np.concatenate((x, np.zeros(n-1)))
    r = np.zeros(n)

    return s.linalg.toeplitz(c, r)


def prony(x, p, q):
    '''
    Prony's Method from Monson H. Hayes' Statistical Signal Processing, p. 154

    Parameters
    ----------

    x: 
        signal to model
    p: 
        order of denominator
    q: 
        order of numerator

    Returns
    -------

    a: 
        numerator coefficients
    b: 
        denominator coefficients
    err: the squared error of approximation
    '''

    nx = x.shape[0]

    if (p+q >= nx):
        raise NameError('Model order too large')

    X = convmtx(x, p+1)

    Xq = X[q:nx+p-1, 0:p]

    a = np.concatenate((np.ones(1), -np.linalg.lstsq(Xq, X[q+1:nx+p, 0])[0]))
    b = np.dot(X[0:q+1, 0:p+1], a)

    err = np.inner(np.conj(x[q+1:nx]), np.dot(X[q+1:nx, :p+1], a))

    return a, b, err


def shanks(x, p, q):
    '''
    Shank's Method from Monson H. Hayes' Statistical Signal Processing, p. 154

    Parameters
    ----------
    x: 
        signal to model
    p: 
        order of denominator
    q: 
        order of numerator

    Returns
    -------
    a: 
        numerator coefficients
    b: 
        denominator coefficients
    err: 
        the squared error of approximation
    '''

    from scipy import signal

    nx = x.shape[0]

    if (p+q >= nx):
        raise NameError('Model order too large')

    a = prony(x, p, q)[0]

    u = np.zeros(nx)
    u[0] = 1.

    g = signal.lfilter(np.ones(1), a, u)

    G = convmtx(g, q+1)
    b = np.linalg.lstsq(G[:nx, :], x)[0]
    err = np.inner(np.conj(x), x) - np.inner(np.conj(x), np.dot(G[:nx, :q+1], b))

    return a, b, err


def low_pass_dirac(t0, alpha, Fs, N):
    '''
    Creates a vector containing a lowpass Dirac of duration T sampled at Fs
    with delay t0 and attenuation alpha.

    If t0 and alpha are 2D column vectors of the same size, then the function
    returns a matrix with each line corresponding to pair of t0/alpha values.
    '''

    return alpha*np.sinc(np.arange(N) - Fs*t0)


def fractional_delay(t0):
    '''
    Creates a fractional delay filter using a windowed sinc function.
    The length of the filter is fixed by the module wide constant
    `frac_delay_length` (default 81).

    Parameters
    ----------
    t0: float
        The delay in fraction of sample. Typically between 0 and 1.

    Returns
    -------
    A fractional delay filter with specified delay.
    '''

    N = constants.get('frac_delay_length')

    return np.hanning(N)*np.sinc(np.arange(N) - (N-1)/2 - t0)


def fractional_delay_filter_bank(delays):
    '''
    Creates a fractional delay filter bank of windowed sinc filters

    Parameters
    ----------
    delays: 1d narray
        The delays corresponding to each filter in fractional samples
        
    Returns
    -------
    An ndarray where the ith row contains the fractional delay filter
    corresponding to the ith delay. The number of columns of the matrix
    is proportional to the maximum delay.
    '''

    # constants and lengths
    N = delays.shape[0]
    L = constants.get('frac_delay_length')
    filter_length = L + int(np.ceil(delays).max())

    # subtract the minimum delay, so that all delays are positive
    delays -= delays.min()

    # allocate a flat array for the filter bank that we'll reshape at the end
    bank_flat = np.zeros(N * filter_length)

    # separate delays in integer and fractional parts
    di = np.floor(delays).astype(np.int)
    df = delays - di

    # broadcasting tricks to compute at once all the locations
    # and sinc times that must be computed
    T = np.tile(np.arange(L), (N, 1))
    indices = (T + (di[:,None] + filter_length * np.arange(N)[:,None]))
    sinc_times = (T - df[:,None] - (L - 1) / 2)

    # we'll need to window also all the sincs at once
    windows = np.tile(np.hanning(L), N)

    # compute all sinc with one call
    bank_flat[indices.ravel()] = windows * np.sinc(sinc_times.ravel())

    return np.reshape(bank_flat, (N, -1))



def levinson(r, b):
    '''
    levinson(r,b)

    Solve a system of the form Rx=b where R is hermitian toeplitz matrix and b
    is any vector using the generalized Levinson recursion as described in M.H.
    Hayes, Statistical Signal Processing and Modelling, p. 268.

    Parameters
    ----------
    r: 
        First column of R, toeplitz hermitian matrix.
    b: 
        The right-hand argument. If b is a matrix, the system is solved
        for every column vector in b.

    Returns
    -------
    The solution of the linear system Rx = b.
    '''

    p = b.shape[0]

    a = np.array([1])
    x = b[np.newaxis, 0, ]/r[0]
    epsilon = r[0]

    for j in np.arange(1, p):

        g = np.sum(np.conj(r[1:j+1])*a[::-1])
        gamma = -g/epsilon
        a = np.concatenate((a, np.zeros(1))) + gamma*np.concatenate((np.zeros(1), np.conj(a[::-1])))
        epsilon = epsilon*(1 - np.abs(gamma)**2)
        delta = np.dot(np.conj(r[1:j+1]), np.flipud(x))
        q = (b[j, ] - delta)/epsilon
        x = np.concatenate((x, np.zeros(1 if len(b.shape) == 1 else (1, b.shape[1]))), axis=0) + q*np.conj(a[::-1, np.newaxis])

    return x

def goertzel(x, k):
    ''' Goertzel algorithm to compute DFT coefficients '''

    N = x.shape[0]
    f = k/float(N)

    a = np.r_[1., -2.*np.cos(2.*np.pi*f), 1.]
    b = np.r_[1]
    s = signal.lfilter(b,a, x)
    y = np.exp(2j*np.pi*f)*s[-1] - s[-2]

    return y
    
    
'''
GEOMETRY UTILITIES
'''
def angle_from_points(x1, x2):
    return np.angle((x1[0, 0]-x2[0, 0]) + 1j*(x1[1, 0] - x2[1, 0]))

