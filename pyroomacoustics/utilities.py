# Utility functions for the package
# Copyright (C) 2019  Sidney Barthe, Robin Scheibler, Ivan Dokmanic, Eric Bezzam
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

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .parameters import constants, eps
from .sync import correlate


def create_noisy_signal(signal_fp, snr, noise_fp=None, offset=None):
    """
    Create a noisy signal of a specified SNR.
    Parameters
    ----------
    signal_fp : string
        File path to clean input.
    snr : float
        SNR in dB.
    noise_fp : string
        File path to noise. Default is to use randomly generated white noise.
    offset : float
        Offset in seconds before starting the signal.

    Returns
    -------
    numpy array
        Noisy signal with specified SNR, between [-1,1] and zero mean.
    numpy array
        Clean signal, untouched from WAV file.
    numpy array
        Added noise such that specified SNR is met.
    int
        Sampling rate in Hz.

    """
    fs, clean_signal = wavfile.read(signal_fp)
    clean_signal = to_float32(clean_signal)

    if offset is not None:
        offset_samp = int(offset * fs)
    else:
        offset_samp = 0
    output_len = len(clean_signal) + offset_samp

    if noise_fp is not None:
        fs_n, noise = wavfile.read(noise_fp)
        noise = to_float32(noise)
        if fs_n != fs:
            raise ValueError("Signal and noise WAV files should have same "
                             "sampling rate.")
        # truncate to same length
        if len(noise) < output_len:
            raise ValueError("Length of signal file should be longer than "
                             "noise file.")
        noise = noise[:output_len]
    else:
        if len(clean_signal.shape) > 1:  # multichannel
            noise = np.random.randn(output_len,
                                    clean_signal.shape[1]).astype(np.float32)
        else:
            noise = np.random.randn(output_len).astype(np.float32)
        noise = normalize(noise)

    # weight noise according to desired SNR
    signal_level = rms(clean_signal)
    noise_level = rms(noise[offset_samp:])
    noise_fact = signal_level / noise_level * 10 ** (-snr / 20)
    noise_weighted = noise * noise_fact

    # add signal and noise
    noisy_signal = clean_signal + noise_weighted

    # ensure between [-1, 1]
    norm_fact = np.abs(noisy_signal).max()
    clean_signal /= norm_fact
    noise_weighted /= norm_fact
    noisy_signal /= norm_fact

    # remove any offset
    noisy_signal -= noisy_signal.mean()

    return noisy_signal, clean_signal, noise_weighted, fs


def rms(data):
    """
    Compute root mean square of input.

    Parameters
    ----------
    data : numpy array
        Real signal in time domain.

    Returns
    -------
    float
        Root mean square.
    """
    return np.sqrt(np.mean(data * data))


def to_float32(data):
    """
    Cast data (typically from WAV) to float32.

    Parameters
    ----------
    data : numpy array
        Real signal in time domain, typically obtained from WAV file.

    Returns
    -------
    numpy array
        `data` as float32.
    """

    if np.issubdtype(data.dtype, np.integer):
        max_val = abs(np.iinfo(data.dtype).min)
        return data.astype(np.float32) / max_val
    else:
        return data.astype(np.float32)


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
        s *= 2 ** (bits - 1) - 1
        s = clip(s, 2 ** (bits - 1) - 1, -2 ** (bits - 1))

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
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings
            warnings.warn('Matplotlib is required for plotting')
            return

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

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import warnings
        warnings.warn('Matplotlib is required for plotting')
        return

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

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import warnings
        warnings.warn('Matplotlib is required for plotting')
        return

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

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import warnings
        warnings.warn('Matplotlib is required for plotting')
        return

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

    a = np.concatenate((np.ones(1), -np.linalg.lstsq(Xq, X[q+1:nx+p, 0], rcond=None)[0]))
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
    b = np.linalg.lstsq(G[:nx, :], x, rcond=None)[0]
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
        The delay in fraction of sample. Typically between -1 and 1.

    Returns
    -------
    numpy array
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
    numpy array
        An ndarray where the ith row contains the fractional delay filter
        corresponding to the ith delay. The number of columns of the matrix
        is proportional to the maximum delay.
    '''

    delays = np.array(delays)

    # subtract the minimum delay, so that all delays are positive
    delays -= delays.min()

    # constants and lengths
    N = delays.shape[0]
    L = constants.get('frac_delay_length')
    filter_length = L + int(np.ceil(delays).max())

    # allocate a flat array for the filter bank that we'll reshape at the end
    bank_flat = np.zeros(N * filter_length)

    # separate delays in integer and fractional parts
    di = np.floor(delays).astype(np.int)
    df = delays - di

    # broadcasting tricks to compute at once all the locations
    # and sinc times that must be computed
    T = np.arange(L)
    indices = (T[None,:] + (di[:,None] + filter_length * np.arange(N)[:,None]))
    sinc_times = (T - df[:,None] - (L - 1) / 2)

    # we'll need to window also all the sincs at once
    windows = np.tile(np.hanning(L), N)

    # compute all sinc with one call
    bank_flat[indices.ravel()] = windows * np.sinc(sinc_times.ravel())

    return np.reshape(bank_flat, (N, -1))


def levinson(r, b):
    '''

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
    numpy array
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
        if len(b.shape) == 1:
            x = np.concatenate((x, np.zeros(1))) + q * np.conj(a[::-1])
        else:
            x = np.concatenate((x, np.zeros((1, b.shape[1]))), axis=0) + q * np.conj(a[::-1, np.newaxis])

    return x


def autocorr(x, p, biased=True, method='numpy'):
    """
    Compute the autocorrelation for real signal `x` up to lag `p`.

    Parameters
    ----------
    x : numpy array
        Real signal in time domain.
    p : int
        Amount of lag. When solving for LPC coefficient, this is typically the
        LPC order.
    biased : bool
        Whether to return biased autocorrelation (default) or unbiased. As
        there are fewer samples for larger lags, the biased estimate tends to
        have better statistical stability under noise.
    method : 'numpy, 'fft', 'time', `pra`
        Method for computing the autocorrelation: in the frequency domain with
        `fft` or `pra` (`np.fft.rfft` is used so only real signals are
        supported), in the time domain with `time`, or with `numpy`'s built-in
        function `np.correlate` (default). For `p < log2(len(x))`, the time
        domain approach may be more efficient.

    Returns
    -------
    numpy array
        Autocorrelation for `x` up to lag `p`.
    """
    L = len(x)
    if method is "fft":
        X = np.fft.rfft(np.r_[x, np.zeros_like(x)])
        r = np.fft.irfft(X * np.conj(X))[:p + 1]
    elif method is "time":
        r = np.array([np.dot(x[:L - m], x[m:]) for m in range(p + 1)])
    elif method is "numpy":
        r = np.correlate(x, x, 'full')[L - 1:L + p]
    elif method is "pra":
        r = correlate(x, x)[L - 1:L + p]
    else:
        raise ValueError("Invalid `method` for computing autocorrelation"
                         "choose one of: `fft`, `time`, `numpy`, `pra`.")

    if biased:
        return r / L
    else:
        return r / np.arange(L, L - p - 1, step=-1)


def lpc(x, p, biased=True):
    """
    Compute `p` LPC coefficients for a speech segment `x`.

    Parameters
    ----------
    x : numpy array
        Real signal in time domain.
    p : int
        Amount of lag. When solving for LPC coefficient, this is typically the
        LPC order.
    biased : bool
        Whether to use biased autocorrelation (default) or unbiased. As there
        are fewer samples for larger lags, the biased estimate tends to have
        better statistical stability under noise.

    Returns
    -------
    numpy array
        `p` LPC coefficients.
    """
    # compute autocorrelation
    r = autocorr(x, p, biased)

    # solve Yule-Walker equations for LPC coefficients
    return levinson(r[:p], r[1:])


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

