
import numpy as np
import constants


def clip(signal, high, low):
    '''
    Clip a signal from above at high and from below at low.
    '''
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


def angle_from_points(x1, x2):

    return np.angle((x1[0,0]-x2[0,0]) + 1j*(x1[1,0] - x2[1,0]))


def normalize_pwr(sig1, sig2):
    '''
    normalize sig1 to have the same power as sig2
    '''

    # average power per sample
    p1 = np.mean(sig1 ** 2)
    p2 = np.mean(sig2 ** 2)

    # normalize
    return sig1.copy() * np.sqrt(p2 / p1)


def highpass(signal, Fs, fc=constants.fc_hp, plot=False):
    '''
    Filter out the really low frequencies, default is below 50Hz
    '''

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
    normalized wrt the number of bits used for the signal
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

    import stft
    import windows

    F = stft.stft(signal, N, N / 2, win=windows.hann(N))
    stft.spectroplot(F.T, N, N / 2, Fs)


def dB(signal, power=False):
    if power is True:
        return 10*np.log10(np.abs(signal))
    else:
        return 20*np.log10(np.abs(signal))


def comparePlot(signal1, signal2, Fs, fft_size=512, norm=False, equal=False, title1=None, title2=None):

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

    plt.subplot(2,2,1)
    plt.plot(np.arange(len(signal1))/float(Fs), signal1)
    plt.axis('tight')
    plt.ylim(-td_amp, td_amp)
    if title1 is not None:
        plt.title(title1)

    plt.subplot(2,2,2)
    plt.plot(np.arange(len(signal2))/float(Fs), signal2)
    plt.axis('tight')
    plt.ylim(-td_amp, td_amp)
    if title2 is not None:
        plt.title(title2)

    from constants import eps
    import stft
    import windows

    F1 = stft.stft(signal1, fft_size, fft_size / 2, win=windows.hann(fft_size))
    F2 = stft.stft(signal2, fft_size, fft_size / 2, win=windows.hann(fft_size))

    # try a fancy way to set the scale to avoid having the spectrum
    # dominated by a few outliers
    p_min = 1
    p_max = 99.5
    all_vals = np.concatenate((dB(F1+eps), dB(F2+eps))).flatten()
    vmin, vmax = np.percentile(all_vals, [p_min, p_max])

    cmap = 'jet'
    interpolation='sinc'

    plt.subplot(2,2,3)
    stft.spectroplot(F1.T, fft_size, fft_size / 2, Fs, vmin=vmin, vmax=vmax,
            cmap=plt.get_cmap(cmap), interpolation=interpolation)

    plt.subplot(2,2,4)
    stft.spectroplot(F2.T, fft_size, fft_size / 2, Fs, vmin=vmin, vmax=vmax, 
            cmap=plt.get_cmap(cmap), interpolation=interpolation)


def real_spectrum(signal, axis=-1, **kwargs):

    import matplotlib.pyplot as plt

    P = dB(np.fft.rfft(signal, axis=axis))
    f = np.arange(P.shape[axis])/float(2*P.shape[axis])

    plt.plot(f, P, **kwargs)


