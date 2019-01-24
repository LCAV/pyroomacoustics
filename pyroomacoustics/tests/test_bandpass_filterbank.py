'''
This tests the construction of a bank of octave filters
'''
import pyroomacoustics as pra

import numpy as np
from scipy.signal import sosfreqz

tol = 1.  # decibel

def test_bandpass_filterbank():
    f0 = 125. / 2.
    fs = 16000

    bands, fc = pra.octave_bands(f0)
    filters = pra.bandpass_filterbank(bands, fs=fs, order=16)

    res = None
    for sos, f in zip(filters, fc):
        w, h = sosfreqz(sos, worN=4096, whole=False)

        # inspect the cumulative response
        if res is None:
            res = np.abs(h) ** 2
        else:
            res += np.abs(h) ** 2

    # check automatically that the cumulative response is (nearly) flat
    freq = w / np.pi * fs / 2
    I = np.where(np.logical_and(freq > fc[0], freq < np.minimum(fc[-1], 0.95 * fs / 2)))[0]
    err = np.max(np.abs(10 * np.log10(res[I])))
    
    assert err <= tol, 'Bandpass filterbank test fails (err {} > tol {})'.format(err, tol)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    f0 = 125. / 2.
    fs = 16000

    bands, fc = pra.octave_bands(f0)
    filters = pra.bandpass_filterbank(bands, fs=fs, order=16)

    plt.figure()
    res = None
    for sos, f in zip(filters, fc):
        w, h = sosfreqz(sos, worN=4096, whole=False)
        plt.semilogx(w / np.pi * fs / 2., 20 * np.log10(np.abs(h)), label=str(f) + ' Hz')

        # inspect the cumulative response
        if res is None:
            res = np.abs(h) ** 2
        else:
            res += np.abs(h) ** 2

    # check automatically that the cumulative response is (nearly) flat
    freq = w / np.pi * fs / 2
    I = np.where(np.logical_and(freq > fc[0], freq < np.minimum(fc[-1], 0.95 * fs / 2)))[0]
    err = np.max(np.abs(10 * np.log10(res[I])))
    print('The error is', err, '(tol is', tol, 'dB)')

    plt.semilogx(w / np.pi * fs / 2., 10 * np.log10(np.abs(res)), label='Sum')

    plt.title('Octave Filter Bank')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green') # cutoff frequency
    plt.ylim(-60,3)
    plt.xlim(10., fs / 2.)
    plt.legend()
    plt.show()

    # run the automatic tests
    test_bandpass_filterbank()

