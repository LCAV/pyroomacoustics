# @version: 1.0  date: 2017/06/20 by Robin Scheibler
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2017

from unittest import TestCase
import numpy as np
from scipy.signal import fftconvolve

import pyroomacoustics as pra

def circ_dist(a1, a2):
    ''' distance on the circle '''
    return np.min(np.mod(np.abs([a1 - a2, 2 * np.pi + a2 - a1]), 2 * np.pi))

# fix the RNG seed for repeatability
np.random.seed(0)

# Location of original source
azimuth = 61. / 180. * np.pi  # 60 degrees
tol = 0.3 / 180. * np.pi  # 0.3 degrees tolerance for the test

# algorithms parameters
c = 343.
fs = 16000
nfft = 256
freq_bins = np.arange(5, 60)

# circular microphone array, 6 mics, radius 15 cm
R = pra.circular_2D_array([0, 0], 12, 0., 0.15)

# propagation filter bank
propagation_vector = -np.array([np.cos(azimuth), np.sin(azimuth)])
delays = np.dot(R.T, propagation_vector) / c * fs  # in fractional samples
filter_bank = pra.fractional_delay_filter_bank(delays)

# we use a white noise signal for the source
x = np.random.randn((nfft // 2 + 1) * nfft)

# convolve the source signal with the fractional delay filters
# to get the microphone input signals
mic_signals = [ fftconvolve(x, filter, mode='same') for filter in filter_bank ]
X = np.array([ pra.stft(signal, nfft, nfft // 2, win=np.hanning(nfft), transform=np.fft.rfft).T for signal in mic_signals ])

# Now we can test all the algorithms available
algo_names = sorted(pra.doa.algos.keys())
for algo_name in algo_names:
    # the max_four parameter is necessary for FRIDA only
    doa = pra.doa.algos[algo_name](R, fs, nfft, c=c, max_four=4)
    doa.locate_sources(X, freq_bins=freq_bins)
    print(algo_name)
    print('  Recovered azimuth:', doa.azimuth_recon / np.pi * 180., 'degrees')
    print('  Error:', circ_dist(azimuth, doa.azimuth_recon) / np.pi * 180., 'degrees')
