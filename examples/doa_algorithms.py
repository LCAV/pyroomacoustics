# -*- coding: utf-8 -*-
'''
DOA Algorithms
==============

This example demonstrates how to use the DOA object to perform direction of arrival
finding in 2D using one of several algorithms

- MUSIC [1]_
- SRP-PHAT [2]_
- CSSM [3]_
- WAVES [4]_
- TOPS [5]_
- FRIDA [6]_

.. [1] R. Schmidt, *Multiple emitter location and signal parameter estimation*, 
    IEEE Trans. Antennas Propag., Vol. 34, Num. 3, pp 276--280, 1986

.. [2] J. H. DiBiase, J H, *A high-accuracy, low-latency technique for talker localization 
    in reverberant environments using microphone arrays*, PHD Thesis, Brown University, 2000

.. [3] H. Wang, M. Kaveh, *Coherent signal-subspace processing for the detection and 
    estimation of angles of arrival of multiple wide-band sources*, IEEE Trans. Acoust., 
    Speech, Signal Process., Vol. 33, Num. 4, pp 823--831, 1985

.. [4] E. D. di Claudio, R. Parisi, *WAVES: Weighted average of signal subspaces for 
    robust wideband direction finding*, IEEE Trans. Signal Process., Vol. 49, Num. 10, 
    2179--2191, 2001

.. [5] Y. Yeo-Sun, L. M. Kaplan, J. H. McClellan, *TOPS: New DOA estimator for wideband 
    signals*, IEEE Trans. Signal Process., Vol. 54, Num 6., pp 1977--1989, 2006

.. [6] H. Pan, R. Scheibler, E. Bezzam, I. Dokmanić, and M. Vetterli, *FRIDA:
    FRI-based DOA estimation for arbitrary array layouts*, Proc. ICASSP,
    pp 3186-3190, 2017

In this example, we generate some random signal for a source in the far field
and then simulate propagation using a fractional delay filter bank
corresponding to the relative microphone delays.

Then we perform DOA estimation and compare the errors for different algorithms

'''

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

import pyroomacoustics as pra
from pyroomacoustics.doa import circ_dist

######
# We define a meaningful distance measure on the circle

# Location of original source
azimuth = 61. / 180. * np.pi  # 60 degrees
distance = 3.  # 3 meters

#######################
# algorithms parameters
SNR = 0.    # signal-to-noise ratio
c = 343.    # speed of sound
fs = 16000  # sampling frequency
nfft = 256  # FFT size
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation

# compute the noise variance
sigma2 = 10**(-SNR / 10) / (4. * np.pi * distance)**2

# Create an anechoic room
room_dim = np.r_[10.,10.]
aroom = pra.ShoeBox(room_dim, fs=fs, max_order=0, sigma2_awgn=sigma2)

# add the source
source_location = room_dim / 2 + distance * np.r_[np.cos(azimuth), np.sin(azimuth)]
source_signal = np.random.randn((nfft // 2 + 1) * nfft)
aroom.add_source(source_location, signal=source_signal)

# We use a circular array with radius 15 cm # and 12 microphones
R = pra.circular_2D_array(room_dim / 2, 12, 0., 0.15)
aroom.add_microphone_array(pra.MicrophoneArray(R, fs=aroom.fs))

# run the simulation
aroom.simulate()

################################
# Compute the STFT frames needed
X = np.array([ 
    pra.stft(signal, nfft, nfft // 2, transform=np.fft.rfft).T 
    for signal in aroom.mic_array.signals ])

##############################################
# Now we can test all the algorithms available
algo_names = sorted(pra.doa.algorithms.keys())

for algo_name in algo_names:
    # Construct the new DOA object
    # the max_four parameter is necessary for FRIDA only
    doa = pra.doa.algorithms[algo_name](R, fs, nfft, c=c, max_four=4)

    # this call here perform localization on the frames in X
    doa.locate_sources(X, freq_bins=freq_bins)

    doa.polar_plt_dirac()
    plt.title(algo_name)
    
    # doa.azimuth_recon contains the reconstructed location of the source
    print(algo_name)
    print('  Recovered azimuth:', doa.azimuth_recon / np.pi * 180., 'degrees')
    print('  Error:', circ_dist(azimuth, doa.azimuth_recon) / np.pi * 180., 'degrees')

plt.show()
