"""
Single Channel Noise Reduction via Subspace Decomposition
========================================================

In this example, we apply a Single Channel Noise Reduction (SCNR) algorithm
in the time domain with the subspace approach.

This implementation shows how the approach can be applied to streaming/online
data. For fixed WAV files, the one-shot function
`pyroomacoustics.denoise.apply_subspace` can be used.
"""

import numpy as np
from scipy.io import wavfile
import os
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import time
from pyroomacoustics.denoise import Subspace

"""
Test and algorithm parameters
"""
snr = 5         # SNR of input signal.
frame_len = 80
mu = 10         # higher value can give more suppression but more distortion

# parameters for covariance matrix estimation
lookback = 10       # how many frames to look back
skip = 1            # how many samples to skip when estimating
threshold = 0.003    # threshold between (signal+noise) and noise

plot_spec = True

"""
Prepare input file
"""
signal_fp = os.path.join(os.path.dirname(__file__), 'input_samples',
                         'cmu_arctic_us_aew_a0001.wav')
noise_fp = os.path.join(os.path.dirname(__file__), 'input_samples',
                        'doing_the_dishes.wav')
noisy_signal, signal, noise, fs = pra.create_noisy_signal(signal_fp,
                                                          snr=snr,
                                                          noise_fp=noise_fp)
wavfile.write(os.path.join(os.path.dirname(__file__), 'output_samples',
                           'denoise_input_Subspace.wav'), fs,
              noisy_signal.astype(np.float32))

""" 
Create noise reduction object and apply the method
"""
scnr = Subspace(frame_len, mu, lookback, skip, threshold)

# parse signal as if streaming
processed_audio = np.zeros(noisy_signal.shape)
n = 0
start_time = time.time()
hop = frame_len // 2
while noisy_signal.shape[0] - n >= hop:

    processed_audio[n:n + hop, ] = scnr.apply(noisy_signal[n:n + hop])

    # update step
    n += hop

proc_time = time.time() - start_time
print("{} minutes".format((proc_time/60)))
# save to output file
enhanced_signal_fp = os.path.join(os.path.dirname(__file__), 'output_samples',
                                  'denoise_output_Subspace.wav')
wavfile.write(enhanced_signal_fp, fs,
              pra.normalize(processed_audio).astype(np.float32))


"""
Plot spectrogram
"""
print("Noisy and denoised file written to: '%s'" %
      os.path.join(os.path.dirname(__file__), 'output_samples'))

signal_norm = signal / np.abs(signal).max()

if plot_spec:
    min_val = -80
    max_val = -40
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.specgram(noisy_signal[:n-hop], NFFT=256, Fs=fs,
                 vmin=min_val, vmax=max_val)
    plt.title('Noisy Signal')
    plt.subplot(3, 1, 2)
    plt.specgram(processed_audio[hop:n], NFFT=256, Fs=fs,
                 vmin=min_val, vmax=max_val)
    plt.title('Denoised Signal')
    plt.subplot(3, 1, 3)
    plt.specgram(signal_norm[:n-hop], NFFT=256, Fs=fs,
                 vmin=min_val, vmax=max_val)
    plt.title('Original Signal')
    plt.tight_layout(pad=0.5)
    plt.show()
