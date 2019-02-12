"""
Single Channel Noise Reduction with Spectral Subtraction
========================================================

In this example, we apply a simple Single Channel Noise Reduction (SCNR)
algorithm in the STFT domain. For a given block, the SNR of each frequency bin
is estimated in order to determine a gain filter that is applied to the given
block in order to suppress noisy bins.

This simple approach is suitable for scenarios with noise that is rather
stationary and where the SNR is positive.

With a large suppression, i.e. large values for `db_reduc`, we can observe a
typical artefact of such spectral subtraction approaches, namely "musical
noise". `Here <https://www.vocal.com/noise-reduction/musical-noise/>`_ is nice
article about noise reduction and musical noise:

This implementation shows how the approach can be applied to streaming/online
data. For fixed WAV files, the one-shot function
`pyroomacoustics.denoise.apply_spectral_sub` can be used.
"""

import numpy as np
from scipy.io import wavfile
import os
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from pyroomacoustics.denoise import SpectralSub

"""
Test and algorithm parameters
"""
snr = 5         # SNR of input signal.
db_reduc = 10   # Maximum suppression per frequency bin. Large suppresion can result in more musical noise.
nfft = 512      # Frame length will be nfft/2 as we will use an STFT with 50% overlap.
lookback = 12   # How many frames to look back for the noise floor estimate.
beta = 3        # An overestimation factor to "push" the suppression towards db_reduc.
alpha = 1.2     # An exponential factor to tune the suppresion (see documentation of 'SpectralSub').

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
                           'denoise_input_SpectralSub.wav'), fs,
              noisy_signal.astype(np.float32))

"""
Create STFT and SCNR objects
"""
hop = nfft // 2
window = pra.hann(nfft, flag='asymmetric', length='full')
stft = pra.transform.STFT(nfft, hop=hop, analysis_window=window,
                          streaming=True)

scnr = SpectralSub(nfft, db_reduc, lookback, beta, alpha)
lookback_time = hop/fs * lookback
print("Lookback : %f seconds" % lookback_time)

"""
Process as in real-time
"""
# collect the processed blocks
processed_audio = np.zeros(signal.shape)
n = 0
while noisy_signal.shape[0] - n >= hop:

    # SCNR in frequency domain
    stft.analysis(noisy_signal[n:(n+hop), ])
    gain_filt = scnr.compute_gain_filter(stft.X)

    # back to time domain
    processed_audio[n:n+hop, ] = stft.synthesis(gain_filt*stft.X)

    # update step
    n += hop


"""
Save and plot spectrogram
"""
wavfile.write(os.path.join(os.path.dirname(__file__),
                           'output_samples',
                           'denoise_output_SpectralSub.wav'), fs,
              pra.normalize(processed_audio).astype(np.float32))
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
