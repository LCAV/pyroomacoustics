import numpy as np
from scipy.io import wavfile
import os
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import time
from pyroomacoustics.denoise import IterativeWiener


"""
Test and algorithm parameters
"""
snr = 5         # SNR of input signal

# the number of LPC coefficients to consider
lpc_order = 15
# the number of iterations to update wiener filter
iterations = 2
# FFT length
frame_len = 512
# parameter update of the sigma in sigma tracking
alpha = 0.1   # smaller value allows noise floor to change faster
threshold = 0.003

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
                           'denoise_input_IterativeWiener.wav'), fs,
              noisy_signal.astype(np.float32))

"""
Apply approach
"""


scnr = IterativeWiener(frame_len, lpc_order, iterations, alpha, threshold)

# derived parameters
hop = frame_len // 2
win = pra.hann(frame_len, flag='asymmetric', length='full')
stft = pra.transform.STFT(frame_len, hop=hop,
                          analysis_window=win,
                          streaming=True)
speech_psd = np.ones(hop+1)   # initialize PSD
noise_psd = 0

start_time = time.time()
processed_audio = np.zeros(noisy_signal.shape)
n = 0
while noisy_signal.shape[0] - n >= hop:

    # to frequency domain, 50% overlap
    stft.analysis(noisy_signal[n:(n + hop), ])

    # compute Wiener output
    X = scnr.compute_filtered_output(current_frame=stft.fft_in_buffer,
                                     frame_dft=stft.X)

    # back to time domain
    processed_audio[n:n + hop, ] = stft.synthesis(X)

    # update step
    n += hop

proc_time = time.time() - start_time
print("Processing time: {} minutes".format(proc_time/60))

"""
Save and plot spectrogram
"""
wavfile.write(os.path.join(os.path.dirname(__file__),
                           'output_samples',
                           'denoise_output_IterativeWiener.wav'), fs,
              pra.normalize(processed_audio).astype(np.float32))
print("Noisy and denoised file written to: '%s'" %
      os.path.join(os.path.dirname(__file__), 'output_samples'))

signal_norm = signal / np.abs(signal).max()
processed_audio_norm = processed_audio / np.abs(processed_audio).max()

if plot_spec:
    min_val = -80
    max_val = -40
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.specgram(noisy_signal[:n-hop], NFFT=256, Fs=fs,
                 vmin=min_val, vmax=max_val)
    plt.title('Noisy Signal')
    plt.subplot(3, 1, 2)
    plt.specgram(processed_audio_norm[hop:n], NFFT=256, Fs=fs,
                 vmin=min_val, vmax=max_val)
    plt.title('Denoised Signal')
    plt.subplot(3, 1, 3)
    plt.specgram(signal_norm[:n-hop], NFFT=256, Fs=fs,
                 vmin=min_val, vmax=max_val)
    plt.title('Original Signal')
    plt.tight_layout(pad=0.5)
    plt.show()

