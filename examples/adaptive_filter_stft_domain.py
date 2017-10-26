'''
Adaptive Filter in STFT Domain Example
======================================

In this example, we will run adaptive filters for system 
identification, but in the frequeny domain.
'''
from __future__ import division, print_function

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# parameters
num_taps = 6       # number of taps in frequency domain
n_samples = 40000   # the number of samples to run
fft_length = 128   # block size
SNR = 40           # signal to noise ratio (dB)
fs = 16000

"""
Length of filter in time domain = <fft_size> / <samp_freq> * <num_taps>
"""

# the unknown filters in the frequency domain
num_bands = fft_length//2+1
W = np.random.randn(num_taps,num_bands) + \
    1j*np.random.randn(num_taps,num_bands)
W /= np.linalg.norm(W, axis=0)

# STFT blocks
window = pra.hann(fft_length)  # the analysis window
hop = fft_length//2
stft_in = pra.realtime.STFT(fft_length, hop=hop, 
    analysis_window=window, channels=1)
stft_out = pra.realtime.STFT(fft_length, hop=hop, 
    analysis_window=window, channels=1)
num_blocks = n_samples//hop

# create a known driving signal (reference) in time and frequency
x = np.random.randn(n_samples)

samp = 0
X_concat = np.zeros((num_bands,num_blocks),dtype=np.complex64)
for n in range(num_blocks):

    stft_in.analysis(x[samp:samp+hop,])
    X_concat[:,n] = stft_in.X

    samp += hop
stft_in.reset()

# convolve in frequency domain with unknown filter
Y_concat = np.zeros((num_bands,num_blocks), dtype=np.complex64)
for k in range(num_bands):
    Y_concat[k,:] = fftconvolve(X_concat[k,:], W[:,k])[:num_blocks]

# get time domain version
y_true = np.zeros(n_samples)
samp = 0
for n in range(num_blocks):

    y_true[samp:samp+hop,] = stft_out.synthesis(Y_concat[:,n])
    samp += hop
stft_out.reset()

# create noise
v = np.random.randn(n_samples) * 10**(-SNR / 20.)

samp = 0
V_concat = np.zeros((num_bands,num_blocks),dtype=np.complex64)
for n in range(num_blocks):

    stft_in.analysis(v[samp:samp+hop,])
    V_concat[:,n] = stft_in.X

    samp += hop
stft_in.reset()

# add noise
D_concat = Y_concat + V_concat

# V = np.random.randn(num_bands,num_blocks) + 
#     1j*np.random.randn(num_bands,num_blocks)
# V /= np.linalg.norm(V, axis=0) * np.linalg.norm(Y_concat, axis=0)
# V *= 10**(-SNR / 20.)
# D_concat = Y_concat + V.astype(np.complex64)


# apply subband LMS
adaptive_filters = pra.adaptive.SubbandLMS(num_taps=num_taps, 
    num_bands=num_bands, mu=0.5, nlms=True)

y_hat = np.zeros(n_samples)
aec_out_time = np.zeros(n_samples)
error_per_band = np.zeros((num_bands,num_blocks), dtype=np.float32)

samp = 0
stft_out.reset()
for n in range(num_blocks):

    # update filter with new samples
    adaptive_filters.update(X_concat[:,n], D_concat[:,n])
    error_per_band[:,n] = np.linalg.norm(adaptive_filters.W.conj() - W, axis=0)

    # back to time domain
    y_hat[samp:samp+hop,] = stft_in.synthesis(
        np.diag(np.dot(adaptive_filters.W.conj().T,
        adaptive_filters.X)))
    aec_out_freq = D_concat[:,n]-np.diag(np.dot(adaptive_filters.W.conj().T,adaptive_filters.X))
    aec_out_time[samp:samp+hop,] = stft_out.synthesis(aec_out_freq)

    samp += hop

# visualization and debug
plt.figure()
time_scale = np.arange(num_blocks)*hop/fs
for k in range(num_bands):
    plt.semilogy(time_scale, error_per_band[k,:])
plt.title('Convergence to unknown filter (per band)')
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Time [s]')
plt.ylabel('Filter error')
plt.show()

# # visualize in time domain
# time_scale = np.arange(n_samples,dtype=np.float32)/fs
# plt.figure()
# plt.plot(time_scale, x)
# plt.plot(time_scale, v)
# plt.plot(time_scale, aec_out_time)
# plt.autoscale(enable=True, axis='x', tight=True)
# plt.xlabel('Time [s]')
# plt.show()

# # visualize in time domain
# plt.figure()
# plt.plot(time_scale, y_true)
# plt.plot(time_scale, y_hat)
# plt.autoscale(enable=True, axis='x', tight=True)
# plt.xlabel('Time [s]')
# plt.show()

