# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from sparseauxiva import sparseauxiva
import sounddevice as sd


# Blind Source Separation techniques such as Independent Vector Analysis (IVA)
# using an Auxiliary function are implemented in ´pyroomacoustics´. IVA based
# algorithms work when the number of microphones is the same as the number of
# sources, i.e., the determinant case. Through this example, we will deal with
# the case of 2 sources and 2 microphones.

# First, open and concatanate wav files from the CMU dataset.
# concatanate audio samples to make them look long enough
wav_files = [
    ['../../../examples/input_samples/cmu_arctic_us_axb_a0004.wav',
     '../../../examples/input_samples/cmu_arctic_us_axb_a0005.wav',
     '../../../examples/input_samples/cmu_arctic_us_axb_a0006.wav', ],
    ['../../../examples/input_samples/cmu_arctic_us_aew_a0001.wav',
     '../../../examples/input_samples/cmu_arctic_us_aew_a0002.wav',
     '../../../examples/input_samples/cmu_arctic_us_aew_a0003.wav', ]
]

fs = 16000

signals = [np.concatenate([wavfile.read(f)[1].astype(np.float32)

                           for f in source_files])

           for source_files in wav_files]

# Define an anechoic room envrionment, as well as the microphone array and source locations.

# Room 4m by 6m
room_dim = [8, 9]
# source locations and delays
locations = [[2.5, 3], [2.5, 6]]
delays = [1., 0.]
# create an anechoic room with sources and mics
room = pra.ShoeBox(room_dim, fs=16000, max_order=15, absorption=0.35, sigma2_awgn=1e-8)

# add mic and good source to room
# Add silent signals to all sources
for sig, d, loc in zip(signals, delays, locations):
    room.add_source(loc, signal=np.zeros_like(sig), delay=d)

# add microphone array

room.add_microphone_array(pra.MicrophoneArray(np.c_[[6.5, 4.49], [6.5, 4.51]], room.fs))

# Compute the RIRs as in the Room Impulse Response generation section.

# compute RIRs
room.compute_rir()

# Mix the microphone recordings to simulate the observed signals by the microphone array in the frequency domain. To that end, we apply the STFT transform as explained in STFT.
from mir_eval.separation import bss_eval_images

# Record each source separately

separate_recordings = []
for source, signal in zip(room.sources, signals):
    source.signal[:] = signal
    room.simulate()
    separate_recordings.append(room.mic_array.signals)
    source.signal[:] = 0.
separate_recordings = np.array(separate_recordings)

# Mix down the recorded signals
mics_signals = np.sum(separate_recordings, axis=0)

print(mics_signals.shape)
wavfile.write('mix1.wav',fs,mics_signals.T)

sd.play(mics_signals.T[:,0], fs)

# STFT frame length
L = 2048

# Observation vector in the STFT domain
X = np.array([pra.stft(ch, L, L, transform=np.fft.rfft, zp_front=L // 2, zp_back=L // 2) for ch in mics_signals])
X = np.moveaxis(X, 0, 2)

# Reference signal to calculate performance of BSS

ref = np.moveaxis(separate_recordings, 1, 2)
SDR, SIR = [], []
ratio = 0.8
average = np.abs(np.mean(np.mean(X, axis=2), axis=0))
k = np.int_(average.shape[0] * ratio)
S = np.argpartition(average, -k)[-k:]
S = np.sort(S)
mu = 0
n_iter = 20

# Run SparseAuxIva
Y = sparseauxiva(X, S, mu, n_iter)

# run iSTFT
y = np.array([pra.istft(Y[:,:,ch], L, L, transform=np.fft.irfft, zp_front=L//2, zp_back=L//2) for ch in range(Y.shape[2])])

# Compare SIR and SDR with our reference signal
sdr, isr, sir, sar, perm = bss_eval_images(ref[:,:y.shape[1]-L//2,0], y[:,L//2:ref.shape[1]+L//2])
sd.play(y[0], fs)

mydata = sd.rec(int(y[0]),fs, blocking=True)
sf.write('demix0.wav', mydata, fs)


mydata = sd.rec(int(y[1]),fs, blocking=True)
sf.write('demix1.wav', mydata, fs)
