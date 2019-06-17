'''
This is a longer example that applies time domain beamforming towards a source
of interest in the presence of a strong interfering source.
'''

from __future__ import division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

import pyroomacoustics as pra

# Spectrogram figure properties
figsize=(15, 7)        # figure size
fft_size = 512         # fft size for analysis
fft_hop  = 8           # hop between analysis frame
fft_zp = 512           # zero padding
analysis_window = np.concatenate((pra.hann(fft_size), np.zeros(fft_zp)))
t_cut = 0.83           # length in [s] to remove at end of signal (no sound)

# Some simulation parameters
Fs = 8000
absorption = 0.1
max_order_sim = 2
sigma2_n = 5e-7

# Microphone array design parameters
mic1 = np.array([2, 1.5])   # position
M = 8                       # number of microphones
d = 0.08                    # distance between microphones
phi = 0.                    # angle from horizontal
max_order_design = 1        # maximum image generation used in design
shape = 'Linear'            # array shape
Lg_t = 0.100                # Filter size in seconds
Lg = np.ceil(Lg_t*Fs)       # Filter size in samples
delay = 0.050               # Beamformer delay in seconds

# Define the FFT length
N = 1024

# Create a microphone array
if shape is 'Circular':
    R = pra.circular_2D_array(mic1, M, phi, d*M/(2*np.pi)) 
else:
    R = pra.linear_2D_array(mic1, M, phi, d) 

# path to samples
path = os.path.dirname(__file__)

# The first signal (of interest) is singing
rate1, signal1 = wavfile.read(path + '/input_samples/singing_'+str(Fs)+'.wav')
signal1 = np.array(signal1, dtype=float)
signal1 = pra.normalize(signal1)
signal1 = pra.highpass(signal1, Fs)
delay1 = 0.

# The second signal (interferer) is some german speech
rate2, signal2 = wavfile.read(path + '/input_samples/german_speech_'+str(Fs)+'.wav')
signal2 = np.array(signal2, dtype=float)
signal2 = pra.normalize(signal2)
signal2 = pra.highpass(signal2, Fs)
delay2 = 1.

# Create the room
room_dim = [4, 6]
room1 = pra.ShoeBox(
    room_dim,
    absorption=absorption,
    fs=Fs,
    max_order=max_order_sim,
    sigma2_awgn=sigma2_n)

# Add sources to room
good_source = np.array([1, 4.5])           # good source
normal_interferer = np.array([2.8, 4.3])   # interferer
room1.add_source(good_source, signal=signal1, delay=delay1)
room1.add_source(normal_interferer, signal=signal2, delay=delay2)

'''
MVDR direct path only simulation
'''

# compute beamforming filters
mics = pra.Beamformer(R, Fs, N=N, Lg=Lg)
room1.add_microphone_array(mics)
room1.compute_rir()
room1.simulate()
mics.rake_mvdr_filters(room1.sources[0][0:1],
                    room1.sources[1][0:1],
                    sigma2_n*np.eye(mics.Lg*mics.M), delay=delay)

# process the signal
output = mics.process()

# save to output file
input_mic = pra.normalize(pra.highpass(mics.signals[mics.M//2], Fs))
wavfile.write(path + '/output_samples/input.wav', Fs, input_mic)

out_DirectMVDR = pra.normalize(pra.highpass(output, Fs))
wavfile.write(path + '/output_samples/output_DirectMVDR.wav', Fs, out_DirectMVDR)


'''
Rake MVDR simulation
'''

# Add the microphone array and compute RIR
mics = pra.Beamformer(R, Fs, N, Lg=Lg)
room1.add_microphone_array(mics)
room1.compute_rir()
room1.simulate()

# Design the beamforming filters using some of the images sources
good_sources = room1.sources[0][:max_order_design+1]
bad_sources = room1.sources[1][:max_order_design+1]
mics.rake_mvdr_filters(good_sources, 
                    bad_sources, 
                    sigma2_n*np.eye(mics.Lg*mics.M), delay=delay)

# process the signal
output = mics.process()

# save to output file
out_RakeMVDR = pra.normalize(pra.highpass(output, Fs))
wavfile.write(path + '/output_samples/output_RakeMVDR.wav', Fs, out_RakeMVDR)

'''
Perceptual direct path only simulation
'''

# compute beamforming filters
mics = pra.Beamformer(R, Fs, N, Lg=Lg)
room1.add_microphone_array(mics)
room1.compute_rir()
room1.simulate()
mics.rake_perceptual_filters(room1.sources[0][0:1],
        room1.sources[1][0:1],
                    sigma2_n*np.eye(mics.Lg*mics.M), delay=delay)

# process the signal
output = mics.process()

# save to output file
out_DirectPerceptual = pra.normalize(pra.highpass(output, Fs))
wavfile.write(path + '/output_samples/output_DirectPerceptual.wav', Fs, out_DirectPerceptual)

'''
Rake Perceptual simulation
'''

# compute beamforming filters
mics = pra.Beamformer(R, Fs, N, Lg=Lg)
room1.add_microphone_array(mics)
room1.compute_rir()
room1.simulate()
mics.rake_perceptual_filters(good_sources, 
                    bad_sources, 
                    sigma2_n*np.eye(mics.Lg*mics.M), delay=delay)

# process the signal
output = mics.process()

# save to output file
out_RakePerceptual = pra.normalize(pra.highpass(output, Fs))
wavfile.write(path + '/output_samples/output_RakePerceptual.wav', Fs, out_RakePerceptual)

'''
Plot all the spectrogram
'''

dSNR = pra.dB(room1.direct_snr(mics.center[:,0], source=0), power=True)
print('The direct SNR for good source is ' + str(dSNR))

# remove a bit of signal at the end
n_lim = int(np.ceil(len(input_mic) - t_cut*Fs))
input_clean = signal1[:n_lim]
input_mic = input_mic[:n_lim]
out_DirectMVDR = out_DirectMVDR[:n_lim]
out_RakeMVDR = out_RakeMVDR[:n_lim]
out_DirectPerceptual = out_DirectPerceptual[:n_lim]
out_RakePerceptual = out_RakePerceptual[:n_lim]


# compute time-frequency planes
F0 = pra.stft(input_clean, fft_size, fft_hop, 
          win=analysis_window, 
          zp_back=fft_zp)
F1 = pra.stft(input_mic, fft_size, fft_hop, 
          win=analysis_window, 
          zp_back=fft_zp)
F2 = pra.stft(out_DirectMVDR, fft_size, fft_hop, 
          win=analysis_window, 
          zp_back=fft_zp)
F3 = pra.stft(out_RakeMVDR, fft_size, fft_hop, 
          win=analysis_window, 
          zp_back=fft_zp)
F4 = pra.stft(out_DirectPerceptual, fft_size, fft_hop, 
          win=analysis_window, 
          zp_back=fft_zp)
F5 = pra.stft(out_RakePerceptual, fft_size, fft_hop, 
          win=analysis_window, 
          zp_back=fft_zp)

# (not so) fancy way to set the scale to avoid having the spectrum
# dominated by a few outliers
p_min = 7
p_max = 100
all_vals = np.concatenate((pra.dB(F1+pra.eps), 
                           pra.dB(F2+pra.eps), 
                           pra.dB(F3+pra.eps),
                           pra.dB(F0+pra.eps),
                           pra.dB(F4+pra.eps),
                           pra.dB(F5+pra.eps))).flatten()
vmin, vmax = np.percentile(all_vals, [p_min, p_max])

cmap = 'afmhot'
interpolation='none'

fig, ax = plt.subplots(figsize=figsize, nrows=2, ncols=3)

def plot_spectrogram(F, title):
    pra.spectroplot(F.T, fft_size+fft_zp, fft_hop, Fs, vmin=vmin, vmax=vmax,
            cmap=plt.get_cmap(cmap), interpolation=interpolation, colorbar=False)
    ax.set_title(title)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_aspect('auto')
    ax.axis('off')

ax = plt.subplot(2,3,1)
plot_spectrogram(F0, 'Desired Signal')

ax = plt.subplot(2,3,4)
plot_spectrogram(F1, 'Microphone Input')

ax = plt.subplot(2,3,2)
plot_spectrogram(F2, 'Direct MVDR')

ax = plt.subplot(2,3,5)
plot_spectrogram(F3, 'Rake MVDR')

ax = plt.subplot(2,3,3)
plot_spectrogram(F4, 'Direct Perceptual')

ax = plt.subplot(2,3,6)
plot_spectrogram(F5, 'Rake Perceptual')

fig.savefig(path + '/figures/spectrograms.png', dpi=150)

plt.show()
