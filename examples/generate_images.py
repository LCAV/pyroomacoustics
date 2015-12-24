
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.io import wavfile
from scipy.signal import resample,fftconvolve

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
t0 = 1./(Fs*np.pi*1e-2)  # starting time function of sinc decay in RIR response
absorption = 0.90
max_order_sim = 2
sigma2_n = 5e-7

# Room 1 : Shoe box
room_dim = [4, 6]

# the good source is fixed for all 
good_source = np.array([1, 4.5])           # good source
normal_interferer = np.array([2.8, 4.3])   # interferer
hard_interferer = np.array([1.5, 3])       # interferer in direct path
#normal_interferer = hard_interferer

# create the room with sources and mics
room1 = pra.Room.shoeBox2D(
    [0,0],
    room_dim,
    fs=Fs,
    t0 = t0,
    max_order=max_order_sim,
    absorption=absorption,
    sigma2_awgn=sigma2_n)

# add mic and good source to room
room1.addSource(good_source)

room1.plot(img_order=4)

plt.show()
