'''
STFT Block Processing
=====================

In this example, we will apply a simple moving average filter in the
frequency domain. We will use the STFT class that lets us do block-wise
processing suitable for real-time application on streaming audio.

In this example, we perform offline processing, but the methodology
is block-wise and thus very easy to transfer to the streaming case.

This example requires sounddevice_ package to be installed.

.. _sounddevice: https://github.com/spatialaudio/python-sounddevice

'''
from __future__ import division, print_function
import numpy as np
from scipy.signal import fftconvolve
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# filter to apply
h_len = 1
h = np.ones(h_len)
h /= np.linalg.norm(h)

# parameters
block_size = 512 - h_len + 1  # make sure the FFT size is a power of 2
hop = block_size  # half overlap
#window = pra.cosine(block_size)  # analysis window (no synthesis window)
window = None

# open single channel audio file
fs, audio = wavfile.read('examples/input_samples/singing_8000.wav')

# Create the STFT object
stft = pra.realtime.STFT(block_size, fs, hop=hop, analysis_window=window)

# set the filter and the appropriate amount of zero padding (back)
if h_len > 1:
    stft.set_filter(h, zb=h.shape[0] - 1)

# collect the processed blocks
processed_audio = np.zeros(audio.shape)

# process the signals while full blocks are available
n = 0
while  audio.shape[0] - n > hop:

    # go to frequency domain
    stft.analysis(audio[n:n+hop])

    stft.process()  # apply the filter

    # copy processed block in the output buffer
    out = stft.synthesis()
    processed_audio[n:n+hop] = stft.synthesis()

    n += hop

plt.figure()
plt.plot(audio)
plt.plot(processed_audio)

plt.figure()
plt.plot(np.abs(audio[:n] - processed_audio[:n]))
plt.show()
