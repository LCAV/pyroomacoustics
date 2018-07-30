"""
STFT Block Processing
=====================

In this example, we will apply a simple moving average filter in the frequency
domain. We will use the STFT class that lets us do block-wise processing
suitable for real-time application on streaming audio.

In this example, we perform offline processing, but the methodology is
block-wise and thus very easy to transfer to the streaming case.  We use half
overlapping blocks with Hann windowing and apply a moving average filter in the
frequency domain. Finally, we plot and compare the spectrograms before and
after filtering.
"""

from __future__ import division, print_function
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import os

# filter to apply
h_len = 99
h = np.ones(h_len)
h /= np.linalg.norm(h)

# parameters
block_size = 512 - h_len + 1  # make sure the FFT size is a power of 2
hop = block_size // 2  # half overlap
window = pra.hann(block_size, flag='asymmetric', length='full')  # analysis window (no synthesis window)

# open single channel audio file
fn = os.path.join(os.path.dirname(__file__), 'input_samples', 'singing_8000.wav')
fs, audio = wavfile.read(fn)

# Create the STFT object
stft = pra.transform.STFT(block_size, hop=hop, analysis_window=window, channels=1, streaming=True)

# set the filter and the appropriate amount of zero padding (back)
if h_len > 1:
    stft.set_filter(h, zb=h.shape[0] - 1)

# collect the processed blocks
processed_audio = np.zeros(audio.shape)

# process the signals while full blocks are available
n = 0
while audio.shape[0] - n > hop:

    # go to frequency domain
    stft.analysis(audio[n:n+hop, ])

    stft.process()  # apply the filter

    # copy processed block in the output buffer
    processed_audio[n:n+hop, ] = stft.synthesis()

    n += hop

# plot the spectrogram before and after filtering
plt.figure()
plt.subplot(2, 1, 1)
plt.specgram(audio[:n-hop].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30)
plt.title('Original Signal')
plt.subplot(2, 1, 2)
plt.specgram(processed_audio[hop:n], NFFT=256, Fs=fs, vmin=-20, vmax=30)
plt.title('Lowpass Filtered Signal')
plt.tight_layout(pad=0.5)
plt.show()
