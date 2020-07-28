from __future__ import division, print_function

import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.transform import STFT
import time
import warnings

# test signal
np.random.seed(0)
num_mic = 25
signals = np.random.randn(100000, num_mic).astype(np.float32)
fs = 16000

# STFT parameters
block_size = 512
hop = block_size//2
win = pra.hann(block_size)
x_r = np.zeros(signals.shape)
num_times = 50

print()

"""
One frame at a time
"""
print("Averaging computation time over %d cases of %d channels of %d samples (%0.1f s at %0.1f kHz)." 
    % (num_times,num_mic,len(signals),(len(signals)/fs),fs/1000) )
print()
print("----- SINGLE FRAME AT A TIME -----")
print("With STFT object (not fixed) : ", end="")
stft = STFT(block_size, hop=hop, channels=num_mic,
    streaming=True, analysis_window=win)
start = time.time()
for k in range(num_times):

    x_r = np.zeros(signals.shape)
    n = 0
    while  signals.shape[0] - n > hop:
        stft.analysis(signals[n:n+hop,])
        x_r[n:n+hop,] = stft.synthesis()
        n += hop
avg_time = (time.time()-start)/num_times
print("%0.3f sec" % avg_time)
err_dB = 20*np.log10(np.max(np.abs(signals[:n-hop,] - x_r[hop:n,])))
print("Error [dB] : %0.3f" % err_dB)


print("With STFT object (fixed) : ", end="")
stft = STFT(block_size, hop=hop, channels=num_mic, num_frames=1, 
    streaming=True, analysis_window=win)
start = time.time()
for k in range(num_times):

    x_r = np.zeros(signals.shape)
    n = 0
    while  signals.shape[0] - n > hop:
        stft.analysis(signals[n:n+hop,])
        x_r[n:n+hop,] = stft.synthesis()
        n += hop
avg_time = (time.time()-start)/num_times
print("%0.3f sec" % avg_time)
err_dB = 20*np.log10(np.max(np.abs(signals[:n-hop,] - x_r[hop:n,])))
print("Error [dB] : %0.3f" % err_dB)


"""
Multiple frame at a time (non-streaming)
"""
print()
print("----- MULTIPLE FRAMES AT A TIME -----")

warnings.filterwarnings("ignore") # to avoid warning of appending zeros to be printed
print("With STFT object (not fixed) : ", end="")
stft = STFT(block_size, hop=hop, channels=num_mic,
    analysis_window=win, streaming=False)
start = time.time()
for k in range(num_times):

    stft.analysis(signals)
    x_r = stft.synthesis()

avg_time = (time.time()-start)/num_times
print("%0.3f sec" % avg_time)
err_dB = 20*np.log10(np.max(np.abs(signals[hop:len(x_r)-hop] - 
    x_r[hop:len(x_r)-hop])))
print("Error [dB] : %0.3f" % err_dB)
warnings.filterwarnings("default")


print("With STFT object (fixed) : ", end="")
num_frames = (len(signals)-block_size)//hop + 1
stft = STFT(block_size, hop=hop, channels=num_mic,
    num_frames=num_frames, analysis_window=win, streaming=False)
start = time.time()
for k in range(num_times):

    stft.analysis(signals[:(num_frames-1)*hop+block_size,:])
    x_r = stft.synthesis()
avg_time = (time.time()-start)/num_times
print("%0.3f sec" % avg_time)
err_dB = 20*np.log10(np.max(np.abs(signals[hop:len(x_r)-hop] - 
    x_r[hop:len(x_r)-hop])))
print("Error [dB] : %0.3f" % err_dB)


