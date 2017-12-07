from __future__ import division, print_function

import numpy as np
import pyroomacoustics as pra
import time

# test signal
num_mic = 25
signals = np.random.randn(100000, num_mic)

# STFT parameters
block_size = 256
hop = block_size//2
num_times = 100

# multiple frames at a time
stft = pra.realtime.STFT(block_size, hop=hop, channels=num_mic)
start = time.time()
for k in range(num_times):

    stft.analysis(signals)
    x_r = stft.synthesis()

avg_time = (time.time()-start)/num_times
print("Multiple frames : %0.3f sec" % avg_time)


# one frame at a time
stft = pra.realtime.STFT(block_size, hop=hop, channels=num_mic)
start = time.time()
for k in range(num_times):

    x_r = np.zeros(signals.shape)
    n = 0
    while  signals.shape[0] - n > hop:
        stft.analysis(signals[n:n+hop,])
        x_r[n:n+hop,] = stft.synthesis()
        n += hop

avg_time = (time.time()-start)/num_times
print("Single frame at a time : %0.3f sec" % avg_time)


# one frame at a time (fixed)
stft = pra.realtime.STFT(block_size, hop=hop, channels=num_mic, num_frames=0)
start = time.time()
for k in range(num_times):

    x_r = np.zeros(signals.shape)
    n = 0
    while  signals.shape[0] - n > hop:
        stft.analysis(signals[n:n+hop,])
        x_r[n:n+hop,] = stft.synthesis()
        n += hop

avg_time = (time.time()-start)/num_times
print("Single frame at a time (fixed): %0.3f sec" % avg_time)


# original method
start = time.time()
for k in range(num_times):

    y_mic_stft = np.array([pra.stft(signals[:, k], block_size, hop,
         transform=np.fft.rfft, win=np.hanning(block_size)).T
        for k in range(num_mic)])
    x_r = np.array([pra.istft(y_mic_stft[k,:,:].T, block_size, hop, transform=np.fft.irfft)
        for k in range(num_mic)])

avg_time = (time.time()-start)/num_times
print("Original method : %0.3f sec" % avg_time)


