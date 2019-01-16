'''
A simple example of using pyroomacoustics to simulate
sound propagation in a shoebox room and record the result
to a wav file.
'''

from __future__ import print_function
import time
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile

fs, audio_anechoic = wavfile.read('examples/samples/guitar_16k.wav')

# room dimension
room_dim = [5, 4, 6]

# Create the shoebox
shoebox = pra.ShoeBox(
    room_dim,
    absorption=0.2,
    fs=fs,
    max_order=15,
    )

# source and mic locations
shoebox.add_source([2, 3.1, 2], signal=audio_anechoic)
shoebox.add_microphone_array(
        pra.MicrophoneArray(
            np.array([[2, 1.5, 2]]).T, 
            shoebox.fs)
        )

import pdb; pdb.set_trace()

# run ism
shoebox.simulate()

audio_reverb = shoebox.mic_array.to_wav('examples/samples/guitar_16k_reverb.wav', norm=True, bitdepth=np.int16)

