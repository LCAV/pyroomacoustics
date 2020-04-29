"""
A simple example of using pyroomacoustics to simulate
sound propagation in a shoebox room and record the result
to a wav file.
"""

from __future__ import print_function
import time
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile

fs, audio_anechoic = wavfile.read("examples/samples/guitar_16k.wav")

# Create the shoebox
shoebox = pra.ShoeBox(
    [5, 4, 6],  # room dimensions
    materials=pra.Material.make_freq_flat(0.35),
    fs=fs,
    max_order=15,
)

# source and mic locations
shoebox.add_source([2, 3.1, 2], signal=audio_anechoic)
shoebox.add_microphone([2, 1.5, 2])

# run ism
shoebox.simulate()

audio_reverb = shoebox.mic_array.to_wav(
    "examples/samples/guitar_16k_reverb.wav", norm=True, bitdepth=np.int16
)
