"""
A simple example of using pyroomacoustics to generate
room impulse responses of shoebox shaped rooms in 3d.
"""

from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra

# Create the shoebox
shoebox = pra.ShoeBox(
    [5, 4, 6],  # room dimension
    absorption=0.2,
    fs=16000,
    max_order=15,
)

# source and mic locations
shoebox.add_source([2, 3.1, 2])
shoebox.add_microphone([2, 1.5, 2])

# run ism
shoebox.compute_rir()

# Plot the result up to fourth order images
shoebox.plot(img_order=4)
plt.show()
