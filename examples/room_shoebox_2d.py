"""
A simple example of using pyroomacoustics to generate
room impulse responses of shoebox shaped rooms in 2d.
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

import pyroomacoustics as pra

# create the room with sources and mics
# Room 4m by 6m
room = pra.ShoeBox([4, 6], fs=16000, absorption=0.1, max_order=4)

# add mic and good source to room
room.add_source([1, 4.5])

# place 1 microphone in the room
room.add_microphone([3, 2])

# Run the image source model
room.image_source_model()

# Plot the result up to fourth order images
room.plot(img_order=4)

plt.figure()
room.plot_rir()
plt.show()
