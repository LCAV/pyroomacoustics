"""
A simple example of using pyroomacoustics to generate
room impulse responses of shoebox shaped rooms in 3d.
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra

# room dimension
room_dim = [7.5, 9.9, 3]

# Create the shoebox

materials = {
    'ceiling': pra.Material.from_db('hard_surface'),
    'floor': pra.Material.from_db('6mm_carpet'),
    'east': pra.Material.from_db('brickwork'),
    'west': pra.Material.from_db('brickwork'),
    'north': pra.Material.from_db('brickwork'),
    'south': pra.Material.from_db('brickwork'),
}

shoebox = pra.ShoeBox(
    room_dim,
    # materials=pra.Material.from_db('brickwork', 'rpg_skyline'),
    materials=materials,
    # materials=pra.Material.from_db('brickwork'),
    # materials=pra.Material.make_freq_flat(0.1, 0.2),
    # absorption=0.2,
    fs=16000,
    max_order=3,
    ray_tracing=True,
    air_absorption=True,
)

# source and mic locations
shoebox.add_source([2, 3.1, 2])
shoebox.add_microphone_array(
        pra.MicrophoneArray(np.array([[2, 1.5, 2]]).T, shoebox.fs)
        )

shoebox.image_source_model()
shoebox.ray_tracing()
shoebox.compute_rir()

plt.figure()
shoebox.plot_rir()

plt.show()
