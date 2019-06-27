"""
A simple example of using pyroomacoustics to generate
room impulse responses of shoebox shaped rooms in 3d.
"""

from __future__ import print_function

import itertools
import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra

def sabine(room_dim, absorption, c):

    volume = np.prod(room_dim)
    surface = np.sum([x * y for x, y in itertools.combinations(room_dim, 2)])

    # reverberation time
    rt60 = 24 * np.log(10) * volume / (c * surface * absorption)

    # critical distance
    d_c = 0.25 * np.sqrt(surface / np.pi)
    return rt60, d_c

# room dimension
room_dim = [30, 15.9, 6]
energy_absorption = 0.25
mic_src_angle = np.radians(37.5)

# compute the theoretical properties
rt60_thy, d_c = sabine(room_dim, energy_absorption, pra.constants.get('c'))

# Create the shoebox
room = pra.ShoeBox(
    room_dim,
    materials=pra.Material.make_freq_flat(energy_absorption),
    fs=16000,
    max_order=3,
    ray_tracing=True,
    air_absorption=False,  # not taken into account in Sabine
)
room.set_ray_tracing(n_rays=50000, receiver_radius=0.5)

# place the source and mic separated
# by at least the critical distance
x, y = 0.5 * d_c * np.cos(mic_src_angle), 0.6 * d_c * np.sin(mic_src_angle)
room.add_source([room_dim[0] / 2 + x, room_dim[1] / 2 + y, 2])
room.add_microphone_array(
        pra.MicrophoneArray(np.array([[room_dim[0] / 2. - x, room_dim[1] / 2. - y, 2]]).T, room.fs)
        )

room.compute_rir()

rt60_exp = pra.experimental.measure_rt60(room.rir[0][0], room.fs)

print("RT60 in theory={:.3f} and practice={:.3f}".format(rt60_thy, rt60_exp))

plt.figure()
room.plot_rir()

plt.show()
