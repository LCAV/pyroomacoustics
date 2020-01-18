"""
A simple example of using pyroomacoustics to generate
room impulse responses of shoebox shaped rooms in 3d.
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra

# room dimension
room_dim = [7.5, 10.0, 3.1]

# Create the shoebox

materials = {
    "ceiling": pra.Material.make_freq_flat(0.25, 0.01),
    "floor": pra.Material.make_freq_flat(0.5, 0.1),
    "east": pra.Material.make_freq_flat(0.15, 0.15),
    "west": pra.Material.make_freq_flat(0.07, 0.15),
    "north": pra.Material.make_freq_flat(0.15, 0.15),
    "south": pra.Material.make_freq_flat(0.10, 0.15),
}

params = {
    "ISM": {"max_order": 150, "ray_tracing": False},
    "Hybrid17": {"max_order": 30, "ray_tracing": True},
    "Hybrid3": {"max_order": 3, "ray_tracing": True},
    "Hybrid0": {"max_order": -1, "ray_tracing": True},
}


def make_room(config):

    shoebox = pra.ShoeBox(
        room_dim,
        # materials=pra.Material.from_db('brickwork', 'rpg_skyline'),
        # materials=materials,
        # materials=pra.Material.from_db("brickwork"),
        materials=pra.Material.make_freq_flat(0.07),
        # absorption=0.2,
        fs=16000,
        max_order=config["max_order"],
        ray_tracing=config["ray_tracing"],
        air_absorption=True,
    )

    if config["ray_tracing"]:
        shoebox.set_ray_tracing(receiver_radius=0.5, n_rays=10000)

    # source and mic locations
    shoebox.add_source([2.5, 7.1, 2])
    shoebox.add_microphone_array(
        pra.MicrophoneArray(np.array([[2, 1.5, 2]]).T, shoebox.fs)
    )

    return shoebox


rirs = {}

plt.figure()

for name, config in params.items():

    print("Simulate: ", name)

    shoebox = make_room(config)

    shoebox.image_source_model()
    shoebox.ray_tracing()
    shoebox.compute_rir()

    rir = shoebox.rir[0][0].copy()
    rirs[name] = rir

    print(
        f"{name}: RT60 == ",
        pra.experimental.measure_rt60(rir, shoebox.fs, decay_db=60),
    )

    time = np.arange(rir.shape[0]) / shoebox.fs
    plt.figure(1)
    plt.plot(time, rir, label=name)
    plt.legend()

    plt.figure(2)
    pra.experimental.measure_rt60(rir, plot=True, decay_db=60, fs=shoebox.fs)

plt.legend()
plt.show()
