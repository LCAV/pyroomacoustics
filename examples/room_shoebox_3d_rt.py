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
    "ceiling": pra.Material.from_db("hard_surface"),
    "floor": pra.Material.from_db("6mm_carpet"),
    "east": pra.Material.from_db("rough_concrete"),
    "west": pra.Material.from_db("rough_concrete"),
    "north": pra.Material.from_db("rough_concrete"),
    "south": pra.Material.from_db("rough_concrete"),
}

params = {
    "ISM": {"max_order": 17, "ray_tracing": False},
    "Hybrid17": {"max_order": 17, "ray_tracing": True},
    "Hybrid3": {"max_order": 3, "ray_tracing": True},
}

rirs = {}

plt.figure()

for name, config in params.items():

    print("Simulate: ", name)

    shoebox = pra.ShoeBox(
        room_dim,
        # materials=pra.Material.from_db('brickwork', 'rpg_skyline'),
        # materials=materials,
        # materials=pra.Material.from_db("brickwork"),
        materials=pra.Material.make_freq_flat(0.15, 0.1),
        # absorption=0.2,
        fs=16000,
        max_order=config["max_order"],
        ray_tracing=config["ray_tracing"],
        # air_absorption=True,
    )

    print("Hybrid ?", shoebox.room_engine.is_hybrid_sim)
    print("RT enabled?", shoebox.simulator_state["rt_needed"])

    if config["ray_tracing"]:
        shoebox.set_ray_tracing(receiver_radius=0.5, n_rays=10000)

    # source and mic locations
    shoebox.add_source([2, 3.1, 2])
    shoebox.add_microphone_array(
        pra.MicrophoneArray(np.array([[2, 1.5, 2]]).T, shoebox.fs)
    )

    shoebox.image_source_model()
    shoebox.ray_tracing()
    shoebox.compute_rir()

    rirs[name] = shoebox.rir[0][0]

    print(
        f"{name}: RT60 == ", pra.experimental.measure_rt60(shoebox.rir[0][0], shoebox.fs)
    )

    rir = shoebox.rir[0][0]

    time = np.arange(rir.shape[0]) / shoebox.fs
    plt.plot(time, rir, label=name)

plt.legend()
plt.show()
