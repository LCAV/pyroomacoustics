"""
A simple example of using pyroomacoustics to generate
room impulse responses of shoebox shaped rooms in 3d.
"""

from __future__ import print_function

import time
import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra

np.random.seed(0)

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
    "ISM17": {"max_order": 17, "ray_tracing": False},
    "Hybrid17": {"max_order": 17, "ray_tracing": True},
    "Hybrid3": {"max_order": 3, "ray_tracing": True},
    "Hybrid0": {"max_order": -1, "ray_tracing": True},
}


def make_room(config):

    shoebox = (
        pra.ShoeBox(
            room_dim,
            materials=materials,
            # materials=pra.Material.make_freq_flat(0.07),
            fs=16000,
            max_order=config["max_order"],
            ray_tracing=config["ray_tracing"],
            air_absorption=True,
        )
        .add_source([2.5, 7.1, 2])
        .add_microphone([2, 1.5, 2])
    )

    if config["ray_tracing"]:
        shoebox.set_ray_tracing(receiver_radius=0.5, n_rays=10000)

    # source and mic locations

    return shoebox


def chrono(f, *args, **kwargs):
    t = time.perf_counter()
    ret = f(*args, **kwargs)
    runtime = time.perf_counter() - t
    return runtime, ret


rirs = {}

plt.figure()

for name, config in params.items():

    print("Simulate: ", name)

    shoebox = make_room(config)

    t_ism, _ = chrono(shoebox.image_source_model)
    t_rt, _ = chrono(shoebox.ray_tracing)
    t_rir, _ = chrono(shoebox.compute_rir)

    rir = shoebox.rir[0][0].copy()
    rirs[name] = rir

    time_vec = np.arange(rir.shape[0]) / shoebox.fs
    plt.figure(1)
    plt.plot(time_vec, rir, label=name)
    plt.legend()

    plt.figure(2)
    rt60 = pra.experimental.measure_rt60(rir, plot=True, decay_db=60, fs=shoebox.fs)

    print(
        f"{name}: RT60 == {rt60:.3f} t_ism == {t_ism:.6f} "
        f"t_rt == {t_rt:.6f} t_rir == {t_rir:.6f}"
    )


plt.legend()
plt.show()
