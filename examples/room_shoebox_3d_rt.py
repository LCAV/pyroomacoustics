"""
This script demonstrates and compares the different simulators

- Image Source Method (ISM) order 17
- Ray Tracing (RT) only
- Hybrid ISM/RT with ISM order 17
- Hybrid ISM/RT with ISM order 3

It compares the theoretical and measured RT60 and the computation times.

We can observe that the order 17 is not sufficient when using only the ISM.
RT is able to simulate the whole tail.
Also, the RT60 of ISM is not completely consistent with the Hybrid method as it
doesn't include scattering. Scattering reduces the length of the reverberent tail.
"""

from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra

np.random.seed(0)

# room dimension
room_dim = [10.67, 7.5, 3.5]

# Create the shoebox
materials = pra.make_materials(
    ceiling=(0.25, 0.01),
    floor=(0.5, 0.1),
    east=(0.15, 0.15),
    west=(0.07, 0.15),
    north=(0.15, 0.15),
    south=(0.10, 0.15),
)

params = {
    "ISM17": {"max_order": 17, "ray_tracing": False},
    "Hybrid0": {"max_order": -1, "ray_tracing": True},
    "Hybrid17": {"max_order": 17, "ray_tracing": True},
    "Hybrid3": {"max_order": 3, "ray_tracing": True},
}


def make_room(config):
    """
    A short helper function to make the room according to config
    """

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

    return shoebox


def chrono(f, *args, **kwargs):
    """
    A short helper function to measure running time
    """
    t = time.perf_counter()
    ret = f(*args, **kwargs)
    runtime = time.perf_counter() - t
    return runtime, ret


if __name__ == "__main__":

    rirs = {}

    for name, config in params.items():

        print("Simulate: ", name)

        shoebox = make_room(config)

        rt60_sabine = shoebox.rt60_theory(formula="sabine")
        rt60_eyring = shoebox.rt60_theory(formula="eyring")

        # run separately the different parts of the simulation
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
        rt60 = shoebox.measure_rt60(plot=True, decay_db=60)

        print(f"  RT60:")
        print(f"    - Eyring   {rt60_eyring:.3f} s")
        print(f"    - Sabine   {rt60_sabine:.3f} s")
        print(f"    - Measured {rt60[0, 0]:.3f} s")

        print("  Computation:")
        print(f"    - ISM  {t_ism:.6f} s")
        print(f"    - RT   {t_rt:.6f} s")
        print(f"    - RIR  {t_rir:.6f} s")
        print(f"    Total: {t_ism + t_rt + t_rir:.6f} s")
        print()

    plt.legend()
    plt.show()
