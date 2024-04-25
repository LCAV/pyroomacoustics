"""
2022, Robin Scheibler

This test just checks that the ray tracing runs properly without segfault
for the Shoebox rooms.

It was created to address Issue #293 on github.
https://github.com/LCAV/pyroomacoustics/issues/293<Paste>
"""

import numpy as np
import pytest

import pyroomacoustics as pra


def test_issu293_segfault_2d():
    np.random.seed(0)
    for i in range(30):
        room_dim = [30.0, 30.0]
        source = [2.0, 3.0]
        mic_array = [[8.0], [8.0]]

        room = pra.ShoeBox(
            room_dim,
            ray_tracing=True,
            materials=pra.Material(energy_absorption=0.1, scattering=0.1),
            air_absorption=False,
            max_order=1,
        )

        room.add_microphone_array(mic_array)
        room.add_source(source)
        room.set_ray_tracing(n_rays=10_000)
        room.compute_rir()


def test_issu293_segfault_3d():
    np.random.seed(0)
    for i in range(30):
        room_dim = [5, 5, 5]
        source = [2, 3, 3]
        mic_array = [[4.5], [4.5], [4.5]]

        room = pra.ShoeBox(
            room_dim,
            ray_tracing=True,
            materials=pra.Material(energy_absorption=0.1, scattering=0.1),
            air_absorption=False,
            max_order=0,
        )
        room.add_microphone_array(mic_array)
        room.add_source(source)
        room.set_ray_tracing(n_rays=10_000)
        room.compute_rir()


if __name__ == "__main__":
    test_issu293_segfault_2d()
    test_issu293_segfault_3d()
