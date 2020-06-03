"""
This sample program demonstrate how to import a model from an STL file.
Currently, the materials need to be set in the program which is not very practical
when different walls have different materials.

The STL file was kindly provided by Diego Di Carlo (@Chutlhu).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

import pyroomacoustics as pra

try:
    from stl import mesh
except ImportError as err:
    print(
        "The numpy-stl package is required for this example. "
        "Install it with `pip install numpy-stl`"
    )
    raise err


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic room from STL file example")
    parser.add_argument("file", type=str, help="Path to STL file")
    args = parser.parse_args()

    path_to_musis_stl_file = "./data/raw/MUSIS_3D_no_mics_simple.stl"

    material = pra.Material(energy_absorption=0.2, scattering=0.1)

    # with numpy-stl
    the_mesh = mesh.Mesh.from_file(args.file)
    ntriang, nvec, npts = the_mesh.vectors.shape
    size_reduc_factor = 500.0  # to get a realistic room size (not 3km)

    # create one wall per triangle
    walls = []
    for w in range(ntriang):
        walls.append(
            pra.wall_factory(
                the_mesh.vectors[w].T / size_reduc_factor,
                material.energy_absorption["coeffs"],
                material.scattering["coeffs"],
            )
        )

    room = (
        pra.Room(walls, fs=16000, max_order=3, ray_tracing=True, air_absorption=True,)
        .add_source([-2.0, 2.0, 1.8])
        .add_microphone_array(np.c_[[-6.5, 8.5, 3 + 0.1], [-6.5, 8.1, 3 + 0.1]])
    )

    # compute the rir
    room.image_source_model()
    room.ray_tracing()
    room.compute_rir()
    room.plot_rir()

    # show the room
    room.plot(img_order=1)
    plt.show()
