import argparse, os
import numpy as np
from stl import mesh
import pyroomacoustics as pra

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic room from STL file example")
    parser.add_argument("file", type=str, help="Path to STL file")
    args = parser.parse_args()

    path_to_musis_stl_file = "./data/raw/MUSIS_3D_no_mics_simple.stl"

    material = pra.Material.make_freq_flat(0.2, 0.1)

    # with numpy-stl
    the_mesh = mesh.Mesh.from_file(args.file)
    ntriang, nvec, npts = the_mesh.vectors.shape
    size_reduc_factor = 500.  # to get a realistic room size (not 3km)

    # create one wall per triangle
    walls = []
    for w in range(ntriang):
        walls.append(
            pra.wall_factory(
                the_mesh.vectors[w].T / size_reduc_factor,
                material.absorption["coeffs"],
                material.scattering["coeffs"],
            )
        )

    room = pra.Room(
        walls,
        fs=16000,
        max_order=3,
        ray_tracing=True,
        air_absorption=True,
    )
    # Set options for the ray tracer
    room.set_ray_tracing(n_rays=30000, receiver_radius=0.5)

    room.add_source([-2.0, 2.0, 1.8])
    room.add_microphone_array(
        pra.MicrophoneArray(
            np.array([[-6.5, 8.5, 3+0.1], [-6.5, 8.1, 3+0.1]]).T, room.fs)
    )

    # compute the rir
    room.image_source_model()
    room.ray_tracing()
    room.compute_rir()
    room.plot_rir()

    # show the room
    room.plot(img_order=1)
    plt.show()
