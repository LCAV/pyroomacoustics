"""
This examples demonstrates how to build a 3D room for multi-band simulation
with rich wall materials.

2022 (c) @noahdeetzers, @fakufaku
"""

import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra

# Define the materials array
ceiling_mat = {
    "description": "Example ceiling material",
    "coeffs": [0.1, 0.2, 0.1, 0.1, 0.1, 0.05],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000],
}
floor_mat = {
    "description": "Example floor material",
    "coeffs": [0.1, 0.2, 0.1, 0.1, 0.1, 0.05],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000],
}
wall_mat = {
    "description": "Example wall material",
    "coeffs": [0.1, 0.2, 0.1, 0.1, 0.1, 0.05],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000],
}


def makeComplexRoom(height):
    # this creates a more complex room with 9 non-orthogonal sides

    # Creating the array of the 2d room
    pol = (
        4
        * np.array(
            [[1, 0], [0, 2], [0, 3], [1, 4], [1.5, 3.8], [2, 4], [3, 3], [3, 2], [2, 0]]
        ).T
    )

    # initial acoustic coeffs
    m = pra.make_materials(*[(wall_mat,) for i in range(pol.shape[1])])

    room = pra.Room.from_corners(
        pol,
        absorption=None,
        fs=8000,
        t0=0.0,
        max_order=1,
        sigma2_awgn=None,
        sources=None,
        mics=None,
        materials=m,
    )

    # Create the 3D room by extruding the 2D by 3 meters
    me = pra.make_materials(floor=(floor_mat,), ceiling=(ceiling_mat,))
    room.extrude(height, materials=me)

    # Add a source somewhere in the room
    room.add_source([3, 3, 0.5])

    # Create a linear array beamformer with 4 microphones
    # Place an array of two microphones
    R = np.array([[3.0, 2.2], [2.25, 2.1], [0.6, 0.55]])
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    return room


room = makeComplexRoom(5)

room.image_source_model()
fig, ax = room.plot()
ax.set_xlim([0, 12])
ax.set_ylim([0, 12])
ax.set_zlim([0, 12])
plt.show()
