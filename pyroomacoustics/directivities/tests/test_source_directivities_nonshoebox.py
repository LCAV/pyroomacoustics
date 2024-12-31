import numpy as np
import pytest

import pyroomacoustics as pra

"""
+ - - - - - - - - - - - +
|                       |
|                       |
|                       |
|       + - - - - - - - +
|       |
|       |
|       |
+ - - - +
"""

corners = np.array([[0, 0], [4, 0], [4, 4], [12, 4], [12, 8], [0, 8]])
source_location = [6, 6, 1.5]
mic_location = [1, 6, 1.5]

images_expected = np.array(
    [
        source_location,  # direct
        [6, 10, 1.5],  # north
        [-6, 6, 1.5],  # west
        [18, 6, 1.5],  # east
        [6, 6, 4.5],  # ceilling
        [6, 6, -1.5],  # floor
    ]
)
directions_expected = np.array(
    [
        [-1, 0, 0],
        np.array([-2.5, 2, 0]) / np.sqrt(2.5**2 + 2**2),
        [-1, 0, 0],
        [1, 0, 0],
        np.array([-2.5, 0, 1.5]) / np.sqrt(2.5**2 + 1.5**2),
        np.array([-2.5, 0, -1.5]) / np.sqrt(2.5**2 + 1.5**2),
    ]
)


room = pra.Room.from_corners(corners.T, materials=pra.Material(0.1), max_order=1)
room.extrude(3.0, materials=pra.Material(0.1))
room.add_source(
    source_location,
    directivity=pra.directivities.Cardioid(
        orientation=pra.directivities.DirectionVector(
            azimuth=0, colatitude=np.pi / 2, degrees=False
        )
    ),
).add_microphone(mic_location).add_microphone([3, 6, 1.5])
# We add one microphone with other images visible to make sure we have some
# image sources not visible.


def test_source_directions_nonshoebox():
    room.image_source_model()
    visible = room.visibility[0][0, :]
    directions_obtained = room.sources[0].directions[0, :, visible]
    images_obtained = room.sources[0].images[:, visible]

    order_obtained = np.lexsort(images_obtained)
    images_obtained = images_obtained[:, order_obtained].T
    directions_obtained = directions_obtained[order_obtained, :]

    order_expected = np.lexsort(images_expected.T)
    images_expected_reordered = images_expected[order_expected, :]
    directions_expected_reordered = directions_expected[order_expected, :]

    np.testing.assert_almost_equal(images_obtained, images_expected_reordered)
    np.testing.assert_almost_equal(directions_obtained, directions_expected_reordered)
