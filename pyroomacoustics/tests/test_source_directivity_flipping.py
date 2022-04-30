import pyroomacoustics as pra
import numpy as np
from unittest import TestCase

import pyroomacoustics.directivities


class TestSourceDirectivityFlipping(TestCase):
    def test_x(self):
        # create room
        mic_loc = [5, 12, 5]
        source_loc = [5, 2, 5]
        room = (
            pra.ShoeBox(
                p=[10, 14, 10],
                max_order=1,
            )
            .add_source(source_loc)
            .add_microphone(mic_loc)
        )

        # compute image sources
        room.image_source_model()

        # compute azimuth_s and colatitude_s pair for images along x-axis
        source_angle_array = pyroomacoustics.directivities.source_angle_shoebox(
            image_source_loc=room.sources[0].images,
            wall_flips=abs(room.sources[0].orders_xyz),
            mic_loc=mic_loc,
        )
        source_angle_array = np.array(source_angle_array)

        x1_idx = np.where(
            room.sources[0].images[0] == source_loc[0] - 2 * source_loc[0]
        )[0][0]
        x2_idx = np.where(
            room.sources[0].images[0] == source_loc[0] + 2 * source_loc[0]
        )[0][0]
        np.testing.assert_almost_equal(
            source_angle_array[:, x1_idx], [3 * np.pi / 4, np.pi / 2]
        )
        np.testing.assert_almost_equal(
            source_angle_array[:, x2_idx], [np.pi / 4, np.pi / 2]
        )

    def test_y(self):
        # create room
        mic_loc = [12, 5, 5]
        source_loc = [2, 5, 5]
        room = (
            pra.ShoeBox(
                p=[14, 10, 10],
                max_order=1,
            )
            .add_source(source_loc)
            .add_microphone(mic_loc)
        )

        # compute image sources
        room.image_source_model()

        # compute azimuth_s and colatitude_s pair for images along x-axis
        source_angle_array = pyroomacoustics.directivities.source_angle_shoebox(
            image_source_loc=room.sources[0].images,
            wall_flips=abs(room.sources[0].orders_xyz),
            mic_loc=mic_loc,
        )
        source_angle_array = np.array(source_angle_array)

        y1_idx = np.where(
            room.sources[0].images[1] == source_loc[1] - 2 * source_loc[1]
        )[0][0]
        y2_idx = np.where(
            room.sources[0].images[1] == source_loc[1] + 2 * source_loc[1]
        )[0][0]
        np.testing.assert_almost_equal(
            source_angle_array[:, y1_idx], [-np.pi / 4, np.pi / 2]
        )
        np.testing.assert_almost_equal(
            source_angle_array[:, y2_idx], [np.pi / 4, np.pi / 2]
        )

    def test_z(self):
        # create room
        mic_loc = [12, 5, 5]
        source_loc = [2, 5, 5]
        room = (
            pra.ShoeBox(p=[14, 10, 10], max_order=1)
            .add_source(source_loc)
            .add_microphone([12, 5, 5])
        )

        # compute image sources
        room.image_source_model()

        # compute azimuth_s and colatitude_s pair for images along z-axis
        source_angle_array = pyroomacoustics.directivities.source_angle_shoebox(
            image_source_loc=room.sources[0].images,
            wall_flips=abs(room.sources[0].orders_xyz),
            mic_loc=mic_loc,
        )
        source_angle_array = np.array(source_angle_array)

        z1_idx = np.where(
            room.sources[0].images[2] == source_loc[2] - 2 * source_loc[2]
        )[0][0]
        z2_idx = np.where(
            room.sources[0].images[2] == source_loc[2] + 2 * source_loc[2]
        )[0][0]
        np.testing.assert_almost_equal(
            source_angle_array[:, z1_idx], [0, 3 * np.pi / 4]
        )
        np.testing.assert_almost_equal(source_angle_array[:, z2_idx], [0, np.pi / 4])
