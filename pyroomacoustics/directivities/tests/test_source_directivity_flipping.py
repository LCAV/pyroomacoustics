from unittest import TestCase

import numpy as np

import pyroomacoustics as pra


def source_angle_shoebox(image_source_loc, wall_flips, mic_loc):
    """
    Determine outgoing angle for each image source for a ShoeBox configuration.

    Implementation of the method described in the paper:
    https://www2.ak.tu-berlin.de/~akgroup/ak_pub/2018/000458.pdf

    Parameters
    -----------
    image_source_loc : array_like
        Locations of image sources.
    wall_flips: array_like
        Number of x, y, z flips for each image source.
    mic_loc: array_like
        Microphone location.

    Returns
    -------
    azimuth : :py:class:`~numpy.ndarray`
        Azimith for each image source, in radians
    colatitude : :py:class:`~numpy.ndarray`
        Colatitude for each image source, in radians.

    """

    image_source_loc = np.array(image_source_loc)
    wall_flips = np.array(wall_flips)
    mic_loc = np.array(mic_loc)

    dim, n_sources = image_source_loc.shape
    assert wall_flips.shape[0] == dim
    assert mic_loc.shape[0] == dim

    p_vector_array = image_source_loc - np.array(mic_loc)[:, np.newaxis]
    d_array = np.linalg.norm(p_vector_array, axis=0)

    # Using (12) from the paper
    power_array = np.ones_like(image_source_loc) * -1
    power_array = np.power(power_array, (wall_flips + np.ones_like(image_source_loc)))
    p_dash_array = p_vector_array * power_array

    # Using (13) from the paper
    azimuth = np.arctan2(p_dash_array[1], p_dash_array[0])
    if dim == 2:
        colatitude = np.ones(n_sources) * np.pi / 2
    else:
        colatitude = np.pi / 2 - np.arcsin(p_dash_array[2] / d_array)

    return azimuth, colatitude


class TestSourceDirectivityFlipping(TestCase):
    def test_source_directions(self):
        # create room
        mic_loc = [5, 12, 5]
        source_loc = [5, 2, 5]
        room = (
            pra.ShoeBox(
                p=[10, 14, 10],
                max_order=2,
            )
            .add_source(source_loc)
            .add_microphone(mic_loc)
        )

        # compute image sources
        room.image_source_model()

        # compute azimuth_s and colatitude_s pair for images along x-axis
        source_angle_array = source_angle_shoebox(
            image_source_loc=room.sources[0].images,
            wall_flips=abs(room.sources[0].orders_xyz),
            mic_loc=mic_loc,
        )
        source_angle_array = np.array(source_angle_array)

        source_dir = room.sources[0].directions[0]
        azimuth = np.arctan2(source_dir[1], source_dir[0])
        colatitude = np.pi / 2 - np.arcsin(source_dir[2])
        source_angle_array_2 = np.array([azimuth, colatitude])

        np.testing.assert_almost_equal(
            source_angle_array, source_angle_array_2, decimal=4
        )

    def test_robin_check(self):
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
        source_angle_array = source_angle_shoebox(
            image_source_loc=room.sources[0].images,
            wall_flips=abs(room.sources[0].orders_xyz),
            mic_loc=mic_loc,
        )
        source_angle_array = np.array(source_angle_array)

        skipped = 0
        for i in range(7):
            img = room.sources[0].images[:, i]
            if np.allclose(img, [5, -2, 5]):
                sd = [-np.pi / 2, np.pi / 2]
            elif np.allclose(img, [-5, 2, 5]):
                sd = [3 * np.pi / 4, np.pi / 2]
            elif np.allclose(img, [5, 26, 5]):
                sd = [np.pi / 2, np.pi / 2]
            elif np.allclose(img, [15, 2, 5]):
                sd = [np.pi / 4, np.pi / 2]
            elif np.allclose(img, [5, 2, -5]):
                sd = [np.pi / 2, 3 * np.pi / 4]
            elif np.allclose(img, [5, 2, 15]):
                sd = [np.pi / 2, np.pi / 4]
            elif np.allclose(img, [5, 2, 5]):
                sd = [np.pi / 2, np.pi / 2]
            else:
                skipped += 1
                continue

            np.testing.assert_almost_equal(source_angle_array[:, i], sd)

        assert skipped == 0

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
        source_angle_array = source_angle_shoebox(
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
        source_angle_array = source_angle_shoebox(
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
        source_angle_array = source_angle_shoebox(
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
