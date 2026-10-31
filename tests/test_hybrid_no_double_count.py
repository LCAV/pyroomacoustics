import numpy as np
import pytest

import pyroomacoustics as pra

room_dim = [6.0, 6.0, 2.4]
source_loc = [2.4, 3.6, 0.96]
mic_locs = np.array(
    [
        [1.6, 3.0, 0.9],
        [3.0, 1.6, 1.8],
        [4.4, 3.0, 0.9],
    ]
).T


@pytest.mark.parametrize("max_order", [0, 1, 2, 3])
def test_hybrid_no_double_counting(max_order):
    """
    In the hybrid simulator, the image source model already produces all the
    specular reflections up to order ``max_order``. The ray tracer should thus
    only log specular rays that were reflected strictly more than ``max_order``
    times. Then, no ray tracing energy can reach a microphone before the
    earliest arrival of the image sources of order ``max_order + 1`` (up to the
    size of the receiver sphere), while the later part of the ray tracing
    histogram is kept.
    """
    room = pra.ShoeBox(
        room_dim,
        fs=16000,
        materials=pra.Material(energy_absorption=0.2),
        max_order=max_order,
        ray_tracing=True,
        air_absorption=False,
    )
    room.add_source(source_loc)
    room.add_microphone_array(mic_locs)
    room.ray_tracing()

    # image sources of order exactly max_order + 1
    ref = pra.ShoeBox(
        room_dim,
        fs=16000,
        materials=pra.Material(energy_absorption=0.2),
        max_order=max_order + 1,
    )
    ref.add_source(source_loc)
    ref.add_microphone_array(mic_locs)
    ref.image_source_model()
    src = ref.sources[0]
    images = src.images[:, src.orders == max_order + 1]

    radius = room.rt_args["receiver_radius"]
    bin_dist = room.rt_args["hist_bin_size"] * room.c

    for m in range(mic_locs.shape[1]):
        d_min = np.min(np.linalg.norm(images - mic_locs[:, [m]], axis=0))
        # the ray tracer logs a hit at the distance to the point closest to
        # the receiver center, i.e., at least sqrt(d_min^2 - radius^2)
        first_bin = int(np.floor((np.sqrt(d_min**2 - radius**2) - 0.01) / bin_dist))
        hist = room.rt_histograms[m][0][0]
        assert np.sum(hist[..., :first_bin]) == 0.0
        assert np.sum(hist[..., first_bin:]) > 0.0


@pytest.mark.parametrize("scattering", [0.1, 1.0])
def test_hybrid_scattered_rays_keep_threshold(scattering):
    """
    A ray that was diffusely scattered by a wall does not follow a specular
    path anymore and is not produced by the image source model. When all the
    walls scatter, every ray is scattered at its first reflection, so only the
    direct path (a specular path of order 0) is left to the image source model
    both for ``max_order=0`` and ``max_order=1``. With the same random sequence,
    the ray tracing histograms must then be the same.
    """
    hists = []
    for max_order in [0, 1]:
        room = pra.ShoeBox(
            room_dim,
            fs=16000,
            materials=pra.Material(energy_absorption=0.2, scattering=scattering),
            max_order=max_order,
            ray_tracing=True,
            air_absorption=False,
        )
        room.add_source(source_loc)
        room.add_microphone_array(mic_locs)
        pra.random.seed(numpy=0, libroom=0)
        room.ray_tracing()
        hists.append([room.rt_histograms[m][0][0] for m in range(mic_locs.shape[1])])

    for h0, h1 in zip(*hists):
        assert np.array_equal(h0, h1)
