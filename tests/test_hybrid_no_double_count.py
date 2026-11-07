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


@pytest.mark.parametrize("scattering", [0.0, 1e-12])
@pytest.mark.parametrize("max_order", [0, 1, 2, 3])
def test_hybrid_no_double_counting(max_order, scattering):
    """
    The image source model already produces the specular paths with up to
    ``max_order`` reflections, so the ray tracer should log no energy before
    the earliest arrival of the image sources of order ``max_order + 1``.
    """
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
