import warnings

import numpy as np
import pytest

import pyroomacoustics as pra


@pytest.mark.parametrize("receiver_radius", [0.5, 0.1])
@pytest.mark.parametrize("use_scattering", [True, False])
def test_nan_with_rt(receiver_radius, use_scattering):
    pra.random.seed(
        140
    )  # this seed seems to reproduce the NaN issue reliably on my machine.

    room = pra.ShoeBox(
        [4.144901752471924, 6.039876461029053, 2.7347633838653564],
        fs=24000,
        materials=pra.Material(
            energy_absorption=0.13035287, scattering=0.1 if use_scattering else 0.0
        ),
        max_order=0,
        ray_tracing=True,
        air_absorption=True,
    )
    room.set_ray_tracing(receiver_radius=receiver_radius)

    # The microphone is within 0.5 m of the wall.
    room.add_microphone([2.304156303866167, 0.17025121891320288, 1.4909059943866196])
    room.add_source([1.9948741012396916, 3.8261834557486685, 0.8465485818395717])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        room.compute_rir()

        sqrt_warns = [
            x for x in w if "invalid value encountered in sqrt" in str(x.message)
        ]
        assert not np.any(np.isnan(room.rir[0][0])), "Found NaNs in RIR."
        assert not sqrt_warns, "Warned about invalid value in sqrt."
