import numpy as np
import pytest

import pyroomacoustics as pra


@pytest.fixture
def room():
    fs = 16000
    c = pra.constants.get("c")
    room_dim = np.array([c, c, 2 * c])
    src = np.ones(3) * 0.5 * c
    mic = np.array([0.5, 0.5, 1.5]) * c
    mat = pra.Material(energy_absorption="hard_surface")
    room = (
        pra.ShoeBox(room_dim, fs, max_order=1, materials=mat)
        .add_source(src)
        .add_microphone(mic)
    )
    return room


def test_set_air_absorption_1(room):
    air_abs = 0.5
    room.set_air_absorption([air_abs])

    air_abs_coeffs = room.air_absorption
    assert len(air_abs_coeffs) == 7
    assert all([a == air_abs for a in air_abs_coeffs])


def test_set_air_absorption_2(room):
    air_abs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    room.set_air_absorption(air_abs)

    air_abs_coeffs = room.air_absorption
    assert len(air_abs_coeffs) == 7
    assert all(air_abs == air_abs_coeffs)


@pytest.mark.parametrize("coefficients", [None, np.linspace(0.01, 0.1, 7)])
@pytest.mark.parametrize("max_order, ray_tracing", [(3, False), (-1, True), (3, True)])
def test_unset_air_absorption_rir(coefficients, max_order, ray_tracing):
    def make_room(air_absorption):
        room = pra.ShoeBox(
            [5, 4, 3],
            fs=16000,
            max_order=max_order,
            materials=pra.Material(0.3, scattering=0.2),
            air_absorption=air_absorption,
            ray_tracing=ray_tracing,
        )
        room.add_source([1, 1, 1])
        room.add_microphone([3, 2, 1.5])
        if ray_tracing:
            room.set_ray_tracing(n_rays=1000, time_thres=0.1)
            pra.random.seed(0)
            room.ray_tracing()
        return room

    def compute_rir(room):
        # Reuse the same random tail when ray tracing is enabled.
        pra.random.seed(1)
        room.compute_rir()
        return room.rir[0][0].copy()

    room = make_room(True)
    if coefficients is not None:
        room.set_air_absorption(coefficients)
    with_air = compute_rir(room)

    room.unset_air_absorption()
    without_air = compute_rir(room)
    reference = compute_rir(make_room(False))

    np.testing.assert_allclose(without_air, reference)
    assert not np.isclose(np.sum(with_air**2), np.sum(without_air**2))

    # Disabling twice is harmless, and absorption can be enabled again.
    room.unset_air_absorption()
    np.testing.assert_allclose(compute_rir(room), reference)
    room.set_air_absorption(coefficients)
    np.testing.assert_allclose(compute_rir(room), with_air)


@pytest.mark.parametrize("formula", ["sabine", "eyring"])
def test_unset_air_absorption_rt60(formula):
    room = pra.ShoeBox([5, 4, 3], materials=pra.Material(0.3), air_absorption=True)
    reference = pra.ShoeBox([5, 4, 3], materials=pra.Material(0.3))
    with_air = room.rt60_theory(formula=formula)

    room.unset_air_absorption()

    assert room.air_absorption is None
    assert not room.simulator_state["air_abs_needed"]
    assert room.rt60_theory(formula=formula) == pytest.approx(
        reference.rt60_theory(formula=formula)
    )
    assert with_air != pytest.approx(room.rt60_theory(formula=formula))
