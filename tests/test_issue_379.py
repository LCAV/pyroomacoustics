"""
# Test for Issue 379

[issue #379](https://github.com/LCAV/pyroomacoustics/issues/379)

Assigning ``room.air_absorption = True`` used to be silently accepted and the
simulation failed much later with ``TypeError: 'bool' object is not
subscriptable`` when the coefficients were indexed.

``Room.air_absorption`` is now a read-only property. Assigning to it raises an
``AttributeError`` pointing the user to ``set_air_absorption()`` and
``unset_air_absorption()``. Reading it keeps working as before.
"""

import numpy as np
import pytest

import pyroomacoustics as pra

_FS = 16000
_MAX_ORDER = 3
_ROOM_DIM = [3.0, 4.0, 5.0]
_SRC_LOC = [1.001, 0.999, 1.002]
_MIC_LOC = [2.0, 3.0, 4.0]
_MAT = pra.Material(energy_absorption=0.1)


def _shoebox(**kwargs):
    room = pra.ShoeBox(
        _ROOM_DIM, fs=_FS, max_order=_MAX_ORDER, materials=_MAT, **kwargs
    )
    room.add_source(_SRC_LOC).add_microphone(_MIC_LOC)
    return room


def _from_corners(**kwargs):
    corners = np.array(
        [[0, 0], [_ROOM_DIM[0], 0], [_ROOM_DIM[0], _ROOM_DIM[1]], [0, _ROOM_DIM[1]]]
    ).T
    room = pra.Room.from_corners(
        corners, fs=_FS, max_order=_MAX_ORDER, materials=_MAT, **kwargs
    )
    room.extrude(height=_ROOM_DIM[2], materials=_MAT)
    room.add_source(_SRC_LOC).add_microphone(_MIC_LOC)
    return room


def _anechoic(**kwargs):
    room = pra.AnechoicRoom(fs=_FS, **kwargs)
    room.add_source(_SRC_LOC).add_microphone(_MIC_LOC)
    return room


_ROOM_FACTORIES = [
    pytest.param(_shoebox, id="shoebox"),
    pytest.param(_from_corners, id="from_corners"),
    pytest.param(_anechoic, id="anechoic"),
]


def test_air_absorption_is_read_only_property_on_room_class():
    assert isinstance(pra.Room.air_absorption, property)


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_assigning_true_raises_attribute_error_naming_set_air_absorption(make_room):
    room = make_room()

    with pytest.raises(AttributeError, match="set_air_absorption") as excinfo:
        room.air_absorption = True

    # The message must also tell the user how to turn air absorption off.
    assert "unset_air_absorption" in str(excinfo.value)

    # The failed assignment must not have changed anything.
    assert room.air_absorption is None


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_assigning_true_after_set_air_absorption_raises_and_keeps_coefficients(
    make_room,
):
    room = make_room()
    room.set_air_absorption()
    coeffs_before = room.air_absorption

    with pytest.raises(AttributeError, match="set_air_absorption"):
        room.air_absorption = True

    assert room.air_absorption is coeffs_before


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_assigning_coefficient_array_raises_attribute_error(make_room):
    room = make_room()

    with pytest.raises(AttributeError, match="set_air_absorption"):
        room.air_absorption = np.ones(7) * 0.5

    assert room.air_absorption is None


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_assigning_true_does_not_break_later_simulation(make_room):
    room = make_room()

    with pytest.raises(AttributeError):
        room.air_absorption = True

    # Before the fix, this is where the error used to show up as a TypeError.
    room.compute_rir()
    rir = room.rir[0][0]
    assert np.all(np.isfinite(rir))
    assert np.any(rir != 0.0)


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_property_reads_none_before_set_air_absorption(make_room):
    room = make_room()
    assert room.air_absorption is None


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_property_reads_coefficients_after_set_air_absorption(make_room):
    room = make_room()
    room.set_air_absorption()

    coeffs = room.air_absorption
    expected = room.octave_bands(**room.physics.get_air_absorption())

    assert coeffs is not None
    assert isinstance(coeffs, np.ndarray)
    assert len(coeffs) == 7
    assert np.all(np.isfinite(coeffs))
    assert np.all(coeffs > 0.0)
    assert np.allclose(coeffs, expected)


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_property_reads_given_coefficients_after_set_air_absorption(make_room):
    room = make_room()
    given = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    room.set_air_absorption(given)

    coeffs = room.air_absorption
    assert len(coeffs) == 7
    assert np.allclose(coeffs, given)


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_property_returns_same_object_on_repeated_reads(make_room):
    room = make_room()
    room.set_air_absorption()

    assert room.air_absorption is room.air_absorption


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_construction_with_air_absorption_true_turns_it_on(make_room):
    room = make_room(air_absorption=True)

    coeffs = room.air_absorption
    expected = room.octave_bands(**room.physics.get_air_absorption())

    assert coeffs is not None
    assert len(coeffs) == 7
    assert np.allclose(coeffs, expected)
    assert room.simulator_state["air_abs_needed"] is True


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_construction_with_air_absorption_false_leaves_it_off(make_room):
    room = make_room(air_absorption=False)

    assert room.air_absorption is None
    assert not room.simulator_state["air_abs_needed"]


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_unset_air_absorption_still_clears_the_simulator_flag(make_room):
    room = make_room(air_absorption=True)
    assert room.simulator_state["air_abs_needed"] is True

    room.unset_air_absorption()

    assert room.simulator_state["air_abs_needed"] is False

    # The simulation still runs after turning air absorption off.
    room.compute_rir()
    rir = room.rir[0][0]
    assert np.all(np.isfinite(rir))
    assert np.any(rir != 0.0)


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_unset_air_absorption_keeps_coefficients_readable(make_room):
    # unset_air_absorption only flips the simulator flag; the coefficients
    # computed by set_air_absorption stay available for reading, as before.
    room = make_room()
    room.set_air_absorption()
    coeffs = room.air_absorption

    room.unset_air_absorption()

    assert room.air_absorption is coeffs


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_set_air_absorption_after_unset_turns_it_on_again(make_room):
    room = make_room(air_absorption=True)
    room.unset_air_absorption()
    room.set_air_absorption()

    assert room.simulator_state["air_abs_needed"] is True
    assert room.air_absorption is not None
    assert len(room.air_absorption) == 7


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_simulation_with_air_absorption_gives_finite_non_zero_rir(make_room):
    room = make_room(air_absorption=True)
    room.compute_rir()

    rir = room.rir[0][0]
    assert rir.ndim == 1
    assert len(rir) > 0
    assert np.all(np.isfinite(rir))
    assert np.any(rir != 0.0)


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_simulation_after_set_air_absorption_gives_finite_non_zero_rir(make_room):
    room = make_room()
    room.set_air_absorption()
    room.compute_rir()

    rir = room.rir[0][0]
    assert np.all(np.isfinite(rir))
    assert np.any(rir != 0.0)


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_constructor_flag_and_set_air_absorption_give_the_same_rir(make_room):
    room_flag = make_room(air_absorption=True)
    room_flag.compute_rir()

    room_set = make_room()
    room_set.set_air_absorption()
    room_set.compute_rir()

    assert np.allclose(room_flag.rir[0][0], room_set.rir[0][0])


def test_air_absorption_changes_the_rir_of_a_reflective_room():
    # Guards that air absorption is really applied in the simulation and
    # not silently dropped by the property.
    room_off = _shoebox()
    room_off.compute_rir()

    room_on = _shoebox(air_absorption=True)
    room_on.compute_rir()

    rir_off = room_off.rir[0][0]
    rir_on = room_on.rir[0][0]
    n = min(len(rir_off), len(rir_on))
    assert not np.allclose(rir_off[:n], rir_on[:n])


def test_rt60_theory_still_reads_air_absorption():
    room = _shoebox(air_absorption=True)
    rt60 = room.rt60_theory()
    assert np.isfinite(rt60)
    assert rt60 > 0.0


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_assigning_none_raises_attribute_error(make_room):
    # Even "resetting" the attribute is rejected; unset_air_absorption() is
    # the supported way to turn air absorption off.
    room = make_room(air_absorption=True)
    coeffs = room.air_absorption

    with pytest.raises(AttributeError, match="unset_air_absorption"):
        room.air_absorption = None

    assert room.air_absorption is coeffs
    assert room.simulator_state["air_abs_needed"] is True


@pytest.mark.parametrize("make_room", _ROOM_FACTORIES)
def test_deleting_air_absorption_raises_and_keeps_coefficients(make_room):
    room = make_room(air_absorption=True)
    coeffs = room.air_absorption

    with pytest.raises(AttributeError, match="set_air_absorption") as excinfo:
        del room.air_absorption

    assert "unset_air_absorption" in str(excinfo.value)
    assert room.air_absorption is coeffs
    assert room.simulator_state["air_abs_needed"] is True


def test_error_message_names_the_actual_room_class():
    with pytest.raises(AttributeError, match="ShoeBox.air_absorption"):
        _shoebox().air_absorption = True

    with pytest.raises(AttributeError, match="AnechoicRoom.air_absorption"):
        _anechoic().air_absorption = True


def test_subclass_assigning_in_init_gets_the_guiding_error():
    class MyRoom(pra.ShoeBox):
        def __init__(self):
            super().__init__(_ROOM_DIM, fs=_FS, max_order=_MAX_ORDER, materials=_MAT)
            self.air_absorption = True

    with pytest.raises(AttributeError, match="set_air_absorption"):
        MyRoom()


def test_property_is_safe_on_a_half_initialized_instance():
    # Property getters that raise AttributeError are swallowed by hasattr, so
    # the getter must not raise when the private attribute is not yet there.
    bare = pra.Room.__new__(pra.Room)
    assert hasattr(bare, "air_absorption")
    assert bare.air_absorption is None


def test_property_reads_coefficients_with_custom_center_freqs():
    room = _shoebox()
    room.set_air_absorption([0.001, 0.002, 0.004], center_freqs=[250, 1000, 4000])

    coeffs = room.air_absorption
    assert len(coeffs) == 7
    assert np.all(np.isfinite(coeffs))
    assert np.all(coeffs > 0.0)


def test_ray_tracing_simulation_with_air_absorption_runs():
    room = _shoebox(air_absorption=True)
    room.set_ray_tracing(n_rays=2000, receiver_radius=0.5)
    room.compute_rir()

    rir = room.rir[0][0]
    assert np.all(np.isfinite(rir))
    assert np.any(rir != 0.0)


def test_two_dimensional_room_with_air_absorption_runs():
    corners = np.array(
        [[0, 0], [_ROOM_DIM[0], 0], [_ROOM_DIM[0], _ROOM_DIM[1]], [0, _ROOM_DIM[1]]]
    ).T
    room = pra.Room.from_corners(
        corners, fs=_FS, max_order=_MAX_ORDER, materials=_MAT, air_absorption=True
    )
    room.add_source(_SRC_LOC[:2]).add_microphone(_MIC_LOC[:2])

    assert len(room.air_absorption) == 7

    with pytest.raises(AttributeError, match="set_air_absorption"):
        room.air_absorption = True

    room.compute_rir()
    rir = room.rir[0][0]
    assert np.all(np.isfinite(rir))
    assert np.any(rir != 0.0)
