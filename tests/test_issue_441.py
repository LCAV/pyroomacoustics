"""
Regression tests for issue #441.

https://github.com/LCAV/pyroomacoustics/issues/441

In a non-convex room simulated with a low ``max_order``, a microphone may see
no image source of a given source at all. ``compute_ism_rir`` used to raise
``ValueError: zero-size array to reduction operation maximum`` in that case.
The expected behaviour is a silent impulse response for the dead channel, and
``Room.compute_rir`` and ``Room.simulate`` must keep working.
"""

import numpy as np
import pytest

import pyroomacoustics as pra
from pyroomacoustics.parameters import constants
from pyroomacoustics.simulation.ism import compute_ism_rir

_FS = 16000
_MAX_ORDER = 2
_HEIGHT = 3.0
_SIGNAL_LENGTH = _FS // 4

# U-shaped floor plan: two 2 m wide arms joined by a 1 m deep corridor along
# the south wall. The source sits at the top of the west arm and the "dead"
# microphone at the top of the east arm. With max_order=2 no image source of
# the source is visible from there. A plain L-shape is not enough: some
# second-order images of the source remain visible around the inner corner.
_U_CORNERS = np.array(
    [[0, 0], [10, 0], [10, 10], [8, 10], [8, 1], [2, 1], [2, 10], [0, 10]]
).T

_SRC_LOC = [1.0, 9.5, 1.5]
_DEAD_MIC_LOC = [9.0, 9.5, 1.5]  # no line of sight, no visible image source
_LIVE_MIC_LOC = [1.0, 2.0, 1.5]  # same arm as the source, direct path exists

SRC = 0
DEAD_MIC = 0
LIVE_MIC = 1


def _make_room(
    mic_locs,
    ray_tracing=False,
    material=None,
    air_absorption=False,
    mic_directivity=None,
    src_directivity=None,
):
    mat = material if material is not None else pra.Material(energy_absorption=0.2)
    room = pra.Room.from_corners(
        _U_CORNERS,
        fs=_FS,
        max_order=_MAX_ORDER,
        materials=mat,
        ray_tracing=ray_tracing,
        air_absorption=air_absorption,
    )
    room.extrude(_HEIGHT, materials=mat)
    if ray_tracing:
        room.set_ray_tracing(n_rays=2000)

    rng = np.random.default_rng(441)
    room.add_source(
        _SRC_LOC,
        signal=rng.standard_normal(_SIGNAL_LENGTH),
        directivity=src_directivity,
    )
    for loc in mic_locs:
        room.add_microphone(loc, directivity=mic_directivity)

    return room


def _assert_silent_rir(rir):
    assert isinstance(rir, np.ndarray)
    assert rir.ndim == 1
    assert rir.shape[0] > 0
    assert np.issubdtype(rir.dtype, np.floating)
    assert np.all(rir == 0.0)


def _call_compute_ism_rir(room, s, m):
    src = room.sources[s]
    return compute_ism_rir(
        src,
        room.mic_array.R[:, m],
        room.mic_array.directivity[m],
        src.directions[m, :, :],
        room.visibility[s][m, :],
        constants.get("frac_delay_length"),
        room.c,
        room.fs,
        room.octave_bands,
        min_phase=room.min_phase,
        air_abs_coeffs=room.air_absorption,
    )


@pytest.fixture
def room():
    return _make_room([_DEAD_MIC_LOC, _LIVE_MIC_LOC])


def test_dead_mic_has_no_visible_image_source(room):
    """The geometry must actually exercise the issue: the dead microphone
    sees no image source at all, while the live one sees at least one."""
    room.image_source_model()

    assert room.visibility[SRC].shape[0] == 2
    assert not np.any(room.visibility[SRC][DEAD_MIC])
    assert np.any(room.visibility[SRC][LIVE_MIC])


def test_compute_ism_rir_returns_silent_rir_for_empty_mask(room):
    """compute_ism_rir returns a non-empty 1-D float array of zeros, with the
    same dtype as a normal return, when is_visible selects nothing."""
    room.image_source_model()
    assert not np.any(room.visibility[SRC][DEAD_MIC])

    live = _call_compute_ism_rir(room, SRC, LIVE_MIC)
    dead = _call_compute_ism_rir(room, SRC, DEAD_MIC)

    assert isinstance(dead, np.ndarray)
    assert dead.ndim == 1
    assert dead.shape[0] > 0
    assert np.issubdtype(dead.dtype, np.floating)
    assert dead.dtype == live.dtype
    assert np.all(dead == 0.0)


def test_compute_rir_succeeds_and_dead_channel_rir_is_zero(room):
    """Room.compute_rir works and the dead channel's RIR is all zeros."""
    room.compute_rir()

    assert not np.any(room.visibility[SRC][DEAD_MIC])

    rir = room.rir[DEAD_MIC][SRC]
    assert isinstance(rir, np.ndarray)
    assert rir.ndim == 1
    assert rir.shape[0] > 0
    assert np.issubdtype(rir.dtype, np.floating)
    assert np.all(rir == 0.0)


def test_live_channel_rir_is_nonzero(room):
    """A microphone that does see the source gets a normal, non-zero RIR."""
    room.compute_rir()

    assert np.any(room.visibility[SRC][LIVE_MIC])

    rir = room.rir[LIVE_MIC][SRC]
    assert rir.ndim == 1
    assert rir.shape[0] > 0
    assert np.all(np.isfinite(rir))
    assert np.any(rir != 0.0)


def test_live_channel_rir_unchanged_by_dead_channel(room):
    """The RIR of a pair with visible image sources is bit-for-bit the same
    whether or not a dead channel is present in the room."""
    room.compute_rir()

    room_live_only = _make_room([_LIVE_MIC_LOC])
    room_live_only.compute_rir()

    np.testing.assert_array_equal(room.rir[LIVE_MIC][SRC], room_live_only.rir[0][SRC])


def test_simulate_succeeds_and_dead_channel_is_silent(room):
    """Room.simulate works; the dead microphone records silence and the live
    microphone records a non-zero signal."""
    room.simulate()

    signals = room.mic_array.signals
    assert signals.shape[0] == 2
    assert signals.shape[1] > 0
    assert np.all(np.isfinite(signals))
    assert np.all(signals[DEAD_MIC] == 0.0)
    assert np.any(signals[LIVE_MIC] != 0.0)


def test_room_with_only_dead_mic_gets_silent_rir():
    """When no microphone in the room sees any image source,
    Room.image_source_model stores an all-zero integer array as the visibility
    instead of a boolean mask. The dead channel must still be silent rather
    than rendering the direct path through the walls."""
    room = _make_room([_DEAD_MIC_LOC])
    room.image_source_model()

    assert room.visibility[SRC].shape[0] == 1
    assert not np.any(room.visibility[SRC][0])

    room.simulate()

    _assert_silent_rir(room.rir[0][SRC])
    assert np.all(room.mic_array.signals[0] == 0.0)


def test_room_with_two_dead_mics_gets_silent_rirs():
    """Same as above with two microphones, both without any visible image
    source."""
    other_dead = [9.0, 8.0, 1.5]  # same arm as the dead microphone
    room = _make_room([_DEAD_MIC_LOC, other_dead])
    room.image_source_model()

    assert room.visibility[SRC].shape[0] == 2
    assert not np.any(room.visibility[SRC])

    room.simulate()

    for m in range(2):
        _assert_silent_rir(room.rir[m][SRC])
    assert np.all(room.mic_array.signals == 0.0)


def test_compute_rir_with_ray_tracing_on_dead_channel():
    """With ray tracing on, compute_rir still works for the dead channel and
    returns a non-empty 1-D finite RIR (the ray-tracing part is kept)."""
    room = _make_room([_DEAD_MIC_LOC, _LIVE_MIC_LOC], ray_tracing=True)
    room.compute_rir()

    assert not np.any(room.visibility[SRC][DEAD_MIC])

    rir = room.rir[DEAD_MIC][SRC]
    assert rir.ndim == 1
    assert rir.shape[0] > 0
    assert np.all(np.isfinite(rir))

    rir_live = room.rir[LIVE_MIC][SRC]
    assert rir_live.ndim == 1
    assert np.any(rir_live != 0.0)


def test_compute_ism_rir_empty_integer_index():
    """An empty integer index array selects nothing as well; the early return
    must not depend on the mask being boolean."""
    room = _make_room([_DEAD_MIC_LOC, _LIVE_MIC_LOC])
    room.image_source_model()
    src = room.sources[SRC]

    dead = compute_ism_rir(
        src,
        room.mic_array.R[:, DEAD_MIC],
        None,
        src.directions[DEAD_MIC, :, :],
        np.array([], dtype=int),
        constants.get("frac_delay_length"),
        room.c,
        room.fs,
        room.octave_bands,
    )

    _assert_silent_rir(dead)
    assert dead.dtype == np.float32


def test_compute_ism_rir_dead_channel_linear_phase(room):
    """The silent RIR is also returned with min_phase=False."""
    room.image_source_model()
    assert not np.any(room.visibility[SRC][DEAD_MIC])
    src = room.sources[SRC]

    dead = compute_ism_rir(
        src,
        room.mic_array.R[:, DEAD_MIC],
        None,
        src.directions[DEAD_MIC, :, :],
        room.visibility[SRC][DEAD_MIC, :],
        constants.get("frac_delay_length"),
        room.c,
        room.fs,
        room.octave_bands,
        min_phase=False,
    )

    _assert_silent_rir(dead)


def test_dead_channel_with_frequency_dependent_material():
    """Octave-band dampings (several rows in src.damping) do not change the
    dead-channel behaviour, and the live channel still gets a non-zero RIR."""
    room = _make_room(
        [_DEAD_MIC_LOC, _LIVE_MIC_LOC], material=pra.Material("hard_surface")
    )
    room.compute_rir()

    assert not np.any(room.visibility[SRC][DEAD_MIC])
    _assert_silent_rir(room.rir[DEAD_MIC][SRC])
    assert np.any(room.rir[LIVE_MIC][SRC] != 0.0)


def test_dead_channel_with_air_absorption():
    """The early return happens before air absorption is applied; with air
    absorption on, the dead channel is still silent and finite."""
    room = _make_room([_DEAD_MIC_LOC, _LIVE_MIC_LOC], air_absorption=True)
    room.compute_rir()

    assert not np.any(room.visibility[SRC][DEAD_MIC])
    _assert_silent_rir(room.rir[DEAD_MIC][SRC])
    assert np.all(np.isfinite(room.rir[LIVE_MIC][SRC]))
    assert np.any(room.rir[LIVE_MIC][SRC] != 0.0)


def test_dead_channel_with_directivities():
    """The early return happens before the microphone and source directivity
    branches; both directivities being set must not break the dead channel."""
    pattern = pra.directivities.Cardioid(
        pra.directivities.DirectionVector(azimuth=0, colatitude=90, degrees=True)
    )
    room = _make_room(
        [_DEAD_MIC_LOC, _LIVE_MIC_LOC],
        mic_directivity=pattern,
        src_directivity=pattern,
    )
    room.compute_rir()

    assert not np.any(room.visibility[SRC][DEAD_MIC])
    _assert_silent_rir(room.rir[DEAD_MIC][SRC])
    assert np.any(room.rir[LIVE_MIC][SRC] != 0.0)


def test_microphone_outside_room_gets_silent_rir():
    """A microphone outside the room is another way to get an all-false
    visibility row (Room.image_source_model clears it). It must yield a silent
    RIR and silence in simulate() rather than a crash."""
    outside = [20.0, 20.0, 1.5]
    room = _make_room([_LIVE_MIC_LOC, outside])
    room.image_source_model()
    assert not room.is_inside(np.array(outside))
    assert not np.any(room.visibility[SRC][1])

    room.simulate()

    _assert_silent_rir(room.rir[1][SRC])
    assert np.all(room.mic_array.signals[1] == 0.0)
    assert np.any(room.mic_array.signals[0] != 0.0)


def test_dead_mic_hears_second_visible_source():
    """A microphone that is dead for one source but sees another source gets
    a silent RIR for the first and a normal one for the second, and its
    simulated signal is the non-zero mix."""
    room = _make_room([_DEAD_MIC_LOC, _LIVE_MIC_LOC])
    second_src = [9.0, 8.0, 1.5]  # same arm as the dead mic
    rng = np.random.default_rng(442)
    room.add_source(second_src, signal=rng.standard_normal(_SIGNAL_LENGTH))

    room.simulate()

    assert not np.any(room.visibility[SRC][DEAD_MIC])
    assert np.any(room.visibility[1][DEAD_MIC])
    _assert_silent_rir(room.rir[DEAD_MIC][SRC])
    assert np.any(room.rir[DEAD_MIC][1] != 0.0)
    assert np.any(room.mic_array.signals[DEAD_MIC] != 0.0)
    assert np.all(np.isfinite(room.mic_array.signals))
