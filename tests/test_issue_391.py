"""
Regression test for issue #391.

``Room.plot()`` raised ``AttributeError: 'MeasuredDirectivity' object has no
attribute 'plot_response'`` when a source or a microphone carried a measured
(SOFA) directivity, so such a room could not be plotted at all.

https://github.com/LCAV/pyroomacoustics/issues/391
"""

import sys
import warnings
from unittest import mock

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    Cardioid,
    DirectionVector,
    MeasuredDirectivity,
    MeasuredDirectivityFile,
    Rotation3D,
)
from pyroomacoustics.doa import GridSphere

FS = 48000
ROOM_DIM = [6.0, 6.0, 2.4]
SRC_POS = [1.52, 0.883, 1.044]
MIC_POS = [2.31, 1.65, 1.163]

# the grids used by the 3-D branch of ``Room.plot``
AZIMUTH_PLOT = np.linspace(start=0, stop=360, num=361, endpoint=True)
COLATITUDE_PLOT = np.linspace(start=0, stop=180, num=180, endpoint=True)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _reporter_source_directivity(orientation=Rotation3D([73, 54], "yz", degrees=True)):
    # The reporter's example: a loudspeaker pattern from the shipped SOFA file.
    return MeasuredDirectivityFile(
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz", fs=FS
    ).get_source_directivity("Yamaha_DXR8", orientation=orientation)


def _measured_mic_directivity():
    # A measured microphone pattern from the shipped SOFA database.
    return MeasuredDirectivityFile("AKG_c480_c414_CUBE", fs=FS).get_mic_directivity(
        "AKG_c480", orientation=Rotation3D([0, 0], "yz", degrees=True)
    )


@pytest.fixture(scope="module")
def source_directivity():
    return _reporter_source_directivity()


@pytest.fixture(scope="module")
def mic_directivity():
    return _measured_mic_directivity()


@pytest.fixture
def reporter_room(source_directivity):
    room = pra.ShoeBox(ROOM_DIM, fs=FS, max_order=2)
    room.add_source(SRC_POS, directivity=source_directivity)
    return room


def _artist_count(ax):
    return len(ax.lines) + len(ax.collections)


def _assert_fig_and_axes(ret):
    assert isinstance(ret, tuple) and len(ret) == 2
    fig, ax = ret
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, Axes)
    return fig, ax


def _polar_radius(ax, offset=(0.0, 0.0)):
    xy = ax.lines[-1].get_xydata()
    return np.hypot(xy[:, 0] - offset[0], xy[:, 1] - offset[1])


def _normalized_magnitude(directivity, irs, freq_bin):
    # normalized magnitude at one bin, as documented by ``plot_response``
    magnitude = np.abs(np.fft.rfft(irs, axis=-1))[:, freq_bin]
    norm = np.abs(np.fft.rfft(directivity._irs, axis=-1))[:, freq_bin].max()
    return magnitude / norm


def _capture_surface(directivity, ax, **kwargs):
    # wrap ``plot_surface`` to read back the X, Y, Z arrays that were drawn
    with mock.patch.object(ax, "plot_surface", wraps=ax.plot_surface) as surface:
        ret = directivity.plot_response(ax=ax, **kwargs)
    assert surface.call_count == 1
    return ret, surface.call_args.args[:3]


def _synthetic_directivity(fs, n_points=200, n_ir=64, zero=False):
    rng = np.random.default_rng(391)
    grid = GridSphere(n_points=n_points)
    irs = np.zeros((n_points, n_ir)) if zero else rng.standard_normal((n_points, n_ir))
    with warnings.catch_warnings():
        # all-zero responses make the energy normalization of the constructor
        # divide by zero; that is existing behaviour and not what is tested
        warnings.simplefilter("ignore", RuntimeWarning)
        return MeasuredDirectivity(
            Rotation3D([0, 0], "yz", degrees=True), grid, irs, fs
        )


# ---------------------------------------------------------------------------
# Room.plot with measured directivities (the reported bug)
# ---------------------------------------------------------------------------


def test_room_plot_with_measured_source_directivity(reporter_room):
    """Issue #391: the default ``room.plot()`` works with a measured source."""
    fig, ax = _assert_fig_and_axes(reporter_room.plot())
    assert isinstance(ax, Axes3D)


def test_room_plot_without_directivity_with_measured_source(reporter_room):
    """``room.plot(plot_directivity=False)`` works with a measured source."""
    fig, ax = _assert_fig_and_axes(reporter_room.plot(plot_directivity=False))
    assert isinstance(ax, Axes3D)


def test_room_plot_with_measured_source_and_mic(reporter_room, mic_directivity):
    """The plot also works when a microphone has a measured directivity."""
    reporter_room.add_microphone(MIC_POS, directivity=mic_directivity)

    fig, ax = _assert_fig_and_axes(reporter_room.plot())
    assert isinstance(ax, Axes3D)

    fig, ax = _assert_fig_and_axes(reporter_room.plot(plot_directivity=False))
    assert isinstance(ax, Axes3D)


def test_room_plot_with_measured_mic_only(mic_directivity):
    """A room whose only directivity is a measured microphone pattern plots."""
    room = pra.ShoeBox(ROOM_DIM, fs=FS, max_order=2)
    room.add_source(SRC_POS)
    room.add_microphone(MIC_POS, directivity=mic_directivity)

    fig, ax = _assert_fig_and_axes(room.plot())
    assert isinstance(ax, Axes3D)


def test_room_plot_draws_measured_directivity(reporter_room, mic_directivity):
    """With ``plot_directivity=True`` the measured patterns are actually drawn."""
    reporter_room.add_microphone(MIC_POS, directivity=mic_directivity)

    fig, ax_without = reporter_room.plot(plot_directivity=False)
    fig, ax_with = reporter_room.plot(plot_directivity=True)

    # one pattern for the source and one for the microphone
    assert _artist_count(ax_with) >= _artist_count(ax_without) + 2


def test_room_plot_with_several_measured_mics(source_directivity, mic_directivity):
    """A microphone array with one measured pattern per microphone plots."""
    room = pra.ShoeBox(ROOM_DIM, fs=FS, max_order=2)
    room.add_source(SRC_POS, directivity=source_directivity)
    mic_positions = np.array([MIC_POS, [3.0, 2.5, 1.2]]).T
    room.add_microphone_array(
        mic_positions, directivity=[mic_directivity, _measured_mic_directivity()]
    )

    fig, ax_without = _assert_fig_and_axes(room.plot(plot_directivity=False))
    fig, ax_with = _assert_fig_and_axes(room.plot())

    assert isinstance(ax_with, Axes3D)
    # one surface for the source and one per microphone
    assert _artist_count(ax_with) >= _artist_count(ax_without) + 3


def test_room_plot_draws_measured_pattern_at_source_position(source_directivity):
    """``Room.plot`` draws the measured pattern centred on the source."""
    room = pra.ShoeBox(ROOM_DIM, fs=FS, max_order=0)
    room.add_source(SRC_POS, directivity=source_directivity)

    with mock.patch.object(
        MeasuredDirectivity, "plot_response", autospec=True
    ) as plot_response:
        plot_response.side_effect = lambda self, *a, **k: k["ax"]
        room.plot()

    assert plot_response.call_count == 1
    kwargs = plot_response.call_args.kwargs
    np.testing.assert_allclose(kwargs["offset"], SRC_POS)
    assert kwargs["degrees"] is True
    np.testing.assert_allclose(kwargs["azimuth"], AZIMUTH_PLOT)
    # the 3-D branch asks for a surface over the whole sphere
    colatitude = np.asarray(kwargs["colatitude"])
    assert colatitude.ndim == 1 and len(colatitude) > 1
    assert colatitude.min() == pytest.approx(0.0)
    assert colatitude.max() == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# MeasuredDirectivity.plot_response: the new public method
# ---------------------------------------------------------------------------


def test_plot_response_polar_returns_given_axes(source_directivity):
    """Without ``colatitude`` the response is drawn on the given axes."""
    fig, ax = plt.subplots()
    before = _artist_count(ax)

    ret = source_directivity.plot_response(
        azimuth=AZIMUTH_PLOT, degrees=True, ax=ax, offset=SRC_POS
    )

    assert ret is ax
    assert _artist_count(ax) > before


def test_plot_response_polar_creates_axes_when_none_given(source_directivity):
    """Without ``ax`` a new figure is created and its axes returned."""
    ret = source_directivity.plot_response(azimuth=AZIMUTH_PLOT, degrees=True)

    assert isinstance(ret, Axes)
    assert _artist_count(ret) > 0


def test_plot_response_3d_returns_given_axes(source_directivity):
    """With ``colatitude`` a 3-D plot is drawn on the given 3-D axes."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    before = _artist_count(ax)

    ret = source_directivity.plot_response(
        azimuth=AZIMUTH_PLOT,
        colatitude=COLATITUDE_PLOT,
        degrees=True,
        ax=ax,
        offset=SRC_POS,
    )

    assert ret is ax
    assert _artist_count(ax) > before


def test_plot_response_3d_creates_axes_when_none_given(source_directivity):
    """With ``colatitude`` and no ``ax`` a new 3-D axes is created and returned."""
    ret = source_directivity.plot_response(
        azimuth=AZIMUTH_PLOT, colatitude=COLATITUDE_PLOT, degrees=True
    )

    assert isinstance(ret, Axes3D)
    assert _artist_count(ret) > 0


def test_plot_response_3d_for_measured_mic(mic_directivity):
    """The method works for measured microphone patterns too."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ret = mic_directivity.plot_response(
        azimuth=AZIMUTH_PLOT,
        colatitude=COLATITUDE_PLOT,
        degrees=True,
        ax=ax,
        offset=MIC_POS,
    )

    assert ret is ax
    assert _artist_count(ax) > 0


def test_plot_response_requires_matplotlib(source_directivity, monkeypatch):
    """Like the analytic one, the method only warns when matplotlib is missing."""
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    with pytest.warns(UserWarning, match="Matplotlib is required"):
        ret = source_directivity.plot_response(azimuth=AZIMUTH_PLOT, degrees=True)

    assert ret is None


# ---------------------------------------------------------------------------
# MeasuredDirectivity.plot_response: what is drawn
# ---------------------------------------------------------------------------


def test_plot_response_polar_matches_get_response_magnitude(source_directivity):
    """The polar radius is the normalized magnitude of ``get_response`` at the bin."""
    freq_bin = 46
    offset = np.array(SRC_POS)
    fig, ax = plt.subplots()

    source_directivity.plot_response(
        azimuth=AZIMUTH_PLOT, degrees=True, ax=ax, offset=offset, freq_bin=freq_bin
    )

    irs = source_directivity.get_response(
        azimuth=AZIMUTH_PLOT, colatitude=np.full(AZIMUTH_PLOT.shape, 90.0), degrees=True
    )
    expected = _normalized_magnitude(source_directivity, irs, freq_bin)
    np.testing.assert_allclose(_polar_radius(ax, offset), expected, atol=1e-12)

    # the curve goes around the offset, i.e. follows the requested azimuths
    xy = ax.lines[-1].get_xydata()
    angle = np.arctan2(xy[:, 1] - offset[1], xy[:, 0] - offset[0])
    nonzero = expected > 1e-12
    expected_angle = np.angle(np.exp(1j * np.radians(AZIMUTH_PLOT[nonzero])))
    np.testing.assert_allclose(angle[nonzero], expected_angle, atol=1e-6)


def test_plot_response_3d_matches_get_response_magnitude(source_directivity):
    """The 3-D surface is the normalized magnitude of ``get_response`` at the bin."""
    freq_bin = 46
    offset = np.array(SRC_POS)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ret, (X, Y, Z) = _capture_surface(
        source_directivity,
        ax,
        azimuth=AZIMUTH_PLOT,
        colatitude=COLATITUDE_PLOT,
        degrees=True,
        offset=offset,
        freq_bin=freq_bin,
    )

    assert ret is ax
    assert X.shape == Y.shape == Z.shape == (len(COLATITUDE_PLOT), len(AZIMUTH_PLOT))
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(Y)) and np.all(np.isfinite(Z))

    AZI, COL = np.meshgrid(AZIMUTH_PLOT, COLATITUDE_PLOT)
    irs = source_directivity.get_response(
        azimuth=AZI.flatten(), colatitude=COL.flatten(), degrees=True
    )
    expected = _normalized_magnitude(source_directivity, irs, freq_bin).reshape(
        AZI.shape
    )
    radius = np.sqrt((X - offset[0]) ** 2 + (Y - offset[1]) ** 2 + (Z - offset[2]) ** 2)
    np.testing.assert_allclose(radius, expected, atol=1e-12)

    # the pattern is normalized: it never exceeds one and reaches one
    assert radius.max() == pytest.approx(1.0)
    assert radius.min() >= 0.0


def test_plot_response_radians(source_directivity):
    """``degrees=False`` draws the same curve and surface for angles in radians."""
    fig, (ax_deg, ax_rad) = plt.subplots(1, 2)
    source_directivity.plot_response(azimuth=AZIMUTH_PLOT, degrees=True, ax=ax_deg)
    source_directivity.plot_response(
        azimuth=np.radians(AZIMUTH_PLOT), degrees=False, ax=ax_rad
    )
    np.testing.assert_allclose(
        ax_deg.lines[-1].get_xydata(), ax_rad.lines[-1].get_xydata()
    )

    fig = plt.figure()
    ax_deg = fig.add_subplot(1, 2, 1, projection="3d")
    ax_rad = fig.add_subplot(1, 2, 2, projection="3d")
    _, deg = _capture_surface(
        source_directivity,
        ax_deg,
        azimuth=AZIMUTH_PLOT,
        colatitude=COLATITUDE_PLOT,
        degrees=True,
    )
    _, rad = _capture_surface(
        source_directivity,
        ax_rad,
        azimuth=np.radians(AZIMUTH_PLOT),
        colatitude=np.radians(COLATITUDE_PLOT),
        degrees=False,
    )
    for a, b in zip(deg, rad):
        np.testing.assert_allclose(a, b)


def test_plot_response_3d_sets_axis_labels_only(source_directivity):
    """The 3-D branch labels the axes but does not overwrite the axes title."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_title("my room")

    source_directivity.plot_response(
        azimuth=AZIMUTH_PLOT, colatitude=COLATITUDE_PLOT, degrees=True, ax=ax
    )

    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_zlabel() == "z"
    assert ax.get_title() == "my room"


def test_plot_response_follows_set_orientation():
    """The drawn pattern comes from the rotated grid, so it follows the orientation."""
    directivity = _reporter_source_directivity(Rotation3D([0, 0], "yz", degrees=True))
    fig, (ax_before, ax_after, ax_fresh) = plt.subplots(1, 3)

    directivity.plot_response(azimuth=AZIMUTH_PLOT, ax=ax_before)
    directivity.set_orientation(Rotation3D([0, 90], "yz", degrees=True))
    directivity.plot_response(azimuth=AZIMUTH_PLOT, ax=ax_after)

    fresh = _reporter_source_directivity(Rotation3D([0, 90], "yz", degrees=True))
    fresh.plot_response(azimuth=AZIMUTH_PLOT, ax=ax_fresh)

    assert not np.allclose(
        ax_before.lines[-1].get_xydata(), ax_after.lines[-1].get_xydata()
    )
    np.testing.assert_allclose(
        ax_after.lines[-1].get_xydata(), ax_fresh.lines[-1].get_xydata()
    )


def test_plot_response_rotation_rotates_polar_pattern():
    """Rotating by 90 degrees about z shifts the horizontal pattern by 90 degrees."""
    front = _reporter_source_directivity(Rotation3D([0, 0], "yz", degrees=True))
    turned = _reporter_source_directivity(Rotation3D([0, 90], "yz", degrees=True))
    # the grid is measured every 10 degrees: sample off the cell boundaries at
    # 5, 15, ... degrees where nearest-neighbour ties could flip
    azimuth = np.arange(0.5, 360, 1.0)
    fig, (ax_front, ax_turned) = plt.subplots(1, 2)

    front.plot_response(azimuth=azimuth, ax=ax_front, freq_bin=46)
    turned.plot_response(azimuth=azimuth, ax=ax_turned, freq_bin=46)

    np.testing.assert_allclose(
        np.roll(_polar_radius(ax_front), 90), _polar_radius(ax_turned), atol=1e-9
    )


def test_plot_response_with_interpolated_directivity():
    """A directivity read with spherical harmonics interpolation plots too."""
    directivity = MeasuredDirectivityFile(
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz",
        fs=FS,
        interp_order=12,
        interp_n_points=1000,
    ).get_source_directivity(
        "Yamaha_DXR8", orientation=Rotation3D([73, 54], "yz", degrees=True)
    )
    fig = plt.figure()
    ax_polar = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    assert directivity.plot_response(azimuth=AZIMUTH_PLOT, ax=ax_polar) is ax_polar
    assert np.all(np.isfinite(_polar_radius(ax_polar)))
    assert (
        directivity.plot_response(
            azimuth=AZIMUTH_PLOT, colatitude=COLATITUDE_PLOT, ax=ax_3d
        )
        is ax_3d
    )
    assert _artist_count(ax_3d) == 1


def test_plot_response_accepts_list_and_scalar_azimuth(source_directivity):
    """Plain Python lists and a single angle are accepted like arrays."""
    fig, (ax_list, ax_scalar) = plt.subplots(1, 2)

    source_directivity.plot_response(azimuth=[0, 90, 180, 270], ax=ax_list)
    assert ax_list.lines[-1].get_xydata().shape == (4, 2)

    source_directivity.plot_response(azimuth=45.0, ax=ax_scalar)
    xy = ax_scalar.lines[-1].get_xydata()
    assert xy.shape == (1, 2)
    assert np.all(np.isfinite(xy))


# ---------------------------------------------------------------------------
# MeasuredDirectivity.plot_response: the ``freq_bin`` argument
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "freq_bin",
    [np.int64(46), np.float32(46 / 2230), np.float64(46 / 2230), 46 / 2230],
    ids=["np.int64", "np.float32", "np.float64", "float"],
)
def test_plot_response_freq_bin_int_and_float_equivalent(source_directivity, freq_bin):
    """``freq_bin`` follows the convention of ``plot``: int bin or float ``f / fs``."""
    assert source_directivity._irs.shape[-1] == 2230
    fig, (ax_int, ax_other) = plt.subplots(1, 2)

    source_directivity.plot_response(azimuth=AZIMUTH_PLOT, ax=ax_int, freq_bin=46)
    source_directivity.plot_response(
        azimuth=AZIMUTH_PLOT, ax=ax_other, freq_bin=freq_bin
    )

    np.testing.assert_allclose(
        ax_int.lines[-1].get_xydata(), ax_other.lines[-1].get_xydata()
    )


def test_plot_response_different_bins_differ(source_directivity):
    """The measured pattern depends on frequency, so different bins draw differently."""
    fig, (ax_low, ax_high) = plt.subplots(1, 2)

    source_directivity.plot_response(azimuth=AZIMUTH_PLOT, ax=ax_low, freq_bin=46)
    source_directivity.plot_response(azimuth=AZIMUTH_PLOT, ax=ax_high, freq_bin=557)

    assert not np.allclose(
        ax_low.lines[-1].get_xydata(), ax_high.lines[-1].get_xydata()
    )


def test_plot_response_default_bin_is_closest_to_1khz(source_directivity):
    """Omitting ``freq_bin`` draws the documented default bin (closest to 1 kHz)."""
    n_fft = source_directivity._irs.shape[-1]
    default_bin = int(round(1000.0 / FS * n_fft))
    fig, (ax_default, ax_explicit) = plt.subplots(1, 2)

    source_directivity.plot_response(azimuth=AZIMUTH_PLOT, ax=ax_default)
    source_directivity.plot_response(
        azimuth=AZIMUTH_PLOT, ax=ax_explicit, freq_bin=default_bin
    )

    np.testing.assert_allclose(
        ax_default.lines[-1].get_xydata(), ax_explicit.lines[-1].get_xydata()
    )


def test_plot_response_default_bin_below_nyquist_for_low_fs():
    """When 1 kHz is above Nyquist, the default bin still lies in the spectrum."""
    n_ir = 64
    directivity = _synthetic_directivity(fs=1000, n_ir=n_ir)
    fig, (ax_default, ax_last) = plt.subplots(1, 2)

    ret = directivity.plot_response(azimuth=AZIMUTH_PLOT, ax=ax_default)

    assert ret is ax_default
    radius = _polar_radius(ax_default)
    assert np.all(np.isfinite(radius))
    assert 0.0 < radius.max() <= 1.0 + 1e-12

    # the bin closest to 1 kHz is clipped to the last bin of the spectrum
    directivity.plot_response(azimuth=AZIMUTH_PLOT, ax=ax_last, freq_bin=n_ir // 2)
    np.testing.assert_allclose(
        ax_default.lines[-1].get_xydata(), ax_last.lines[-1].get_xydata()
    )


def test_plot_response_dc_bin_is_finite(source_directivity):
    """Bin 0 (nearly silent for a loudspeaker) draws a finite curve."""
    fig, ax = plt.subplots()

    source_directivity.plot_response(azimuth=AZIMUTH_PLOT, ax=ax, freq_bin=0)

    radius = _polar_radius(ax)
    assert np.all(np.isfinite(radius))
    assert radius.max() <= 1.0 + 1e-12


def test_plot_response_all_zero_responses_draw_a_point():
    """Silent responses draw a finite curve at the offset instead of dividing by zero."""
    directivity = _synthetic_directivity(fs=16000, zero=True)
    offset = [1.0, 2.0, 3.0]
    fig, ax = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        directivity.plot_response(azimuth=AZIMUTH_PLOT, ax=ax, offset=offset)

    radius = _polar_radius(ax, offset)
    assert np.all(np.isfinite(radius))
    np.testing.assert_allclose(radius, 0.0)


@pytest.mark.parametrize(
    "freq_bin, error",
    [
        (1116, ValueError),  # n_fft // 2 + 1 for the 2230-sample responses
        (2230, ValueError),
        (-1, ValueError),
        (0.75, ValueError),  # above Nyquist as a normalized frequency
        ("46", TypeError),
        (True, TypeError),  # bool is an int but not a bin index
        (np.bool_(True), TypeError),
        (None.__class__, TypeError),
    ],
)
def test_plot_response_rejects_bad_freq_bin(source_directivity, freq_bin, error):
    """Bins outside the one-sided spectrum or of the wrong type are rejected."""
    with pytest.raises(error):
        source_directivity.plot_response(azimuth=AZIMUTH_PLOT, freq_bin=freq_bin)


# ---------------------------------------------------------------------------
# Untouched behaviour
# ---------------------------------------------------------------------------


def test_measured_directivity_plot_still_works(source_directivity):
    """The existing full-pattern ``plot`` is untouched."""
    ret = source_directivity.plot()
    assert isinstance(ret, Axes3D)


def test_analytic_plot_response_polar_unchanged():
    """The analytic polar plot draws exactly ``get_response`` magnitude, offset."""
    directivity = Cardioid(DirectionVector(azimuth=30, colatitude=60, degrees=True))
    offset = np.array([1.0, 2.0, 3.0])
    azimuth = np.radians(AZIMUTH_PLOT)

    fig, ax = plt.subplots()
    ret = directivity.plot_response(
        azimuth=AZIMUTH_PLOT, degrees=True, ax=ax, offset=offset
    )

    assert ret is ax
    assert len(ax.lines) == 1
    resp = directivity.get_response(azimuth=azimuth, magnitude=True, degrees=False)
    np.testing.assert_allclose(
        ax.lines[0].get_xdata(), resp * np.cos(azimuth) + offset[0]
    )
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), resp * np.sin(azimuth) + offset[1]
    )


def test_analytic_plot_response_3d_unchanged():
    """The analytic 3-D plot still draws one surface and returns the axes."""
    directivity = Cardioid(DirectionVector(azimuth=30, colatitude=60, degrees=True))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ret = directivity.plot_response(
        azimuth=AZIMUTH_PLOT,
        colatitude=COLATITUDE_PLOT,
        degrees=True,
        ax=ax,
        offset=[1.0, 2.0, 3.0],
    )

    assert ret is ax
    assert len(ax.collections) == 1
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_zlabel() == "z"


def test_room_plot_with_analytic_directivity_unchanged():
    """Rooms with analytic directivities keep plotting as before."""
    room = pra.ShoeBox(ROOM_DIM, fs=FS, max_order=2)
    room.add_source(
        SRC_POS,
        directivity=Cardioid(DirectionVector(azimuth=0, colatitude=90, degrees=True)),
    )
    room.add_microphone(
        MIC_POS,
        directivity=Cardioid(DirectionVector(azimuth=180, colatitude=90, degrees=True)),
    )

    fig, ax = _assert_fig_and_axes(room.plot())
    assert isinstance(ax, Axes3D)
