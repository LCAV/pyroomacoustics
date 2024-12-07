import numpy as np
import pytest

import pyroomacoustics as pra

_FS = 16000

mic_dir0 = pra.FigureEight(
    orientation=pra.DirectionVector(azimuth=90, colatitude=15, degrees=True)
)
mic_dir1 = pra.FigureEight(
    orientation=pra.DirectionVector(azimuth=180, colatitude=15, degrees=True)
)


@pytest.mark.parametrize("shape", ((1, 2, 3), (10, 2), (1, 10), (10,)))
def test_microphone_array_invalid_shape(shape):

    locs = np.ones(shape)
    with pytest.raises(ValueError):
        pra.MicrophoneArray(locs, fs=_FS)


@pytest.mark.parametrize(
    "directivity, exception_type",
    (
        ("omni", TypeError),
        (["omni"] * 3, TypeError),
        ([mic_dir0, "omni", mic_dir1] * 3, TypeError),
        ([mic_dir0, mic_dir1], ValueError),
    ),
)
def test_microphone_array_invalid_directivity(directivity, exception_type):

    locs = np.ones((3, 3))
    with pytest.raises(exception_type):
        pra.MicrophoneArray(locs, fs=_FS, directivity=directivity)


@pytest.mark.parametrize(
    "shape, with_dir, same_dir",
    (
        ((2, 1), False, False),
        ((2, 2), False, False),
        ((2, 3), False, False),
        ((3, 1), False, False),
        ((3, 3), False, False),
        ((3, 4), False, False),
        ((2, 3), True, False),
        ((3, 4), True, False),
        ((2, 3), True, True),
        ((3, 4), True, True),
    ),
)
def test_microphone_array_shape_correct(shape, with_dir, same_dir):

    locs = np.ones(shape)
    if with_dir:
        if same_dir:
            mdir = [mic_dir0] * shape[1]
        else:
            mdir = [mic_dir0, mic_dir1] + [None] * (shape[1] - 2)
    else:
        mdir = None
    mic_array = pra.MicrophoneArray(locs, fs=_FS, directivity=mdir)

    assert mic_array.dim == shape[0]
    assert mic_array.M == shape[1]
    assert mic_array.nmic == mic_array.M
    assert len(mic_array.directivity) == shape[1]


@pytest.mark.parametrize(
    "shape1, shape2, with_dir, from_raw_locs",
    (
        ((3, 2), (3, 2), False, False),
        ((3, 2), (3, 2), False, True),
        ((3, 2), (3, 2), False, False),
        ((3, 2), (3, 2), False, True),
        ((3, 2), (3, 2), True, False),
        ((3, 2), (3, 2), True, True),
        ((3, 2), (3, 1), False, False),
        ((3, 2), (3, 1), False, True),
    ),
)
def test_microphone_array_append(shape1, shape2, with_dir, from_raw_locs):
    if with_dir:
        mdir = [mic_dir0, mic_dir1] + [None] * (shape1[1] - 2)
    else:
        mdir = None

    mic_array = pra.MicrophoneArray(np.ones(shape1), fs=_FS, directivity=mdir)

    if from_raw_locs:
        mic_array.append(np.ones(shape2))

    else:
        mic_array.append(pra.MicrophoneArray(np.ones(shape2), fs=_FS))

    assert mic_array.nmic == shape1[1] + shape2[1]
    assert len(mic_array.directivity) == shape1[1] + shape2[1]
