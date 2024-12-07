import numpy as np
import pytest

import pyroomacoustics as pra

sig = np.arange(10)
room_size = [10, 9, 8]
source_loc0 = [1.5, 1.7, 2.1]
source_loc1 = [3.5, 7.7, 2.1]
mic0 = [7, 8, 3.9]
mic1 = [7.87, 3.6, 6.1]
mic_dir0 = pra.FigureEight(
    orientation=pra.DirectionVector(azimuth=90, colatitude=15, degrees=True)
)
mic_dir1 = pra.FigureEight(
    orientation=pra.DirectionVector(azimuth=180, colatitude=15, degrees=True)
)
src_dir0 = pra.FigureEight(
    orientation=pra.DirectionVector(azimuth=270, colatitude=15, degrees=True)
)
src_dir1 = pra.FigureEight(
    orientation=pra.DirectionVector(azimuth=0, colatitude=15, degrees=True)
)


@pytest.mark.parametrize("with_dir", ((True,), (False,)))
def test_add_source_mic(with_dir):
    room = pra.ShoeBox(room_size)

    if with_dir:
        sdir0 = src_dir0
        sdir1 = src_dir1
        mdir0 = mic_dir0
        mdir1 = mic_dir1
    else:
        sdir0 = sdir1 = None
        mdir0 = mdir1 = None

    room = (
        pra.ShoeBox(room_size)
        .add_source(source_loc0, directivity=sdir0)
        .add_microphone(mic0, directivity=mdir0)
    )

    assert len(room.sources) == 1
    assert np.allclose(room.sources[0].position, source_loc0)
    assert len(room.mic_array) == 1
    assert room.mic_array.R.shape == (3, 1)
    assert np.allclose(room.mic_array.R[:, 0], mic0)
    # Test directivities.
    assert room.sources[0].directivity is sdir0
    assert all(d is md for d, md in zip(room.mic_array.directivity, [mdir0]))

    room.add_microphone(mic1, directivity=mdir1).add_source(
        source_loc1, directivity=sdir1
    )

    assert len(room.sources) == 2
    assert np.allclose(room.sources[1].position, source_loc1)
    assert len(room.mic_array) == 2
    assert np.allclose(room.mic_array.R[:, 0], mic0)
    assert np.allclose(room.mic_array.R[:, 1], mic1)
    assert room.mic_array.R.shape == (3, 2)
    # Test directivities.
    assert room.sources[0].directivity is sdir0
    assert room.sources[1].directivity is sdir1
    assert all(d is md for d, md in zip(room.mic_array.directivity, [mdir0, mdir1]))


@pytest.mark.parametrize("with_dir", ((True,), (False,)))
def test_add_source_mic_obj(with_dir):
    room = pra.ShoeBox(room_size)

    if with_dir:
        sdir0 = src_dir0
        sdir1 = src_dir1
        mdir0 = mic_dir0
        mdir1 = mic_dir1
    else:
        sdir0 = sdir1 = None
        mdir0 = mdir1 = None

    source0 = pra.SoundSource(source_loc0, signal=sig, directivity=sdir0)
    source1 = pra.SoundSource(source_loc1, signal=sig, directivity=sdir1)

    mic_array0 = pra.MicrophoneArray(np.c_[mic0], fs=room.fs, directivity=mdir0)
    mic_array1 = pra.MicrophoneArray(np.c_[mic1], fs=room.fs, directivity=mdir1)

    room.add(source0).add(mic_array0)

    assert len(room.sources) == 1
    assert np.allclose(room.sources[0].position, source_loc0)
    assert len(room.mic_array) == 1
    assert room.mic_array.R.shape == (3, 1)
    assert np.allclose(room.mic_array.R[:, 0], mic0)
    # Test directivities.
    assert room.sources[0].directivity is sdir0
    assert all(d is md for d, md in zip(room.mic_array.directivity, [mdir0]))

    room.add(mic_array1).add(source1)

    assert len(room.sources) == 2
    assert np.allclose(room.sources[1].position, source_loc1)
    assert len(room.mic_array) == 2
    assert np.allclose(room.mic_array.R[:, 0], mic0)
    assert np.allclose(room.mic_array.R[:, 1], mic1)
    assert room.mic_array.R.shape == (3, 2)
    # Test directivities.
    assert room.sources[0].directivity is sdir0
    assert room.sources[1].directivity is sdir1
    assert all(d is md for d, md in zip(room.mic_array.directivity, [mdir0, mdir1]))


@pytest.mark.parametrize("with_dir", ((True,), (False,)))
def test_add_source_mic_obj_2(with_dir):
    room = pra.ShoeBox(room_size)

    if with_dir:
        sdir0 = src_dir0
        sdir1 = src_dir1
        mdir = [mic_dir0, mic_dir1]
    else:
        sdir0 = sdir1 = None
        mdir = [None, None]

    source0 = pra.SoundSource(source_loc0, signal=sig, directivity=sdir0)
    source1 = pra.SoundSource(source_loc1, signal=sig, directivity=sdir1)
    mic_array = pra.MicrophoneArray(np.c_[mic0, mic1], fs=room.fs, directivity=mdir)

    room.add(source0).add(source1).add(mic_array)

    assert len(room.sources) == 2
    assert np.allclose(room.sources[0].position, source_loc0)
    assert np.allclose(room.sources[1].position, source_loc1)
    assert len(room.mic_array) == 2
    assert np.allclose(room.mic_array.R[:, 0], mic0)
    assert np.allclose(room.mic_array.R[:, 1], mic1)
    assert room.mic_array.R.shape == (3, 2)
    # Test directivities.
    assert room.sources[0].directivity is sdir0
    assert room.sources[1].directivity is sdir1
    assert all(d is md for d, md in zip(room.mic_array.directivity, mdir))


def test_add_source_mic_obj_with_dir_error():
    room = pra.ShoeBox(room_size)

    mic_array = pra.MicrophoneArray(np.c_[mic0, mic1], fs=room.fs)

    with pytest.raises(ValueError):
        room.add_microphone_array(mic_array, directivity=[mic_dir0, mic_dir1])


@pytest.mark.parametrize("with_dir", ((True,), (False,)))
def test_add_source_mic_ndarray(with_dir):
    if with_dir:
        sdir0 = src_dir0
        sdir1 = src_dir1
        mdir = [mic_dir0, mic_dir1]
    else:
        sdir0 = sdir1 = None
        mdir = [None, None]

    source0 = pra.SoundSource(source_loc0, signal=sig, directivity=sdir0)
    source1 = pra.SoundSource(source_loc1, signal=sig, directivity=sdir1)
    mic_array = np.c_[mic0, mic1]

    room = (
        pra.ShoeBox(room_size)
        .add(source0)
        .add(source1)
        .add_microphone_array(mic_array, directivity=mdir)
    )

    assert len(room.sources) == 2
    assert np.allclose(room.sources[0].position, source_loc0)
    assert np.allclose(room.sources[1].position, source_loc1)
    assert len(room.mic_array) == 2
    assert np.allclose(room.mic_array.R[:, 0], mic0)
    assert np.allclose(room.mic_array.R[:, 1], mic1)
    assert room.mic_array.R.shape == (3, 2)
    # Test directivities.
    assert room.sources[0].directivity is sdir0
    assert room.sources[1].directivity is sdir1
    assert all(d is md for d, md in zip(room.mic_array.directivity, mdir))


@pytest.mark.parametrize("with_dir", ((True,), (False,)))
def test_add_source_mic_ndarray_2(with_dir):
    if with_dir:
        sdir0 = src_dir0
        sdir1 = src_dir1
        mdir = [mic_dir0, mic_dir1]
    else:
        sdir0 = sdir1 = None
        mdir = [None, None]

    source0 = pra.SoundSource(source_loc0, signal=sig, directivity=sdir0)
    source1 = pra.SoundSource(source_loc1, signal=sig, directivity=sdir1)
    mic_array = np.c_[mic0, mic1]

    room = (
        pra.ShoeBox(room_size)
        .add(source0)
        .add(source1)
        .add_microphone(mic_array, directivity=mdir)
    )

    assert len(room.sources) == 2
    assert np.allclose(room.sources[0].position, source_loc0)
    assert np.allclose(room.sources[1].position, source_loc1)
    assert len(room.mic_array) == 2
    assert np.allclose(room.mic_array.R[:, 0], mic0)
    assert np.allclose(room.mic_array.R[:, 1], mic1)
    assert room.mic_array.R.shape == (3, 2)
    # Test directivities.
    assert room.sources[0].directivity is sdir0
    assert room.sources[1].directivity is sdir1
    assert all(d is md for d, md in zip(room.mic_array.directivity, mdir))


if __name__ == "__main__":
    test_add_source_mic()
    test_add_source_mic_obj()
    test_add_source_mic_obj_2()
    test_add_source_mic_ndarray()
    test_add_source_mic_ndarray_2()
