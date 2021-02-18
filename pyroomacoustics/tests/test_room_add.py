import numpy as np
import pyroomacoustics as pra

sig = np.arange(10)
room_size = [10, 9, 8]
source_loc0 = [1.5, 1.7, 2.1]
source_loc1 = [3.5, 7.7, 2.1]
mic0 = [7, 8, 3.9]
mic1 = [7.87, 3.6, 6.1]


def test_add_source_mic():

    room = pra.ShoeBox(room_size).add_source(source_loc0).add_microphone(mic0)

    assert len(room.sources) == 1
    assert np.allclose(room.sources[0].position, source_loc0)
    assert len(room.mic_array) == 1
    assert room.mic_array.R.shape == (3, 1)
    assert np.allclose(room.mic_array.R[:, 0], mic0)

    room.add_microphone(mic1).add_source(source_loc1)

    assert len(room.sources) == 2
    assert np.allclose(room.sources[1].position, source_loc1)
    assert len(room.mic_array) == 2
    assert np.allclose(room.mic_array.R[:, 0], mic0)
    assert np.allclose(room.mic_array.R[:, 1], mic1)
    assert room.mic_array.R.shape == (3, 2)


def test_add_source_mic_obj():

    room = pra.ShoeBox(room_size)

    source0 = pra.SoundSource(source_loc0, signal=sig)
    source1 = pra.SoundSource(source_loc1, signal=sig)

    mic_array0 = pra.MicrophoneArray(np.c_[mic0], fs=room.fs)
    mic_array1 = pra.MicrophoneArray(np.c_[mic1], fs=room.fs)

    room.add(source0).add(mic_array0)

    assert len(room.sources) == 1
    assert np.allclose(room.sources[0].position, source_loc0)
    assert len(room.mic_array) == 1
    assert room.mic_array.R.shape == (3, 1)
    assert np.allclose(room.mic_array.R[:, 0], mic0)

    room.add(mic_array1).add(source1)

    assert len(room.sources) == 2
    assert np.allclose(room.sources[1].position, source_loc1)
    assert len(room.mic_array) == 2
    assert np.allclose(room.mic_array.R[:, 0], mic0)
    assert np.allclose(room.mic_array.R[:, 1], mic1)
    assert room.mic_array.R.shape == (3, 2)


def test_add_source_mic_obj_2():

    room = pra.ShoeBox(room_size)

    source0 = pra.SoundSource(source_loc0, signal=sig)
    source1 = pra.SoundSource(source_loc1, signal=sig)
    mic_array = pra.MicrophoneArray(np.c_[mic0, mic1], fs=room.fs)

    room.add(source0).add(source1).add(mic_array)

    assert len(room.sources) == 2
    assert np.allclose(room.sources[0].position, source_loc0)
    assert np.allclose(room.sources[1].position, source_loc1)
    assert len(room.mic_array) == 2
    assert np.allclose(room.mic_array.R[:, 0], mic0)
    assert np.allclose(room.mic_array.R[:, 1], mic1)
    assert room.mic_array.R.shape == (3, 2)


def test_add_source_mic_ndarray():

    source0 = pra.SoundSource(source_loc0, signal=sig)
    source1 = pra.SoundSource(source_loc1, signal=sig)
    mic_array = np.c_[mic0, mic1]

    room = (
        pra.ShoeBox(room_size).add(source0).add(source1).add_microphone_array(mic_array)
    )

    assert len(room.sources) == 2
    assert np.allclose(room.sources[0].position, source_loc0)
    assert np.allclose(room.sources[1].position, source_loc1)
    assert len(room.mic_array) == 2
    assert np.allclose(room.mic_array.R[:, 0], mic0)
    assert np.allclose(room.mic_array.R[:, 1], mic1)
    assert room.mic_array.R.shape == (3, 2)


def test_add_source_mic_ndarray_2():

    source0 = pra.SoundSource(source_loc0, signal=sig)
    source1 = pra.SoundSource(source_loc1, signal=sig)
    mic_array = np.c_[mic0, mic1]

    room = pra.ShoeBox(room_size).add(source0).add(source1).add_microphone(mic_array)

    assert len(room.sources) == 2
    assert np.allclose(room.sources[0].position, source_loc0)
    assert np.allclose(room.sources[1].position, source_loc1)
    assert len(room.mic_array) == 2
    assert np.allclose(room.mic_array.R[:, 0], mic0)
    assert np.allclose(room.mic_array.R[:, 1], mic1)
    assert room.mic_array.R.shape == (3, 2)


if __name__ == "__main__":
    test_add_source_mic()
    test_add_source_mic_obj()
    test_add_source_mic_obj_2()
    test_add_source_mic_ndarray()
    test_add_source_mic_ndarray_2()
