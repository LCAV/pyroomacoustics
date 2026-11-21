import sys

import numpy as np
import pytest
from scipy.signal import resample_poly

import pyroomacoustics as pra


@pytest.fixture
def resample_backend():
    previous = pra.constants.get("resample_backend")
    yield
    pra.constants.set("resample_backend", previous)


@pytest.mark.parametrize(
    "fs_in, fs_out", [(16000, 8000), (8000, 16000), (16000, 22050)]
)
@pytest.mark.parametrize("channels", [1, 3])
def test_record_resamples_without_samplerate(
    fs_in, fs_out, channels, monkeypatch, resample_backend
):
    # The scipy backend must also work when optional samplerate is absent.
    monkeypatch.setitem(sys.modules, "samplerate", None)
    pra.constants.set("resample_backend", "scipy")
    signal = np.random.default_rng(0).normal(size=(channels, 1600))
    original = signal.copy()
    array = pra.MicrophoneArray(np.ones((2, channels)), fs=fs_out)
    array.record(signal, fs_in)
    expected = resample_poly(signal, fs_out, fs_in, axis=-1)
    np.testing.assert_array_equal(array.signals, expected)
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize("backend", ["soxr", "samplerate"])
def test_record_preserves_selected_backend(backend, resample_backend):
    module = pytest.importorskip(backend)
    pra.constants.set("resample_backend", backend)
    signal = np.random.default_rng(0).normal(size=(2, 1600))
    array = pra.MicrophoneArray(np.ones((2, 2)), fs=8000)
    if backend == "soxr":
        expected = module.resample(signal.T, 16000, 8000).T
    else:
        expected = module.resample(signal.T, 0.5, "sinc_best").T
    array.record(signal, 16000)
    np.testing.assert_array_equal(array.signals, expected)


def test_record_at_same_rate():
    signal = np.random.default_rng(0).normal(size=(2, 1600))
    array = pra.MicrophoneArray(np.ones((2, 2)), fs=16000)
    array.record(signal, 16000)
    np.testing.assert_array_equal(array.signals, signal)


def test_room_simulation_with_different_microphone_rate(resample_backend):
    pra.constants.set("resample_backend", "scipy")
    room = pra.ShoeBox([4, 5], fs=16000, max_order=1)
    room.add_source([1, 1], signal=np.random.default_rng(0).normal(size=1600))
    room.add_microphone_array(
        pra.MicrophoneArray(np.array([[2.0, 3.0], [2.0, 2.0]]), fs=8000)
    )
    premix = room.simulate(return_premix=True)
    expected = resample_poly(premix.sum(axis=0), 1, 2, axis=-1)
    np.testing.assert_array_equal(room.mic_array.signals, expected)
