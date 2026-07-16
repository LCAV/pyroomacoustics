"""Tests for :func:`pyroomacoustics.utilities.normalize`."""

import numpy as np
from scipy.io import wavfile

import pyroomacoustics as pra
from pyroomacoustics.utilities import normalize


def test_integer_pcm_is_normalized_as_floating_point():
    signal = np.array([-16384, 0, 8192], dtype=np.int16)

    normalized = normalize(signal)

    assert np.issubdtype(normalized.dtype, np.floating)
    np.testing.assert_array_equal(normalized, [-1.0, 0.0, 0.5])


def test_silent_signal_remains_finite_and_silent():
    signal = np.zeros(16, dtype=np.float32)

    normalized = normalize(signal)

    assert normalized.dtype == signal.dtype
    np.testing.assert_array_equal(normalized, signal)
    assert np.all(np.isfinite(normalized))


def test_integer_dtype_limits_do_not_overflow_before_normalization():
    info = np.iinfo(np.int64)
    signal = np.array([info.min, 0, info.max], dtype=np.int64)

    normalized = normalize(signal)

    np.testing.assert_allclose(normalized, [-1.0, 0.0, 1.0])


def test_float_precision_and_bit_scaling_are_preserved():
    signal = np.array([-0.5, 0.0, 0.25], dtype=np.float32)

    normalized = normalize(signal, bits=16)

    assert normalized.dtype == signal.dtype
    np.testing.assert_array_equal(normalized, [-32767.0, 0.0, 16383.5])


def test_silent_microphone_recording_writes_finite_float_wav(tmp_path):
    mic_array = pra.MicrophoneArray(np.zeros((2, 1)), fs=16000)
    mic_array.signals = np.zeros((1, 32), dtype=np.float32)
    wav_path = tmp_path / "silence.wav"

    mic_array.to_wav(wav_path, norm=True, bitdepth=np.float32)
    _, recording = wavfile.read(wav_path)

    assert recording.dtype == np.float32
    np.testing.assert_array_equal(recording, np.zeros(32, dtype=np.float32))
    assert np.all(np.isfinite(recording))
