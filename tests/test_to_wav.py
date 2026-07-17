"""
Tests for the float -> integer-PCM write path of
``MicrophoneArray.to_wav`` (and ``utilities.to_16b``).

``to_wav`` used to write ``np.array(signal, dtype=bitdepth)`` directly, which

* raised ``ValueError: Unsupported data type 'int8'`` for the documented
  ``bitdepth=np.int8`` (WAV 8-bit PCM is *unsigned*),
* produced an all-zero file for ``norm=False`` + any integer bitdepth (a float
  signal in [-1, 1] truncates to 0), and
* truncated toward zero instead of rounding, giving a sign-dependent ~1 LSB
  bias.

The oracle here is the read side (``to_float32``, tests in
``test_to_float32.py``): writing then reading back must recover the waveform.
"""

import os
import tempfile

import numpy as np
import pytest
from scipy.io import wavfile

from pyroomacoustics.beamforming import MicrophoneArray
from pyroomacoustics.utilities import to_16b, to_float32

FS = 16000


def _tone():
    t = np.arange(FS) / FS
    return (
        0.6 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 700 * t)
    ).astype(np.float64)


def _mono_array(signal):
    mics = MicrophoneArray(np.zeros((2, 1)), FS)
    mics.record(signal[np.newaxis, :], FS)
    return mics


_INT_BITDEPTHS = [np.int8, np.int16, np.int32, np.int64]


@pytest.mark.parametrize("bitdepth", [float, np.float32, np.float64] + _INT_BITDEPTHS)
@pytest.mark.parametrize("norm", [False, True])
def test_round_trip_write_read(bitdepth, norm):
    # Every (bitdepth, norm) combination must survive write -> read -> to_float32.
    # This covers the int8 crash and the norm=False all-zero regressions.
    sig = _tone()
    mics = _mono_array(sig)

    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "out.wav")
        mics.to_wav(fp, mono=True, norm=norm, bitdepth=bitdepth)
        _, back = wavfile.read(fp)

    backf = to_float32(back).astype(np.float64)
    assert not np.allclose(backf, 0.0), "silent (all-zero) output"

    # Compare shape: norm rescales, so allow an overall gain factor.
    scale = np.dot(sig, backf) / np.dot(backf, backf)
    resid = np.max(np.abs(sig - scale * backf)) / np.max(np.abs(sig))
    # int8 has only 256 levels; the others are effectively exact.
    tol = 1e-2 if bitdepth is np.int8 else 1e-3
    assert resid < tol, f"round-trip residual {resid} too large"


def test_int8_written_as_unsigned_offset_binary():
    # WAV has no signed 8-bit format; scipy returns 8-bit PCM as uint8 centered
    # at 128. Silence must map to the midpoint, not 0.
    sig = _tone()
    mics = _mono_array(sig)

    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "out.wav")
        mics.to_wav(fp, mono=True, norm=False, bitdepth=np.int8)
        _, back = wavfile.read(fp)

    assert back.dtype == np.uint8
    assert 0 <= back.min() and back.max() <= 255

    silence = np.zeros(16)
    mics_s = _mono_array(silence)
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "sil.wav")
        mics_s.to_wav(fp, mono=True, norm=False, bitdepth=np.int8)
        _, back = wavfile.read(fp)
    assert np.all(back == 128)


def test_full_scale_saturates_without_wraparound():
    # +1.0 / -1.0 must map to the rails, never wrap to the opposite sign. int64
    # is the sharp case: 2 ** 63 - 1 is not float64-representable, so a naive
    # cast of the rounded full-scale value wraps to INT64_MIN.
    sig = np.concatenate([np.ones(16), -np.ones(16)]).astype(np.float64)
    mics = _mono_array(sig)

    with tempfile.TemporaryDirectory() as d:
        for bitdepth in _INT_BITDEPTHS:
            fp = os.path.join(d, f"fs_{bitdepth.__name__}.wav")
            mics.to_wav(fp, mono=True, norm=False, bitdepth=bitdepth)
            _, back = wavfile.read(fp)
            top = to_float32(back[:16]).astype(np.float64)
            bottom = to_float32(back[16:]).astype(np.float64)
            assert np.all(top > 0.9), f"{bitdepth} positive rail wrapped: {back[0]}"
            assert np.all(
                bottom < -0.9
            ), f"{bitdepth} negative rail wrapped: {back[16]}"


def test_rounds_not_truncates():
    # A value between two codes must go to the nearest one, not toward zero.
    # 0.5 / 32768 is exactly halfway to code 1 at int16; truncation writes 0.
    sig = np.full(16, 0.7 / 32768.0, dtype=np.float64)
    mics = _mono_array(sig)
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "out.wav")
        mics.to_wav(fp, mono=True, norm=False, bitdepth=np.int16)
        _, back = wavfile.read(fp)
    assert np.all(back == 1), f"expected code 1 (rounded), got {back[0]}"


def test_multichannel_norm_false_int16():
    # Non-mono write of two channels must round-trip column-wise.
    ch0 = _tone()
    ch1 = 0.3 * np.sin(2 * np.pi * 440 * np.arange(FS) / FS)
    mics = MicrophoneArray(np.zeros((2, 2)), FS)
    mics.record(np.vstack([ch0, ch1]), FS)
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "stereo.wav")
        mics.to_wav(fp, mono=False, norm=False, bitdepth=np.int16)
        _, back = wavfile.read(fp)
    assert back.shape == (FS, 2)
    for orig, col in ((ch0, 0), (ch1, 1)):
        rec = to_float32(back[:, col]).astype(np.float64)
        np.testing.assert_allclose(rec, orig, atol=2e-4)


def test_integer_input_signal_passthrough():
    # An already-quantized integer signal must be written unchanged (the scaling
    # is only for float input).
    pcm = np.array([-32768, -100, 0, 100, 32767] * 4, dtype=np.int16)
    mics = MicrophoneArray(np.zeros((2, 1)), FS)  # M == 1
    mics.signals = pcm[np.newaxis, :]
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "int.wav")
        mics.to_wav(fp, mono=True, norm=False, bitdepth=np.int16)
        _, back = wavfile.read(fp)
    np.testing.assert_array_equal(back, pcm)


def test_to_16b_rounds_without_bias():
    # to_16b must round to nearest (max |error| <= 0.5 LSB); truncation gives a
    # sign-dependent bias of up to a full LSB.
    x = np.linspace(-1.0, 1.0, 100001)
    q = to_16b(x).astype(np.float64)
    err = q - x * (2**15 - 1)
    assert err.min() >= -0.5 - 1e-9
    assert err.max() <= 0.5 + 1e-9
    assert abs(err[x > 0].mean()) < 1e-2
    assert abs(err[x < 0].mean()) < 1e-2
    # Exact values.
    np.testing.assert_array_equal(
        to_16b(np.array([0.5, 0.25, -0.25])),
        np.round((2**15 - 1) * np.array([0.5, 0.25, -0.25])).astype(np.int16),
    )


if __name__ == "__main__":
    for norm in (False, True):
        for bd in [float, np.float32, np.float64] + _INT_BITDEPTHS:
            test_round_trip_write_read(bd, norm)
    test_int8_written_as_unsigned_offset_binary()
    test_full_scale_saturates_without_wraparound()
    test_rounds_not_truncates()
    test_multichannel_norm_false_int16()
    test_integer_input_signal_passthrough()
    test_to_16b_rounds_without_bias()
    print("ok")
