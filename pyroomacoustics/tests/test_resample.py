"""
Very basic tests to verify that all resampling backends
can be called and are doing their job.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pyroomacoustics as pra


@pytest.mark.parametrize(
    "fs_in, fs_out, backend",
    [
        (240, 160, None),
        (240, 160, "soxr"),
        (240, 160, "samplerate"),
        (240, 160, "scipy"),
    ],
)
def test_downsample(fs_in, fs_out, backend):
    """Idea use a sine above Nyquist of fs_out. It should disappear."""
    assert fs_in > fs_out
    f_sine = fs_out / 2.0 + (fs_in - fs_out) / 2.0 * 0.75
    time = np.arange(fs_in * 10) / fs_in
    signal_in = np.sin(2.0 * np.pi * time * f_sine)
    signal_in = signal_in * np.hanning(signal_in.shape[0])
    signal_out = pra.resample(signal_in, fs_in, fs_out, backend=backend)

    assert abs(signal_out).max() < 1e-3


@pytest.mark.parametrize(
    "fs_in, fs_out, backend",
    [
        (160, 240, None),
        (160, 240, "soxr"),
        (160, 240, "samplerate"),
        (160, 240, "scipy"),
    ],
)
def test_upsample(fs_in, fs_out, backend):
    """Idea use a sine above Nyquist of fs_out. It should disappear."""
    assert fs_in < fs_out

    # make a random signal
    signal_in = np.random.randn(10 * fs_in)
    signal_in = signal_in * np.hanning(signal_in.shape[0])

    signal_out = pra.resample(signal_in, fs_in, fs_out, backend=backend)

    # the test relies on upper frequency being empty
    f_cut = fs_in / 2.0 + (fs_out - fs_in) / 2.0 * 0.75
    signal_out_filt = pra.highpass(signal_out, fs_out, fc=f_cut)

    assert abs(signal_out_filt).max() < 1e-3


if __name__ == "__main__":

    test_cases = []

    # Test 1 is the eigenmike impulse response
    # Reads the file containing the Eigenmike's directivity measurements
    eigenmike = pra.MeasuredDirectivityFile("EM32_Directivity")
    fs_tgt = 16000
    fs_file = eigenmike.fs
    test_cases.append((fs_file, fs_tgt, eigenmike.impulse_responses[0, 0]))

    # Test 2 is a sine
    fs_in = 240
    fs_out = 160
    f_sine = fs_out / 2.0 + (fs_in - fs_out) / 2.0 * 0.95
    time = np.arange(fs_in * 10) / fs_in
    signal_in = np.sin(2.0 * np.pi * time * f_sine)
    signal_in = signal_in * np.hanning(signal_in.shape[0])
    test_cases.append((fs_in, fs_out, signal_in))

    # Test 3 is some random noise
    np.random.seed(0)
    fs_in = 160
    fs_out = 240
    signal_in = np.random.randn(fs_in * 10)
    test_cases.append((fs_in, fs_out, signal_in))

    for fs_in, fs_out, rir_original in test_cases:
        time_file = np.arange(rir_original.shape[0]) / fs_file

        rirs = {}
        for backend in ["soxr", "samplerate", "scipy"]:
            rirs[backend] = pra.resample(rir_original, fs_file, fs_tgt, backend=backend)

        fig, ax = plt.subplots(1, 1)
        ax.plot(time_file, rir_original, label="Original")
        for idx, (backend, rir) in enumerate(rirs.items()):
            time_rir = np.arange(rir.shape[0]) / fs_tgt
            ax.plot(time_rir, rir, label=backend, linewidth=(3 - idx))
        ax.legend()
        plt.show()
