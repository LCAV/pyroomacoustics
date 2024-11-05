import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import pytest


@pytest.mark.parametrize(
    "fs_in, fs_out, backend",
    [
        (240, 160, "soxr"),
        (240, 160, "samplerate"),
        (240, 160, "scipy"),
    ],
)
def test_downsample(fs_in, fs_out, backend):
    """Idea use a sine above Nyquist of fs_out. It should disappear."""
    assert fs_in > fs_out
    f_sine = fs_out / 2.0 + (fs_in - fs_out) / 2.0 * 0.95
    time = np.arange(fs_in * 10) / fs_in
    signal_in = np.sin(2.0 * np.pi * time * f_sine)
    signal_out = pra.resample(signal_in, fs_in, fs_out, backend=backend)

    assert abs(signal_out).mean() < 1e-3


if __name__ == "__main__":

    # Reads the file containing the Eigenmike's directivity measurements
    eigenmike = pra.MeasuredDirectivityFile("EM32_Directivity")

    fs_tgt = 16000
    fs_file = eigenmike.fs

    print(f"{fs_file=} {fs_tgt=}")

    rir_original = eigenmike.impulse_responses[10, 20]
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
