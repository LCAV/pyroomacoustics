"""
# Octave Band Filter Banks

Multi-frequency simulation relies on octave-band filterbanks
to run the simulation in different perceptually relevant frequency bands
before merging the results into a single room impulse response.

This scripts demonstrate some of the octave band filters available
in pyroomacoustics.

Two ocatave band filter banks are implemented in pyroomacoustics.

## Cosine Filterbank

This filterbank uses a number of overlapping cosine filters to
cover the octaves.
It guarantees perfect reconstruction, but does not conserve the energy
in the bands.

## Antoni's Orthogonal-like Fractional Octave Bands

This class implements a type of fractional octave filter bank with
both perfect reconstruction and energy conservation.

J. Antoni, Orthogonal-like fractional-octave-band filters, J. Acoust. Soc.
Am., 127, 2, February 2010
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp

import pyroomacoustics as pra

if __name__ == "__main__":

    # Test unit energy.
    fs = 16000
    n_fft = 2**10  # 1024
    base_freq = 125.0  # Hertz, default frequency in pyroomacoustics.

    # The cosine filter bank
    octave_bands = pra.OctaveBandsFactory(
        fs=fs,
        n_fft=n_fft,
        keep_dc=True,
        base_frequency=base_freq,
    )

    # The orthogonal-like filterbankd with perfect reconstruction and energy
    # conservation.
    # The `band_overlap_ratio` and `slope` parameters control the transition
    # between adjacent bands.
    antoni_octave_bands = pra.AntoniOctaveFilterBank(
        fs=fs,
        base_frequency=base_freq,
        band_overlap_ratio=0.5,
        slope=0,
        n_fft=n_fft,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    filters_orth = np.zeros(n_fft)
    filters_orth[n_fft // 2] = 1.0
    filters_orth = antoni_octave_bands.analysis(filters_orth)

    for i, (lbl, ob) in enumerate(
        {"Cosine": octave_bands.filters, "Antoni": filters_orth}.items()
    ):
        filters_spectrum = np.fft.rfft(ob, axis=0)
        n_freq = filters_spectrum.shape[0]
        freq = np.arange(n_freq) / n_freq * (fs / 2)
        time = np.arange(ob.shape[0]) / fs

        sum_reconstruction = np.sum(abs(filters_spectrum), axis=1)

        for b in range(octave_bands.n_bands):
            c = antoni_octave_bands.centers[b]
            line_label = (
                (f"{c if c < 1000 else c / 1000:.0f}{'k' if c >= 1000 else ''}Hz")
                if lbl == "Antoni"
                else None
            )
            axes[0, i].plot(freq, abs(filters_spectrum[:, b]) ** 2, label=line_label)
            axes[1, i].plot(time, ob[:, b])
        axes[0, i].plot(
            freq, sum_reconstruction, label="sum magnitude" if lbl == "Antoni" else None
        )
        axes[0, i].set_title(f"{lbl} - energy response")
        axes[1, i].set_title(f"{lbl} - impulse response")
        axes[1, i].set_xlim(0.025, 0.04)

    fig.legend(loc="lower right")
    fig.tight_layout()

    # Test octave bands interpolation
    coeffs = np.arange(antoni_octave_bands.n_bands)[::-1] + 1

    mat_interp = {
        "Cosine": octave_bands.synthesis(coeffs, min_phase=True),
        "Cosine, min. phase": octave_bands.synthesis(coeffs, min_phase=False),
        "Antoni": antoni_octave_bands.synthesis(coeffs, min_phase=False),
        "Antoni, min. phase": antoni_octave_bands.synthesis(coeffs, min_phase=True),
    }

    # Compare the energy of the original coefficients to those after filtering.
    energy = (antoni_octave_bands.get_bw() / octave_bands.fs * 2.0) * coeffs**2
    bar_labels = [
        f"{c if c < 1000 else c / 1000:.0f}{'k' if c > 1000 else ''}"
        for c in antoni_octave_bands.centers
    ] + ["Total"]
    bar_energy_original = energy.tolist() + [energy.sum()]

    bar_width = 0.9 / (len(mat_interp) + 2)

    def bar_x(idx):
        bar_space = 0.1 / (len(mat_interp) + 2)
        x = np.arange(len(bar_energy_original))
        return bar_width / 2.0 + idx * (bar_width + bar_space) + x

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].set_title("Impulse response")
    axes[1].set_title("Magnitude response")
    axes[2].set_title("Per-band energy")
    for idx, (lbl, values) in enumerate(mat_interp.items()):
        axes[0].plot(np.arange(values.shape[-1]) / fs, values, label=lbl)
        axes[0].set_xlabel("Time (s)")

        H = abs(np.fft.rfft(values))
        axes[1].plot(np.arange(H.shape[-1]) * fs / values.shape[-1], H, label=lbl)
        axes[1].set_xlabel("Frequency (Hz)")

        energy_bands = antoni_octave_bands.energy(values).tolist()
        energy_bands += [sum(energy_bands)]
        axes[2].bar(bar_x(idx), energy_bands, label=lbl, width=bar_width)

    axes[2].bar(
        bar_x(len(mat_interp)), bar_energy_original, label="True", width=bar_width
    )
    axes[2].set_xticks(np.arange(len(bar_energy_original)) + 0.5, bar_labels)
    axes[2].set_xlabel("Band centers (Hz)")
    axes[2].legend()
    fig.tight_layout()

    # Test reconstruction
    time = np.arange(fs * 5) / fs
    f0 = 1.0
    x = np.zeros_like(time)
    x[fs : 3 * fs] = np.sin(2 * np.pi * f0 * time[fs : 3 * fs])
    x = chirp(time, 100, time[-1], 7500, method="linear")

    x_bands = octave_bands.analysis(x)
    x_rec = x_bands.sum(axis=-1)

    band_energy = antoni_octave_bands.energy(x, oversampling=4)
    print(
        f"Energy of input signal: {np.square(x).sum()}, "
        f"sum of band energies: {band_energy.sum()}"
    )
    print(f"Reconstruction error: {abs(x - x_rec).max()}")

    low = None
    high = None

    low = 0 if low is None else low
    high = x.shape[-1] if high is None else high

    num_plots = x_bands.shape[-1] + 1
    freq = np.arange(x_bands.shape[-2] // 2 + 1) / x_bands.shape[-2] * fs
    time = np.arange(x.shape[-1]) / fs
    fig, axes = plt.subplots(num_plots, 2, figsize=(10, 6))
    axes[0, 0].plot(time[low:high], x_rec[low:high], label="reconstructed")
    axes[0, 0].plot(time[low:high], x[low:high], label="original")
    axes[0, 0].plot(time[low:high], x[low:high] - x_rec[low:high], label="error")
    L = axes[0, 1].magnitude_spectrum(x, label="reconstructed", Fs=fs)
    L = axes[0, 1].magnitude_spectrum(x_rec, label="reconstructed", Fs=fs)
    axes[0, 0].legend(fontsize="xx-small")

    ylim_time = x.min(), x.max()
    ylim = -0.01 * max(L[0]), max(L[0])
    axes[0, 0].set_title("Filtered signal")
    axes[0, 1].set_title("Magnitude response")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")
    axes[0, 1].set_ylim(ylim)
    axes[0, 1].set_xticks([])
    axes[0, 0].set_ylim(ylim_time)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_ylabel("Sum")
    for b in range(1, num_plots):
        axes[b, 0].plot(time[low:high], x_bands[low:high, b - 1])
        axes[b, 0].set_ylim(ylim_time)
        axes[b, 1].magnitude_spectrum(x_bands[low:high, b - 1], Fs=fs)
        axes[b, 1].set_ylim(ylim)
        axes[b, 1].set_xlabel("")
        axes[b, 1].set_ylabel("")
        if b < num_plots - 1:
            axes[b, 0].set_xticks([])
            axes[b, 1].set_xticks([])
        else:
            axes[b, 0].set_xlabel("Time (s)")
            ticks = np.arange(0, fs / 2 + 1, 1000)
            ticklabels = [
                f"{c if c < 1000 else c / 1000:.0f}{'k' if c > 1000 else ''} Hz"
                for c in ticks
            ]
            axes[b, 1].set_xticks(ticks, ticklabels)
        if b > 0:
            c = antoni_octave_bands.centers[b - 1]
            axes[b, 0].set_ylabel(
                f"{c if c < 1000 else c / 1000:.0f}{'k' if c > 1000 else ''} Hz"
            )
            axes[b, 0].set_xlabel("Frequency (Hz)")
    fig.tight_layout()

    plt.show()
