"""
This test verifies properties of octave band filters.
"""

import numpy as np
import pytest

import pyroomacoustics as pra

_FS = 16000
_NFFT = 4096
_BASE_FREQ = 125.0

octave_band_objects = [
    pra.OctaveBandsFactory(
        fs=_FS,
        n_fft=_NFFT,
        keep_dc=True,
        base_frequency=_BASE_FREQ,
    ),
    pra.AntoniOctaveFilterBank(
        fs=_FS,
        n_fft=pra.constants.get("octave_bands_n_fft"),
        base_frequency=_BASE_FREQ,
    ),
]


@pytest.mark.parametrize("octave_bands", octave_band_objects)
def test_octave_bands_perfect_reconstruction(octave_bands):
    fs = octave_bands.fs

    np.random.seed(0)
    x = np.random.randn(fs)

    x_bands = octave_bands.analysis(x)

    x_rec = x_bands.sum(axis=-1)

    error = abs(x - x_rec).max()
    print(error)
    assert np.allclose(x, x_rec)


@pytest.mark.parametrize("octave_bands", octave_band_objects)
def test_octave_bands_single_band(octave_bands):
    fs = octave_bands.fs

    np.random.seed(0)
    x = np.random.randn(fs)

    x_bands = octave_bands.analysis(x)

    x_bands_single = [
        octave_bands.analysis(x, band=b) for b in range(octave_bands.n_bands)
    ]
    x_bands_single = np.stack(x_bands_single, axis=-1)

    assert np.allclose(x_bands, x_bands_single)


@pytest.mark.parametrize("octave_bands", octave_band_objects)
def test_octave_bands_multi_band(octave_bands):
    fs = octave_bands.fs

    np.random.seed(0)
    x = np.random.randn(fs)

    x_bands = octave_bands.analysis(x)

    x_bands_multi = octave_bands.analysis(x, band=list(range(octave_bands.n_bands)))

    assert np.allclose(x_bands, x_bands_multi)


def test_octave_bands_interpolation():
    octave_bands = pra.AntoniOctaveFilterBank(fs=_FS, n_fft=256)
    coeffs = np.arange(octave_bands.n_bands)[::-1] + 1

    interpolation_filter = octave_bands.synthesis(
        coeffs, min_phase=False, filter_length=256
    )

    # Compare the energy of the original coefficients to those after filtering.
    expected_energy = np.sum(
        (octave_bands.get_bw() / octave_bands.fs * 2.0) * coeffs**2
    )
    filter_energy = np.sum(octave_bands.energy(interpolation_filter))

    assert abs(expected_energy - filter_energy) < 0.1


def test_energy_preserving_filtering():
    fs = 16000

    pra.constants.set("octave_bands_keep_dc", True)

    octave_bands = pra.AntoniOctaveFilterBank(
        fs=fs,
        n_fft=pra.constants.get("octave_bands_n_fft"),
        base_frequency=pra.constants.get("octave_bands_base_freq"),
    )

    # Reads the file containing the Eigenmike's directivity measurements
    eigenmike = pra.MeasuredDirectivityFile("EM32_Directivity", fs=fs)
    ir = eigenmike.impulse_responses[0, 0]

    ir_energy = np.square(ir).sum()
    ir_energy_bands = octave_bands.energy(ir)

    assert abs(ir_energy - ir_energy_bands.sum()) < 1e-3
