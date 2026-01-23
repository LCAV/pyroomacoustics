import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import curve_fit

import pyroomacoustics as pra

pra.constants.set("octave_bands_keep_dc", True)


@pytest.fixture
def samplerate():
    return 16_000


def get_scene_geometry():
    room_dim = np.array([3.0, 4.0, 5.0])
    source_loc = room_dim / 2.0 + np.array([0.001, -0.002, 0.003])
    co, az = np.pi / 2.5, 0.1 * np.pi
    unit_vec = np.array([np.cos(az) * np.sin(co), np.sin(az) * np.sin(co), np.cos(co)])
    mic_loc = source_loc + 1.5 * unit_vec
    return room_dim, source_loc, mic_loc


@pytest.fixture
def scene_geometry():
    return get_scene_geometry()


def get_wall_material(multiband=True):
    # Material with maximum energy absorption
    if multiband:
        full_eabs = {
            "description": "Multi-band fully absorbant",
            "coeffs": [0.11, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
            "center_freqs": 125 * 2.0 ** np.arange(7),
        }
        return pra.Material(energy_absorption=full_eabs)
    else:
        return pra.Material(energy_absorption=0.15)


@pytest.fixture(params=[(True,), (False,)], ids=["Multiband", "Single-band"])
def wall_material(request):
    return get_wall_material(multiband=request.param)


def energy_hist_2_rt60(hist, hist_bin_size, decay_db):
    # Integrate the energy from the tail.
    energy = np.cumsum(hist[::-1])[::-1]
    schroeder = 10.0 * np.log10(energy)
    # Remove the first 5 dB.
    schroeder += 5.0 - schroeder[0]

    t0 = np.where(schroeder < 0.0)[0][0]
    t1 = np.where(schroeder < -decay_db)[0][0]
    N = t1 - t0

    data = schroeder[t0:t1]
    data -= data[0]

    t = np.arange(N) * hist_bin_size
    X = np.column_stack((t, np.ones(N)))
    p, *_ = np.linalg.lstsq(X, data)

    rt60_1 = -60.0 / p[0]

    # 1. Define your model function
    def model(x, a, b):
        return b * np.exp(a * x)

    def jac(x, a, b):
        u = np.exp(a * x)
        return np.column_stack((x * b * u, u))

    # 2. Provide an initial guess [a, b, c]
    # This is crucial for convergence!
    initial_guess = [p[0] / (10.0 * np.log10(np.e)), 10.0 ** (p[1] / 10.0)]

    # 3. Perform the fit
    popt, pcov, info_dict, *_ = curve_fit(
        model, t, 10.0 ** (data / 10.0), p0=initial_guess, jac=jac, full_output=True
    )

    rt60_2 = -6.0 * np.log(10.0) / popt[0]

    return rt60_1, rt60_2


def simulate(room_dim, source_loc, mic_loc, material, use_rt, fs):
    hist_bin_size = 0.004

    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=material,
        max_order=-1 if use_rt else 100,
        ray_tracing=use_rt,
    )
    if use_rt:
        room.set_ray_tracing(
            n_rays=250_000, receiver_radius=0.5, hist_bin_size=hist_bin_size
        )

    room.add_source(source_loc, signal=np.zeros(10))
    room.add_microphone(mic_loc)

    room.simulate()
    rir = room.rir[0][0]

    bin_size = int(hist_bin_size * fs)
    n_bins = rir.shape[0] // bin_size
    n_max = n_bins * bin_size
    rir_hist = rir[:n_max]
    rir_hist = rir_hist.reshape((n_bins, bin_size))
    hist = np.sum(rir_hist**2, axis=1)

    bands_energy = np.mean(room.octave_bands.analysis(rir) ** 2, axis=0)
    bands_energy = bands_energy / bands_energy.sum()
    norm_bw = room.octave_bands.get_bw() / (room.fs / 2.0)

    rt60 = room.measure_rt60(decay_db=20.0)[0][0]
    N60 = int(rt60 * room.fs)

    rir_energy = np.sum(rir[:N60] ** 2)

    return room, rir[:N60], hist, bands_energy, norm_bw, rt60, rir_energy


def compute_tail_decay_curve(t0, rir, fs, hist_bin_size, n_max):
    rir = rir[t0:n_max]
    bin_size = int(hist_bin_size * fs)
    n_bins = rir.shape[0] // bin_size
    n_max = n_bins * bin_size
    rir_hist = rir[:n_max]
    rir_hist = rir_hist.reshape((n_bins, bin_size))
    return np.sum(rir_hist**2, axis=1)


def ism_to_rt_energy_ratio_db(t0, rir_rt, rir_ism, fs, hist_bin_size=0.004):
    n_max = min(rir_rt.shape[0], rir_ism.shape[0])
    hist_rt = compute_tail_decay_curve(t0, rir_rt, fs, hist_bin_size, n_max)
    hist_ism = compute_tail_decay_curve(t0, rir_ism, fs, hist_bin_size, n_max)
    return 10.0 * np.log10(np.mean(hist_ism**2) / np.mean((hist_rt - hist_ism) ** 2))


def test_energy_decay_ism_vs_rt(samplerate, scene_geometry, wall_material):
    np.random.seed(42)

    room_dim, source_loc, mic_loc = scene_geometry
    fs = samplerate

    room, rir_ism, *_ = simulate(
        room_dim, source_loc, mic_loc, wall_material, use_rt=False, fs=fs
    )
    room, rir_rt, *_ = simulate(
        room_dim, source_loc, mic_loc, wall_material, use_rt=True, fs=fs
    )

    c = pra.constants.get("c")
    t0 = int(
        fs * np.linalg.norm(source_loc - mic_loc) / c
        + pra.constants.get("frac_delay_length")
    )

    # Compare the energy decay only for the full band, because low
    # frequencies are not reliable with a shoebox room.
    irer_db = ism_to_rt_energy_ratio_db(t0, rir_ism, rir_rt, fs, hist_bin_size=0.04)
    assert irer_db >= 10.0, f"Failed IRER {irer_db:.2f} dB < 10.0"

    bws = room.octave_bands.get_bw()
    tols = [0.25, 0.1, 0.08]
    for b, bw in enumerate(np.sort(bws)):  # Loop through every band
        rir_band_ism = room.octave_bands.analysis(rir_ism, band=b)
        rir_band_rt = room.octave_bands.analysis(rir_rt, band=b)

        # Check the RT60 are within 15% of each other.
        rt60_band_ism = pra.experimental.measure_rt60(rir_band_ism, fs=fs)
        rt60_band_rt = pra.experimental.measure_rt60(rir_band_rt, fs=fs)
        rt60_delta = np.abs(rt60_band_ism - rt60_band_rt)
        rt60_rel_error = rt60_delta / rt60_band_ism
        tol = tols[min(b, 2)]
        assert rt60_rel_error < tol, f"Failed {rt60_rel_error=:.3} > {tol}"


if __name__ == "__main__":
    # Room dimensions
    fs = 16_000
    hist_bin_size = 0.004
    room_dim, source_loc, mic_loc = get_scene_geometry()
    material = get_wall_material(multiband=True)

    c = pra.constants.get("c")
    t0 = int(
        fs * np.linalg.norm(source_loc - mic_loc) / c
        + pra.constants.get("frac_delay_length")
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

    rirs = {}

    for label, use_rt in {"RT": True, "ISM": False}.items():

        room, rir, hist, bands_energy, norm_bw, rt60, rir_energy = simulate(
            room_dim, source_loc, mic_loc, material, use_rt=use_rt, fs=fs
        )
        rirs[label] = rir

        d = int(t0 / fs / room.rt_args["hist_bin_size"])
        d = 0
        et60_1, et60_2 = energy_hist_2_rt60(
            hist[d + 1 :], room.rt_args["hist_bin_size"], 20.0
        )

        print(f"{label}:")
        print(f"  T60={rt60:.3f}")
        print(f"  ET60 1={et60_1:.3f}")
        print(f"  ET60 2={et60_2:.3f}")
        for formula in ["sabine", "eyring"]:
            rt60_th = room.rt60_theory(formula)
            print(f"  T60={rt60_th:.3} ({formula})")

        print(f"  Energy(RIR)={rir_energy:.5f}")

        axes[0, 0].plot(
            np.arange(hist.shape[0]) * hist_bin_size, hist.T**0.5, label=label
        )

        axes[0, 1].plot(np.arange(rir.shape[0]) / room.fs, rir, label=label)

        axes[1, 0].magnitude_spectrum(rir, Fs=room.fs, scale="dB")

        axes[1, 1].plot(bands_energy, label=f"{label} measured")
        axes[1, 1].plot(norm_bw, label=f"{label} expected")

    irer_fb = ism_to_rt_energy_ratio_db(
        t0, rirs["ISM"], rirs["RT"], room.fs, hist_bin_size=0.04
    )
    print(f"Full band IRER: {irer_fb:.2f} dB")
    bws = room.octave_bands.get_bw()
    fig2, axes2 = plt.subplots(2, len(bws))
    for b, bw in enumerate(np.sort(bws)):  # Loop through every band
        # Compute the energy difference between the RIR at different bands.
        rir_band_ism = room.octave_bands.analysis(rirs["ISM"][t0:], band=b)
        rir_band_rt = room.octave_bands.analysis(rirs["RT"][t0:], band=b)
        n_max = min(rir_band_ism.shape[0], rir_band_rt.shape[0])

        rt60_band_ism = pra.experimental.measure_rt60(
            rir_band_ism, fs=fs, decay_db=60.0
        )
        rt60_band_rt = pra.experimental.measure_rt60(rir_band_rt, fs=fs, decay_db=60.0)
        rt60_delta = np.abs(rt60_band_ism - rt60_band_rt)
        rt60_rel_error = rt60_delta / rt60_band_ism
        print(
            f"Band {b}: RT60 error abs={rt60_delta:.3f} rel={rt60_rel_error:.3f}. ISM: {rt60_band_ism:.3f} RT: {rt60_band_rt:.3f}"
        )

        for lbl, ir in zip(["ism", "rt"], [rir_band_ism, rir_band_rt]):
            axes2[0, b].plot(
                np.arange(ir.shape[0]) / fs, ir, label=lbl if b == 0 else None
            )
            axes2[0, b].set_title(f"band {bw:.0f}")

            hist_band = compute_tail_decay_curve(t0, ir, fs, 0.04, n_max)
            axes2[1, b].plot(
                np.arange(hist_band.shape[0]) * 0.04,
                hist_band,
                label=lbl if b == 0 else None,
            )
            axes2[1, b].set_title(f"band energy {bw:.0f}")

        irer_db = ism_to_rt_energy_ratio_db(
            0, rir_band_ism, rir_band_rt, fs, hist_bin_size=0.04
        )
        print(f"Band {b}: IRER={irer_db:.2f} dB")

    fig2.legend()

    axes[0, 0].legend()
    axes[0, 1].set_title("RIR")
    axes[0, 0].set_title("Magnitude histogram")
    axes[1, 0].set_title("Magnitude spectrum of the RIR")
    axes[1, 1].set_title("Band energies")
    axes[1, 1].legend()

    plt.show()
