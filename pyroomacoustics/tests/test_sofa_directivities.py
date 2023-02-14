"""
This is a regression test for the SOFA source and receiver measured directivity patterns

The tests compare the output of the simulation with some pre-generated samples.

To generate the samples run this file: `python ./test_sofa_directivities.py`
"""
import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pyroomacoustics as pra
from pyroomacoustics.datasets.sofa import (
    DEFAULT_SOFA_PATH,
    download_sofa_files,
    get_sofa_db_info,
)
from pyroomacoustics.directivities import (
    CardioidFamily,
    DirectionVector,
    DirectivityPattern,
    DIRPATRir,
)
from pyroomacoustics.doa import GridSphere
from pyroomacoustics.open_sofa_interpolate import (
    calculation_pinv_voronoi_cells,
    calculation_pinv_voronoi_cells_general,
)

sofa_info = get_sofa_db_info()
supported_sofa = [name for name, info in sofa_info.items() if info["supported"] == True]

TEST_DATA = Path(__file__).parent / "data"

# tolerances for the regression tests
atol = 5e-5
rtol = 1e-8

room_dim = [6, 6, 2.4]

all_materials = {
    "east": pra.Material(
        energy_absorption={
            "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.54,
    ),
    "west": pra.Material(
        energy_absorption={
            "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.54,
    ),
    "north": pra.Material(
        energy_absorption={
            "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.54,
    ),
    "south": pra.Material(
        energy_absorption={
            "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.54,
    ),
    "ceiling": pra.Material(
        energy_absorption={
            "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.54,
    ),
    "floor": pra.Material(
        energy_absorption={
            "coeffs": [0.11, 0.14, 0.37, 0.43, 0.27, 0.25],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.54,
    ),
}


def test_dirpat_download():
    files = download_sofa_files(verbose=True)
    for file in files:
        assert file.exists()


SOFA_ONE_SIDE_PARAMETERS = [
    ("AKG_c480", "AKG_c480_c414_CUBE.sofa", False, False),
    ("AKG_c414K", "AKG_c480_c414_CUBE.sofa", False, False),
    ("AKG_c414N", "AKG_c480_c414_CUBE.sofa", False, False),
    ("AKG_c414S", "AKG_c480_c414_CUBE.sofa", False, False),
    ("AKG_c414A", "AKG_c480_c414_CUBE.sofa", False, False),
    ("EM_32_0", "EM32_Directivity.sofa", False, False),
    ("EM_32_31", "EM32_Directivity.sofa", False, False),
    ("Genelec_8020", "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa", False, False),
    (
        "Vibrolux_2x10inch",
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
        False,
        False,
    ),
]


@pytest.mark.parametrize(
    "pattern_id,sofa_file_name,save_flag,plot_flag", SOFA_ONE_SIDE_PARAMETERS
)
def test_sofa_one_side(pattern_id, sofa_file_name, save_flag, plot_flag):
    """
    Tests with only microphone *or* source from a SOFA file
    """

    # make sure all the files are present
    # in case tests are run out of order
    download_sofa_files(overwrite=False, verbose=False)

    # create room
    room = pra.ShoeBox(
        room_dim,
        fs=16000,
        max_order=2,
        materials=all_materials,
        air_absorption=True,
        ray_tracing=False,
        min_phase=False,
    )

    # define source with figure_eight directivity
    directivity = DIRPATRir(
        orientation=DirectionVector(azimuth=0, colatitude=0, degrees=True),
        path=Path(DEFAULT_SOFA_PATH) / sofa_file_name,
        DIRPAT_pattern_enum=pattern_id,
        fs=16000,
    )

    # add source with figure_eight directivity
    if sofa_info[sofa_file_name]["type"] == "receivers":
        directivity_SRC = None
        directivity_MIC = directivity
    elif sofa_info[sofa_file_name]["type"] == "sources":
        directivity_SRC = directivity
        directivity_MIC = None
    else:
        raise ValueError("what!!")

    room.add_source([1.52, 0.883, 1.044], directivity=directivity_SRC)

    # add microphone in its null
    room.add_microphone([2.31, 1.65, 1.163], directivity=directivity_MIC)

    # Check set different orientation after intailization of the DIRPATRir class
    directivity.set_orientation(np.radians(0), np.radians(0))
    # directivity_SRC.set_orientation(np.radians(70), np.radians(34))

    room.compute_rir()

    rir_1_0 = room.rir[0][0]

    filename = "-".join([sofa_file_name.split(".")[0], pattern_id]) + "-oneside.npy"
    test_file_path = TEST_DATA / filename
    if save_flag:
        TEST_DATA.mkdir(exist_ok=True, parents=True)
        np.save(test_file_path, rir_1_0)
    elif test_file_path.exists():
        reference_data = np.load(test_file_path)
        print("Max diff.:", abs(reference_data - rir_1_0).max())
        print(
            "Rel diff.:",
            abs(reference_data - rir_1_0).max() / abs(reference_data).max(),
        )
        if plot_flag:
            fig, ax = plt.subplots(1, 1)
            ax.plot(rir_1_0, label="test")
            ax.plot(reference_data, label="ref")
            ax.legend()
            fig.savefig(test_file_path.with_suffix(".pdf"))
        assert np.allclose(reference_data, rir_1_0, atol=atol, rtol=rtol)
    else:
        warnings.warn("Did not find the reference data. Output was not checked.")


SOFA_TWO_SIDES_PARAMETERS = [
    (
        "Genelec_8020",
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
        "AKG_c480",
        "AKG_c480_c414_CUBE.sofa",
        False,
        False,
    ),
    (
        "Vibrolux_2x10inch",
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
        "AKG_c414K",
        "AKG_c480_c414_CUBE.sofa",
        False,
        False,
    ),
    (
        "Vibrolux_2x10inch",
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
        "EM_32_0",
        "EM32_Directivity.sofa",
        False,
        False,
    ),
    (
        "Genelec_8020",
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
        "EM_32_31",
        "EM32_Directivity.sofa",
        False,
        False,
    ),
]


@pytest.mark.parametrize(
    "src_pattern_id, src_sofa_file_name, mic_pattern_id, "
    "mic_sofa_file_name, save_flag, plot_flag",
    SOFA_TWO_SIDES_PARAMETERS,
)
def test_sofa_two_sides(
    src_pattern_id,
    src_sofa_file_name,
    mic_pattern_id,
    mic_sofa_file_name,
    save_flag,
    plot_flag,
):
    """
    Tests with only microphone *or* source from a SOFA file
    """

    # make sure all the files are present
    # in case tests are run out of order
    download_sofa_files(overwrite=False, verbose=False)

    # create room
    room = pra.ShoeBox(
        room_dim,
        fs=16000,
        max_order=2,
        materials=all_materials,
        air_absorption=True,
        ray_tracing=False,
        min_phase=False,
    )

    src_directivity = DIRPATRir(
        orientation=DirectionVector(azimuth=0, colatitude=0, degrees=True),
        path=Path(DEFAULT_SOFA_PATH) / src_sofa_file_name,
        DIRPAT_pattern_enum=src_pattern_id,
        fs=16000,
    )

    mic_directivity = DIRPATRir(
        orientation=DirectionVector(azimuth=0, colatitude=0, degrees=True),
        path=Path(DEFAULT_SOFA_PATH) / mic_sofa_file_name,
        DIRPAT_pattern_enum=mic_pattern_id,
        fs=16000,
    )

    room.add_source([1.52, 0.883, 1.044], directivity=src_directivity)

    # add microphone in its null
    room.add_microphone([2.31, 1.65, 1.163], directivity=mic_directivity)

    # Check set different orientation after intailization of the DIRPATRir class
    mic_directivity.set_orientation(np.radians(np.pi), np.radians(0))
    src_directivity.set_orientation(np.radians(0), np.radians(np.pi / 2.0))

    room.compute_rir()

    rir_1_0 = room.rir[0][0]

    filename = (
        "-".join(
            [
                src_sofa_file_name.split(".")[0],
                src_pattern_id,
                mic_sofa_file_name.split(".")[0],
                mic_pattern_id,
            ]
        )
        + "-twosides.npy"
    )
    test_file_path = TEST_DATA / filename
    if save_flag:
        TEST_DATA.mkdir(exist_ok=True, parents=True)
        np.save(test_file_path, rir_1_0)
    elif test_file_path.exists():
        reference_data = np.load(test_file_path)

        print("Max diff.:", abs(reference_data - rir_1_0).max())
        print(
            "Rel diff.:",
            abs(reference_data - rir_1_0).max() / abs(reference_data).max(),
        )

        if plot_flag:
            fig, ax = plt.subplots(1, 1)
            ax.plot(rir_1_0, label="test")
            ax.plot(reference_data, label="ref")
            ax.legend()
            fig.savefig(test_file_path.with_suffix(".pdf"))

        assert np.allclose(reference_data, rir_1_0, atol=atol, rtol=rtol)
    else:
        warnings.warn("Did not find the reference data. Output was not checked.")


SOFA_CARDIOID_PARAMETERS = [
    ("AKG_c480", "AKG_c480_c414_CUBE.sofa", False, False),
    ("AKG_c414K", "AKG_c480_c414_CUBE.sofa", False, False),
    ("AKG_c414N", "AKG_c480_c414_CUBE.sofa", False, False),
    ("AKG_c414S", "AKG_c480_c414_CUBE.sofa", False, False),
    ("AKG_c414A", "AKG_c480_c414_CUBE.sofa", False, False),
    ("EM_32_0", "EM32_Directivity.sofa", False, False),
    ("EM_32_31", "EM32_Directivity.sofa", False, False),
    ("Genelec_8020", "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa", False, False),
    (
        "Vibrolux_2x10inch",
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
        False,
        False,
    ),
]


@pytest.mark.parametrize(
    "pattern_id,sofa_file_name,save_flag,plot_flag", SOFA_CARDIOID_PARAMETERS
)
def test_sofa_and_cardioid(pattern_id, sofa_file_name, save_flag, plot_flag):
    """
    Tests with only microphone *or* source from a SOFA file
    """

    # make sure all the files are present
    # in case tests are run out of order
    download_sofa_files(overwrite=False, verbose=False)

    # create room
    room = pra.ShoeBox(
        room_dim,
        fs=16000,
        max_order=2,
        materials=all_materials,
        air_absorption=True,
        ray_tracing=False,
        min_phase=False,
    )

    # define source with figure_eight directivity
    directivity = DIRPATRir(
        orientation=DirectionVector(azimuth=0, colatitude=0, degrees=True),
        path=Path(DEFAULT_SOFA_PATH) / sofa_file_name,
        DIRPAT_pattern_enum=pattern_id,
        fs=16000,
    )

    # add source with figure_eight directivity
    if sofa_info[sofa_file_name]["type"] == "receivers":
        directivity_SRC = CardioidFamily(
            orientation=DirectionVector(azimuth=90, colatitude=90, degrees=True),
            pattern_enum=DirectivityPattern.FIGURE_EIGHT,
        )
        directivity_MIC = directivity
    elif sofa_info[sofa_file_name]["type"] == "sources":
        directivity_SRC = directivity
        directivity_MIC = CardioidFamily(
            orientation=DirectionVector(azimuth=90, colatitude=90, degrees=True),
            pattern_enum=DirectivityPattern.FIGURE_EIGHT,
        )
    else:
        raise ValueError("what!!")

    room.add_source([1.52, 0.883, 1.044], directivity=directivity_SRC)

    # add microphone in its null
    room.add_microphone([2.31, 1.65, 1.163], directivity=directivity_MIC)

    # Check set different orientation after intailization of the DIRPATRir class
    directivity.set_orientation(np.radians(0), np.radians(0))
    # directivity_SRC.set_orientation(np.radians(70), np.radians(34))

    room.compute_rir()

    rir_1_0 = room.rir[0][0]

    filename = "-".join([sofa_file_name.split(".")[0], pattern_id]) + "-cardioid.npy"
    test_file_path = TEST_DATA / filename
    if save_flag:
        TEST_DATA.mkdir(exist_ok=True, parents=True)
        np.save(test_file_path, rir_1_0)
    elif test_file_path.exists():
        reference_data = np.load(test_file_path)

        print("Max diff.:", abs(reference_data - rir_1_0).max())
        print(
            "Rel diff.:",
            abs(reference_data - rir_1_0).max() / abs(reference_data).max(),
        )

        if plot_flag:
            fig, ax = plt.subplots(1, 1)
            ax.plot(rir_1_0, label="test")
            ax.plot(reference_data, label="ref")
            ax.legend()
            fig.savefig(test_file_path.with_suffix(".pdf"))

        assert np.allclose(reference_data, rir_1_0, atol=atol, rtol=rtol)
    else:
        warnings.warn("Did not find the reference data. Output was not checked.")


PINV_PARAMETERS = [
    (30, 16, np.pi / 32, np.pi - np.pi / 32, "F", 5e-5, 1e-8),
    (30, 16, np.pi / 32, np.pi - np.pi / 32, "C", 5e-5, 1e-8),
    (30, 16, 1e-5, np.pi - np.pi / 32, "C", 5e-5, 1e-8),
    (30, 16, 1e-5, np.pi - np.pi / 32, "F", 5e-5, 1e-8),
    (30, 16, np.pi / 32, np.pi - 1e-5, "C", 5e-5, 1e-8),
    (30, 16, np.pi / 32, np.pi - 1e-5, "F", 5e-5, 1e-8),
    (35, 17, np.pi / 32, np.pi - np.pi / 32, "F", 5e-5, 1e-8),
    (24, 24, np.pi / 32, np.pi - np.pi / 32, "C", 5e-5, 1e-8),
    (40, 20, np.pi / 32, np.pi - 1e-5, "C", 5e-5, 1e-8),
    (40, 20, np.pi / 32, np.pi - 1e-5, "F", 5e-5, 1e-8),
]


@pytest.mark.parametrize(
    "n_azimuth, n_col, col_start, col_end, order, atol, rtol",
    PINV_PARAMETERS,
)
def test_weighted_pinv(n_azimuth, n_col, col_start, col_end, order, atol, rtol):
    azimuth = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    colatitude = np.linspace(col_start, col_end, n_col)
    A, C = np.meshgrid(azimuth, colatitude)
    alin = A.flatten(order=order)
    clin = C.flatten(order=order)

    Ysh = np.random.randn(alin.shape[0])

    points = np.array(
        [
            np.cos(alin) * np.sin(clin),
            np.sin(alin) * np.sin(clin),
            np.cos(clin),
        ]
    ).T

    Y_reg, w_reg = calculation_pinv_voronoi_cells(Ysh, clin, colatitude, len(azimuth))
    Y_gen, w_gen = calculation_pinv_voronoi_cells_general(Ysh, points)

    assert np.allclose(w_reg, w_gen, rtol=rtol, atol=atol)
    assert np.allclose(Y_reg, Y_gen, rtol=rtol, atol=atol)


if __name__ == "__main__":
    # generate the test files for regression testing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save", action="store_true", help="save the signal as a reference"
    )
    parser.add_argument(
        "--plot", action="store_true", help="plot the generated signals"
    )
    args = parser.parse_args()

    download_sofa_files(verbose=True)

    for params in SOFA_ONE_SIDE_PARAMETERS:
        new_params = params[:-2] + (args.save, args.plot)
        test_sofa_one_side(*new_params)

    for params in SOFA_TWO_SIDES_PARAMETERS:
        new_params = params[:-2] + (args.save, args.plot)
        test_sofa_two_sides(*new_params)

    for params in SOFA_CARDIOID_PARAMETERS:
        new_params = params[:-2] + (args.save, args.plot)
        test_sofa_and_cardioid(*new_params)

    for params in PINV_PARAMETERS:
        test_weighted_pinv(*params)
