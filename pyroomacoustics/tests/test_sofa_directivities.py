"""
This is a regression test for the SOFA source and receiver measured directivity patterns

The tests compare the output of the simulation with some pre-generated samples.

To generate the samples run this file: `python ./test_sofa_directivities.py`
"""
import os
import warnings
from pathlib import Path

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

sofa_info = get_sofa_db_info()
supported_sofa = [name for name, info in sofa_info.items() if info["supported"] == True]

TEST_DATA = Path(__file__).parent / "data"

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
    ("AKG_c480", "AKG_c480_c414_CUBE.sofa", False),
    ("AKG_c414K", "AKG_c480_c414_CUBE.sofa", False),
    ("AKG_c414N", "AKG_c480_c414_CUBE.sofa", False),
    ("AKG_c414S", "AKG_c480_c414_CUBE.sofa", False),
    ("AKG_c414A", "AKG_c480_c414_CUBE.sofa", False),
    ("EM_32_0", "EM32_Directivity.sofa", False),
    ("EM_32_31", "EM32_Directivity.sofa", False),
    ("Genelec_8020", "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa", False),
    ("Vibrolux_2x10inch", "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa", False),
]


@pytest.mark.parametrize(
    "pattern_id,sofa_file_name,save_flag", SOFA_ONE_SIDE_PARAMETERS
)
def test_sofa_one_side(pattern_id, sofa_file_name, save_flag):
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
        orientation=DirectionVector(azimuth=90, colatitude=90, degrees=True),
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
        assert np.allclose(reference_data, rir_1_0)
    else:
        warnings.warn("Did not find the reference data. Output was not checked.")


SOFA_TWO_SIDES_PARAMETERS = [
    (
        "Genelec_8020",
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
        "AKG_c480",
        "AKG_c480_c414_CUBE.sofa",
        False,
    ),
    (
        "Vibrolux_2x10inch",
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
        "AKG_c414K",
        "AKG_c480_c414_CUBE.sofa",
        False,
    ),
    (
        "Vibrolux_2x10inch",
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
        "EM_32_0",
        "EM32_Directivity.sofa",
        False,
    ),
    (
        "Genelec_8020",
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
        "EM_32_31",
        "EM32_Directivity.sofa",
        False,
    ),
]


@pytest.mark.parametrize(
    "src_pattern_id, src_sofa_file_name, mic_pattern_id, mic_sofa_file_name, save_flag",
    SOFA_TWO_SIDES_PARAMETERS,
)
def test_sofa_two_sides(
    src_pattern_id, src_sofa_file_name, mic_pattern_id, mic_sofa_file_name, save_flag
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
        orientation=DirectionVector(azimuth=90, colatitude=90, degrees=True),
        path=Path(DEFAULT_SOFA_PATH) / src_sofa_file_name,
        DIRPAT_pattern_enum=src_pattern_id,
        fs=16000,
    )

    mic_directivity = DIRPATRir(
        orientation=DirectionVector(azimuth=90, colatitude=90, degrees=True),
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
        assert np.allclose(reference_data, rir_1_0)
    else:
        warnings.warn("Did not find the reference data. Output was not checked.")


SOFA_CARDIOID_PARAMETERS = [
    ("AKG_c480", "AKG_c480_c414_CUBE.sofa", False),
    ("AKG_c414K", "AKG_c480_c414_CUBE.sofa", False),
    ("AKG_c414N", "AKG_c480_c414_CUBE.sofa", False),
    ("AKG_c414S", "AKG_c480_c414_CUBE.sofa", False),
    ("AKG_c414A", "AKG_c480_c414_CUBE.sofa", False),
    ("EM_32_0", "EM32_Directivity.sofa", False),
    ("EM_32_31", "EM32_Directivity.sofa", False),
    ("Genelec_8020", "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa", False),
    ("Vibrolux_2x10inch", "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa", False),
]


@pytest.mark.parametrize(
    "pattern_id,sofa_file_name,save_flag", SOFA_CARDIOID_PARAMETERS
)
def test_sofa_and_cardioid(pattern_id, sofa_file_name, save_flag):
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
        orientation=DirectionVector(azimuth=270, colatitude=90, degrees=True),
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
        assert np.allclose(reference_data, rir_1_0)
    else:
        warnings.warn("Did not find the reference data. Output was not checked.")


if __name__ == "__main__":
    download_sofa_files(verbose=True)
    for params in SOFA_ONE_SIDE_PARAMETERS:
        new_params = params[:-1] + (True,)
        test_sofa_one_side(*new_params)
    for params in SOFA_TWO_SIDES_PARAMETERS:
        new_params = params[:-1] + (True,)
        test_sofa_two_sides(*new_params)
    for params in SOFA_CARDIOID_PARAMETERS:
        new_params = params[:-1] + (True,)
        test_sofa_and_cardioid(*new_params)
