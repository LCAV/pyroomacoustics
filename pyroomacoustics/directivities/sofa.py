from pathlib import Path

import numpy as np

try:
    import sofa

    has_sofa = True
except ImportError:
    has_sofa = False


from ..datasets import SOFADatabase
from ..doa import cart2spher, spher2cart
from ..utilities import resample

DIRPAT_FILES = [
    "Soundfield_ST450_CUBE",
    "AKG_c480_c414_CUBE",
    "Oktava_MK4012_CUBE",
    "LSPs_HATS_GuitarCabinets_Akustikmessplatz",
]


def get_sofa_db():
    # we want to avoid loading the database multiple times
    global sofa_db
    try:
        return sofa_db
    except NameError:
        sofa_db = SOFADatabase()
        return sofa_db


def _resolve_sofa_path(path):
    path = Path(path)

    if path.exists():
        return path

    sofa_db = get_sofa_db()
    if path.stem in sofa_db:
        return Path(sofa_db[path.stem].path)

    raise ValueError(f"SOFA file {path} could not be found")


def open_sofa_file(path, fs=16000):
    """
    Open a SOFA file and read the impulse responses

    Parameters
    ----------
    path: str or Path
        Path to the SOFA file
    fs: int, optional
        The desired sampling frequency. If the impulse responses were stored at
        a different sampling frequency, they are resampled at ``fs``.
    """
    # Memo for notation of SOFA dimensions
    # From: https://www.sofaconventions.org/mediawiki/index.php/SOFA_conventions#AnchorDimensions
    # M 	number of measurements 	integer >0
    # R 	number of receivers or harmonic coefficients describing receivers 	integer >0
    # E 	number of emitters or harmonic coefficients describing emitters 	integer >0
    # N 	number of data samples describing one measurement 	integer >0
    # S 	number of characters in a string 	integer ≥0
    # I 	singleton dimension, constant 	always 1
    # C 	coordinate triplet, constant 	always 3

    # Open DirPat database
    if not has_sofa:
        raise ValueError(
            "The package 'python-sofa' needs to be installed to call this function. Install by doing `pip install python-sofa`"
        )

    path = _resolve_sofa_path(path)

    file_sofa = sofa.Database.open(path)

    # we have a special case for DIRPAT files because they need surgery
    if path.stem in DIRPAT_FILES:
        return _read_dirpat(file_sofa, path.name, fs)

    conv_name = file_sofa.convention.name

    if conv_name == "SimpleFreeFieldHRIR":
        return _read_simple_free_field_hrir(file_sofa, fs)

    elif conv_name == "GeneralFIR":
        return _read_general_fir(file_sofa, fs)

    else:
        raise NotImplementedError(f"SOFA convention {conv_name} not implemented")


def _parse_locations(sofa_pos, target_format):
    """
    Reads and normalize a position stored in a SOFA file

    Parameters
    ----------
    sofa_pos:
        SOFA position object
    target_format:
        One of 'spherical' or 'cartesian'. For 'spherical', the
        angles are always in radians

    Returns
    -------
    A numpy array in the correct format
    """

    if target_format not in ("spherical", "cartesian"):
        raise ValueError("Target format should be 'spherical' or 'cartesian'")

    # SOFA dimensions
    dim = sofa_pos.dimensions()

    # source positions
    pos = sofa_pos.get_values()

    if len(dim) == 3 and dim[-1] == "I":
        pos = pos[..., 0]
        dim = dim[:-1]

    # get units
    pos_units = sofa_pos.Units
    if "," in pos_units:
        pos_units = pos_units.split(",")
        pos_units = [p.strip() for p in pos_units]
    else:
        pos_units = [pos_units] * pos.shape[1]

    pos_type = sofa_pos.Type

    if pos_type == "cartesian":
        if any([p != "metre" for p in pos_units]):
            raise ValueError(f"Found unit '{pos_units}' in SOFA file")

        if target_format == "spherical":
            return np.array(cart2spher(pos.T))
        else:
            return pos

    elif pos_type == "spherical":
        azimuth = pos[:, 0] if pos_units[0] != "degree" else np.deg2rad(pos[:, 0])
        colatitude = pos[:, 1] if pos_units[0] != "degree" else np.deg2rad(pos[:, 1])
        distance = pos[:, 2]

        if np.any(colatitude < 0.0):
            # it looks like the data is using elevation format
            colatitude = np.pi / 2.0 - colatitude

        if target_format == "cartesian":
            return spher2cart(azimuth, colatitude, distance)
        else:
            return np.array([azimuth, colatitude, distance])

    else:
        raise NotImplementedError(f"{pos_type} not implemented")


def _read_simple_free_field_hrir(file_sofa, fs):
    """
    Reads the HRIRs stored in a SOFA file with the SimpleFreeFieldHRIR convention

    Parameters
    ----------
    file_sofa: SOFA object
        Path to the SOFA file
    fs: int
        The desired sampling frequency. If the impulse responses were stored at
        a different sampling frequency, they are resampled at ``fs``

    Returns
    -------
    ir: np.ndarray
        The impulse responses in format ``(n_sources, n_mics, taps)``
    source_dir: np.ndarray
        The direction of the sources in spherical coordinates
        ``(3, n_sources)`` where the first row is azimuth and the second is colatitude
        and the third is distance
    rec_loc: np.ndarray
        The location of the receivers in cartesian coordinates with respect to the
        origin of the SOFA file
    fs: int
        The sampling frequency of the impulse responses
    """
    # read the mesurements (source direction, receiver location, taps)
    msr = file_sofa.Data.IR.get_values()

    # downsample the fir filter.
    fs_file = file_sofa.Data.SamplingRate.get_values()[0]
    if fs is None:
        fs = fs_file
    else:
        msr = resample(msr, fs_file, fs)

    # Source positions
    source_loc = _parse_locations(file_sofa.Source.Position, target_format="spherical")

    # Receivers locations (i.e., "ears" for HRIR)
    rec_loc = _parse_locations(file_sofa.Receiver.Position, target_format="cartesian")

    return msr, source_loc, rec_loc, fs


def _read_general_fir(file_sofa, fs):
    # read the mesurements (source direction, receiver location, taps)
    msr = file_sofa.Data.IR.get_values()

    # downsample the fir filter.
    fs_file = file_sofa.Data.SamplingRate.get_values()[0]
    if fs is None:
        fs = fs_file
    else:
        msr = resample(msr, fs_file, fs)

    # Source positions: (azimuth, colatitude, distance)
    source_loc = _parse_locations(file_sofa.Source.Position, target_format="spherical")

    # Receivers locations (i.e., "ears" for HRIR)
    rec_loc = _parse_locations(file_sofa.Receiver.Position, target_format="cartesian")

    return msr, source_loc, rec_loc, fs


def _read_dirpat(file_sofa, filename, fs=None):
    # read the mesurements
    msr = file_sofa.Data.IR.get_values()  # (n_sources, n_mics, taps)

    # downsample the fir filter.
    fs_file = file_sofa.Data.SamplingRate.get_values()[0]
    if fs is None:
        fs = fs_file
    else:
        msr = resample(msr, fs_file, fs)

    # Receiver positions
    mic_pos = file_sofa.Receiver.Position.get_values()  # (3, n_mics)
    mic_pos_units = file_sofa.Receiver.Position.Units.split(",")

    # Source positions
    src_pos = file_sofa.Source.Position.get_values()
    src_pos_units = file_sofa.Source.Position.Units.split(",")

    # There is a bug in the DIRPAT measurement files where the array of
    # measurement locations were not flattened correctly
    src_pos_units[0:1] = "radian"
    if filename == "LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa":
        # this is a source file
        mic_pos_RS = np.reshape(mic_pos, [36, -1, 3])
        mic_pos = np.swapaxes(mic_pos_RS, 0, 1).reshape([mic_pos.shape[0], -1])

        if np.any(mic_pos[:, 1] < 0.0):
            # it looks like the data is using elevation format
            mic_pos[:, 1] = np.pi / 2.0 - mic_pos[:, 1]

        # by convention, we keep the microphone locations in cartesian coordinates
        mic_pos = spher2cart(*mic_pos.T).T

        # create source locations, they are all at the center
        src_pos = np.zeros((msr.shape[0], 3))
    else:
        src_pos_RS = np.reshape(src_pos, [30, -1, 3])
        src_pos = np.swapaxes(src_pos_RS, 0, 1).reshape([src_pos.shape[0], -1])

        if np.any(src_pos[:, 1] < 0.0):
            # it looks like the data is using elevation format
            src_pos[:, 1] = np.pi / 2.0 - src_pos[:, 1]

        # create fake microphone locations, they are all at the center
        mic_pos = np.zeros((msr.shape[1], 3))

    return msr, src_pos.T, mic_pos.T, fs
