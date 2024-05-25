from pathlib import Path

import numpy as np
import scipy

try:
    import sofa

    has_sofa = True
except ImportError:
    has_sofa = False

import collections
import math

from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import griddata
from scipy.spatial import SphericalVoronoi, cKDTree

from .datasets import SOFADatabase
from .directivities import DirectionVector, Directivity
from .doa import (
    Grid,
    GridSphere,
    cart2spher,
    detect_regular_grid,
    fibonacci_spherical_sampling,
    spher2cart,
)
from .utilities import requires_matplotlib, resample

_DATA_SOFA_DIR = Path(__file__).parent / "data/sofa"

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


def cal_sph_basis(azimuth, colatitude, degree):  # theta_target,phi_target
    """
    Calculate a spherical basis matrix

    Parameters
    -----------
    azimuth: array_like
       Azimuth of the spherical coordinates of the grid points
    phi: array_like
       Colatitude spherical coordinates of the grid points
    degree:(int)
       spherical harmonic degree

    Return
    ------
    Ysh (np.array) shape: (no_of_nodes, (degree + 1)**2)
        Spherical harmonics basis matrix

    """
    # build linear array of indices
    #        0
    #     -1 0 1
    #  -2 -1 0 1 2
    # ...
    ms = []
    ns = []
    for i in range(degree + 1):
        for order in range(-i, i + 1):
            ms.append(order)
            ns.append(i)
    m, n = np.array([ms]), np.array([ns])

    # compute all the spherical harmonics at once
    Ysh = scipy.special.sph_harm(m, n, azimuth[:, None], colatitude[:, None])

    return Ysh


def _weighted_pinv(weights, Y, rcond=1e-2):
    return np.linalg.pinv(
        weights[:, None] * Y, rcond=rcond
    )  # rcond is inverse of the condition number


def calculation_pinv_voronoi_cells(Ysh, colatitude, colatitude_grid, len_azimuth_grid):
    """
    Weighted least square solution "Analysis and Synthesis of Sound-Radiation with Spherical Arrays: Franz Zotter Page 76"

    Calculation of pseudo inverse and voronoi cells for regular sampling in spherical coordinates

    Parameters
    -----------
    Ysh: (np.ndarray)
        Spherical harmonic basis matrix
    colatitude: (np.ndarray)
        The colatitudes of the measurements
    colatitude_grid: (int)
        the colatitudes of the grid lines
    len_azimuth_grid:
        The number of distinct azimuth values in the grid

    Returns:
    -------------------------------
    Ysh_tilda_inv : (np.ndarray)
        Weighted psuedo inverse of spherical harmonic basis matrix Ysh
    w_ : (np.ndarray)
        Weight on the original grid
    """
    # compute the areas of the voronoi regions of the grid analytically
    # assuming that the grid is regular in azimuth/colatitude
    res = (colatitude_grid[:-1] + colatitude_grid[1:]) / 2
    res = np.insert(res, len(res), np.pi)
    res = np.insert(res, 0, 0)
    w = -np.diff(np.cos(res)) * (2 * np.pi / len_azimuth_grid)
    w_dict = {t: ww for t, ww in zip(colatitude_grid, w)}

    # select the weights
    w_ = np.array([w_dict[col] for col in colatitude])
    w_ /= 4 * np.pi  # normalizing by unit sphere area

    return _weighted_pinv(w_, Ysh), w_


def calculation_pinv_voronoi_cells_general(Ysh, points):
    """
    Weighted least square solution "Analysis and Synthesis of Sound-Radiation with
    Spherical Arrays: Franz Zotter Page 76"

    Calculation of pseudo inverse and voronoi cells for arbitrary sampling of the sphere.

    Parameters
    -----------
    Ysh: (np.ndarray)
        Spherical harmonic basis matrix
    points: numpy.ndarray, (n_points, 3)
        The sampling points on the sphere

    Returns:
    -------------------------------
    Ysh_tilda_inv : (np.ndarray)
        Weighted pseudo inverse of spherical harmonic basis matrix Ysh
    w_ : (np.ndarray)
        Weight on the original grid
    """

    # The weights are the areas of the voronoi cells
    sv = SphericalVoronoi(points)
    w_ = sv.calculate_areas()
    w_ /= 4 * np.pi  # normalizing by unit sphere area

    Ysh_tilda_inv = np.linalg.pinv(
        w_[:, None] * Ysh, rcond=1e-2
    )  # rcond is inverse of the condition number

    return _weighted_pinv(w_, Ysh), w_


def spherical_interpolation(
    grid,
    impulse_responses,
    new_grid,
    spherical_harmonics_order=12,
    axis=-2,
    nfft=None,
):
    """
    Parameters
    ----------
    grid: pyroomacoustics.doa.GridSphere
        The grid of the measurements
    impulse_responses: numpy.ndarray, (..., n_measurements, ..., n_samples)
        The impulse responses to interpolate, the last axis is time and one other
        axis should have dimension matching the length of the grid. By default,
        it is assumed to be second from the end, but can be specified with the
        `axis` argument.
    new_grid: pyroomacoustics.doa.GridSphere
        Grid of points at which to interpolate
    spherical_harmonics_order: int
        The order of spherical harmonics to use for interpolation
    axis: int
        The axis of the grid in the impulse responses array
    nfft: int
        The length of the FFT to use for the interpolation (default ``n_samples``)
    """
    ir = np.swapaxes(impulse_responses, axis, -2)
    if nfft is None:
        nfft = ir.shape[-1]

    if len(grid) != ir.shape[-2]:
        raise ValueError(
            "The length of the grid should be the same as the number of impulse"
            f"responses provide (grid={len(grid)}, impulse response={ir.shape[-2]})"
        )

    # Calculate spherical basis for the original grid
    Ysh = cal_sph_basis(grid.azimuth, grid.colatitude, spherical_harmonics_order)

    # Calculate spherical basis for the target grid (fibonacci grid)
    Ysh_fibo = cal_sph_basis(
        new_grid.azimuth,
        new_grid.colatitude,
        spherical_harmonics_order,
    )

    # this will check if the points are on a regular grid.
    # If they are, then the azimuths and colatitudes of the grid
    # are returned
    regular_grid = detect_regular_grid(grid.azimuth, grid.colatitude)

    # calculate pinv and voronoi cells for least square solution for the whole grid
    if regular_grid is not None:
        Ysh_tilda_inv, w_ = calculation_pinv_voronoi_cells(
            Ysh,
            grid.colatitude,
            regular_grid.colatitude,
            len(regular_grid.azimuth),
        )
    else:
        Ysh_tilda_inv, w_ = calculation_pinv_voronoi_cells_general(
            Ysh, grid.cartesian.T
        )

    # Do the interpolation in the frequency domain

    # shape: (..., n_measurements, n_samples // 2 + 1)
    tf = np.fft.rfft(ir, axis=-1, n=nfft)

    g_tilda = w_[:, None] * tf

    gamma_full_scale = np.matmul(Ysh_tilda_inv, g_tilda)

    interpolated_original_grid = np.fft.irfft(
        np.matmul(Ysh, gamma_full_scale), n=nfft, axis=-1
    )
    interpolated_target_grid = np.fft.irfft(
        np.matmul(Ysh_fibo, gamma_full_scale), n=nfft, axis=-1
    )

    # restore the order of the axes
    interpolated_target_grid = np.swapaxes(interpolated_target_grid, -2, axis)
    interpolated_original_grid = np.swapaxes(interpolated_original_grid, -2, axis)

    return interpolated_target_grid, interpolated_original_grid


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


class MeasuredDirectivity(Directivity):
    """
    A class to store directivity patterns obtained by measurements.

    Parameters
    ----------
    orientation: DirectionVector
        The direction of where the directivity should point
    grid: doa.Grid
        The grid of unit vectors where the measurements were taken
    impulse_responses: np.ndarray, (n_grid, n_ir)
        The impulse responses corresponding to the grid points
    fs: int
        The sampling frequency of the impulse responses
    """

    def __init__(self, orientation, grid, impulse_responses, fs):
        super().__init__(orientation)
        assert isinstance(grid, Grid)
        self._original_grid = grid
        self._irs = impulse_responses
        self.fs = fs

        # set the initial orientation
        self.set_orientation(self._orientation)

    @property
    def is_impulse_response(self):
        return True

    @property
    def filter_len_ir(self):
        """Length of the impulse response in samples"""
        return self._irs.shape[-1]

    def set_orientation(self, orientation):
        """
        Set orientation of directivity pattern.

        Parameters
        ----------
        orientation : DirectionVector
            New direction for the directivity pattern.
        """
        assert isinstance(orientation, DirectionVector)
        n_a = orientation.get_azimuth(degrees=False)
        n_c = orientation.get_colatitude(degrees=False)

        R_y = np.array(
            [[np.cos(n_c), 0, np.sin(n_c)], [0, 1, 0], [-np.sin(n_c), 0, np.cos(n_c)]]
        )
        R_z = np.array(
            [[np.cos(n_a), -np.sin(n_a), 0], [np.sin(n_a), np.cos(n_a), 0], [0, 0, 1]]
        )
        res = np.matmul(R_z, R_y)

        # rotate the grid and re-build the KD-tree
        self._grid = GridSphere(
            cartesian_points=np.matmul(res, self._original_grid.cartesian)
        )
        # create the kd-tree
        self._kdtree = cKDTree(self._grid.spherical.T)

    def get_response(
        self, azimuth, colatitude=None, magnitude=False, frequency=None, degrees=True
    ):
        """
        Get response for provided angles and frequency.

        Parameters
        ----------
        azimuth: np.ndarray, (n_points,)
            The azimuth of the desired responses
        colatitude: np.ndarray, (n_points,)
            The colatitude of the desired responses
        magnitude: bool
            Ignored
        frequency: np.ndarray, (n_freq,)
            Ignored
        degrees: bool
            If ``True``, indicates that azimuth and colatitude are provided in degrees
        """
        if degrees:
            azimuth = np.radians(azimuth)
            colatitude = np.radians(colatitude)

        _, index = self._kdtree.query(np.column_stack((azimuth, colatitude)))
        return self._irs[index, :]

    @requires_matplotlib
    def plot(self, freq_bin=0, n_grid=100, ax=None, depth=False, offset=None):
        import matplotlib.pyplot as plt
        from matplotlib import cm

        cart = self._grid.cartesian.T
        length = np.abs(np.fft.rfft(self._irs, axis=-1)[:, freq_bin])

        # regrid the data on a 2D grid
        g = np.linspace(-1, 1, n_grid)
        AZ, COL = np.meshgrid(
            np.linspace(0, 2 * np.pi, n_grid), np.linspace(0, np.pi, n_grid // 2)
        )
        # multiply by 0.99 to make sure the interpolation grid is inside the convex hull
        # of the original points, otherwise griddata returns NaN
        shrink_factor = 0.99
        while True:
            X = np.cos(AZ) * np.sin(COL) * shrink_factor
            Y = np.sin(AZ) * np.sin(COL) * shrink_factor
            Z = np.cos(COL) * shrink_factor
            grid = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
            interp_len = griddata(cart, length, grid, method="linear")
            V = interp_len.reshape((n_grid // 2, n_grid))

            # there may be some nan
            if np.any(np.isnan(V)):
                shrink_factor *= 0.99
            else:
                break

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="3d")

        # Colour the plotted surface according to the sign of Y.
        cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("coolwarm"))
        cmap.set_clim(0, V.max())

        if depth:
            X *= V
            Y *= V
            Z *= V

        surf = ax.plot_surface(
            X, Y, Z, facecolors=cmap.to_rgba(V), linewidth=0, antialiased=False
        )

        return ax


class SOFADirectivityFactory:
    """
    This class reads measured directivities from a
    `SOFA <https://www.sofaconventions.org>`_ format file.
    Optionally, it can perform interpolation of the impulse responses onto a finer grid.
    The interpolation is done in the spherical harmonics domain.


    Parameters
    --------------
    path : (string)
        Path towards the specific DIRPAT file
    fs: (int)
        The desired sampling frequency. If the impulse responses were stored at
        a different sampling frequency, they are resampled at ``fs``.
    interp_order: (int)
        The order of spherical harmonics to use for interpolation.
        If ``None`` interpolation is not used.
    interp_n_points: (int)
        Number of points for the interpolation grid. The interpolation grid is a
        Fibonnaci pseudo-uniform sampling of the sphere.
    sofa_file_reader_callback: (callable)
        A callback function that reads the SOFA file and returns the impulse responses
        The signature should be the same as the function `open_sofa_file`
    mic_labels: (list of strings)
        List of labels for the microphones. If not provided, the labels are simply the
        indices of the microphones in the array
    source_labels: (list of strings)
        List of labels for the sources. If not provided, the labels are simply the
        indices of the measurements in the array
    """

    def __init__(
        self,
        path,
        fs=None,
        interp_order=None,
        interp_n_points=1000,
        sofa_file_reader_callback=None,
        mic_labels=None,
        source_labels=None,
    ):
        self.path = Path(path)
        self.mic_labels, self.source_labels = self._set_labels(
            self.path, mic_labels, source_labels
        )

        if sofa_file_reader_callback is None:
            sofa_file_reader_callback = open_sofa_file

        (
            self.impulse_responses,  # (n_sources, n_mics, taps)
            self.sources_loc,  # (3, n_sources), spherical coordinates
            self.mics_loc,  # (3, n_mics), cartesian coordinates
            self.fs,
        ) = sofa_file_reader_callback(
            path=self.path,
            fs=fs,
        )

        self.interp_order = interp_order
        self.interp_n_points = interp_n_points

        self._ir_interp_cache = {}
        if interp_order is not None:
            self.interp_grid = GridSphere(
                cartesian_points=fibonacci_spherical_sampling(n_points=interp_n_points)
            )
        else:
            self.interp_grid = None

    def _set_labels(self, path, mic_labels, src_labels):
        sofa_db = get_sofa_db()
        if path.stem in sofa_db:
            info = sofa_db[path.stem]
            if info.type == "microphones" and mic_labels is None:
                mic_labels = info.contains
            elif info.type == "sources" and src_labels is None:
                src_labels = info.contains
        return mic_labels, src_labels

    def _interpolate(self, type, mid, grid, impulse_responses):
        if self.interp_order is None:
            return grid, impulse_responses

        label = f"{type}_{mid}"

        if label not in self._ir_interp_cache:
            self._ir_interp_cache[label], _ = spherical_interpolation(
                grid,
                impulse_responses,
                self.interp_grid,
                spherical_harmonics_order=self.interp_order,
                axis=-2,
            )

        return self.interp_grid, self._ir_interp_cache[label]

    def _get_measurement_index(self, meas_id, labels):
        if isinstance(meas_id, int):
            return meas_id
        elif labels is not None:
            idx = labels.index(meas_id)
            if idx >= 0:
                return idx
            else:
                raise KeyError(f"Measurement id {meas_id} not found")

        raise ValueError(f"Measurement id {meas_id} not found")

    def get_microphone(self, measurement_id, orientation, offset=None):
        mid = self._get_measurement_index(measurement_id, self.mic_labels)

        # select the measurements corresponding to the mic id
        ir = self.impulse_responses[:, mid, :]
        src_grid = GridSphere(spherical_points=self.sources_loc[:2])

        mic_loc = self.mics_loc[:, mid]
        if offset is not None:
            mic_loc += offset

        # interpolate the IR
        grid, ir = self._interpolate("mic", mid, src_grid, ir)

        dir_obj = MeasuredDirectivity(orientation, grid, ir, self.fs)
        return mic_loc, dir_obj

    def get_source(self, measurement_id, orientation, offset=None):
        mid = self._get_measurement_index(measurement_id, self.source_labels)

        # select the measurements corresponding to the mic id
        ir = self.impulse_responses[mid, :, :]

        # here we need to swap the coordinate types
        mic_pos = np.array(cart2spher(self.mics_loc))
        mic_grid = GridSphere(spherical_points=mic_pos[:2])

        # source location
        src_loc = spher2cart(*self.sources_loc[:, mid])
        if offset is not None:
            src_loc += offset

        # interpolate the IR
        grid, ir = self._interpolate("source", mid, mic_grid, ir)

        dir_obj = MeasuredDirectivity(orientation, grid, ir, self.fs)
        return src_loc, dir_obj

    def create(self, orientation):
        return MeasuredDirectivity(
            orientation, self.grid, self.impulse_responses, self.fs
        )
