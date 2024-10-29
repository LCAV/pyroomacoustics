# Some classes to apply rotate objects or indicate directions in 3D space.
# Copyright (C) 2022-2024  Prerak Srivastava, Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# You should have received a copy of the MIT License along with this program. If
# not, see <https://opensource.org/licenses/MIT>.
r"""
Source and microphone directivities can be measured in an anechoic chamber.
Such measurements result in a collection of impulse responses or transfer functions
each associated with a specific source and receiver (i.e., microphone) location.
The `SOFA file format <https://www.sofaconventions.org>`_ has been proposed
as a standard for the storage of such measurements.

This sub-module offers a way to read such measurements from (SOFA) files and
use the measurement to obtain a more faithful simulation.

The workhorse of this module is the class :py:class:`MeasuredDirectivityFile`
which reads the content of a file and standardize the data for futher use.
A single SOFA file can contain multiple measurements (for example corresponding
to different devices). The class provies a method to retrieve measurements
from individual sources and turn them into a py:class:`MeasuredDirectivity` object
that can be used to create a py:class:`pyroomacoustics.MicrophoneArray` object
with this directivity.

Such measurements do not provide impulse responses for every possible impinging
direction. Instead, during simulation the impulse response closest to the
desired direction is used instead. To avoid sharp transitions, the py:class:`MeasuredDirectivityFile`
provides an interpolation method in the spherical harmonics domain.
This can be activated by providing an order for the interpolation, e.g, `interp_order=12`.

Here is an example of loading a head-related transfer function and load
the directivities for left and right ears of a dummy head HRTF.

.. code-block:: python

    from pyroomacoustics.directivities import MeasuredDirectivityFile, Rotation3D

    # the file reader object reads the file and optionally performs interpolation
    # if the file contains multiple directivities, they are all read
    hrtf = MeasuredDirectivityFile(
        path="mit_kemar_normal_pinna.sofa",  # SOFA file is in the database
        fs=fs,
        interp_order=12,
        interp_n_points=1000,
    )

    # orientations can be provided as rotation matrices
    orientation = Rotation3D([colatitude_deg, azimuth_deg], "yz", degrees=True)

    # we can then choose which directivities we want from the file
    dir_left = hrtf.get_mic_directivity("left", orientation=orientation)
    dir_right = hrtf.get_mic_directivity("right", orientation=orientation)

"""
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from ..datasets import SOFADatabase
from ..doa import Grid, GridSphere, cart2spher, fibonacci_spherical_sampling, spher2cart
from ..utilities import requires_matplotlib
from .base import Directivity
from .direction import Rotation3D
from .interp import spherical_interpolation
from .sofa import open_sofa_file


class MeasuredDirectivity(Directivity):
    """
    A class to store directivity patterns obtained by measurements.

    Parameters
    ----------
    orientation: Rotation3D
        A rotation to apply to the pattern
    grid: doa.Grid
        The grid of unit vectors where the measurements were taken
    impulse_responses: np.ndarray, (n_grid, n_ir)
        The impulse responses corresponding to the grid points
    fs: int
        The sampling frequency of the impulse responses
    """

    def __init__(self, orientation, grid, impulse_responses, fs):
        if not isinstance(orientation, Rotation3D):
            raise ValueError("Orientation must be a Rotation3D object")

        if not isinstance(grid, Grid):
            raise ValueError("Grid must be a Grid object")

        self._original_grid = grid
        self._irs = impulse_responses
        self.fs = fs

        # set the initial orientation
        self.set_orientation(orientation)

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
        orientation : Rotation3D
            New direction for the directivity pattern.
        """
        if not isinstance(orientation, Rotation3D):
            raise ValueError("Orientation must be a Rotation3D object")

        self._orientation = orientation

        # rotate the grid and re-build the KD-tree
        self._grid = GridSphere(
            cartesian_points=self._orientation.rotate(self._original_grid.cartesian)
        )
        # create the kd-tree
        self._kdtree = cKDTree(self._grid.cartesian.T)

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

        cart = spher2cart(azimuth, colatitude)

        _, index = self._kdtree.query(cart.T)
        return self._irs[index, :]

    @requires_matplotlib
    def plot(self, freq_bin=0, n_grid=100, ax=None, depth=False, offset=None):
        """
        Plot the directivity pattern at a given frequency.

        Parameters
        ----------
        freq_bin: int
            The frequency bin to plot
        n_grid: int
            The number of points to use for the interpolation grid
        ax: matplotlib.axes.Axes, optional
            The axes to plot on. If not provided, a new figure is created
        depth: bool
            If ``True``, directive response is both depicted by color and depth
            of the surface. If ``False``, then only the color map denotes the
            intensity. (default ``False``)
        offset: float
            An offset to apply to the directivity pattern

        Returns
        -------
        ax: matplotlib.axes.Axes
            The axes on which the directivity is plotted
        """
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


class MeasuredDirectivityFile:
    """
    This class reads measured directivities from a
    `SOFA`_ format file.
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
    file_reader_callback: (callable)
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
        file_reader_callback=None,
        mic_labels=None,
        source_labels=None,
    ):
        self.path = Path(path)

        if file_reader_callback is None:
            # default reader is for SOFA files
            file_reader_callback = open_sofa_file

        (
            self.impulse_responses,  # (n_sources, n_mics, taps)
            self.fs,
            self.source_locs,  # (3, n_sources), spherical coordinates
            self.mic_locs,  # (3, n_mics), cartesian coordinates
            src_labels_file,
            mic_labels_file,
        ) = file_reader_callback(
            path=self.path,
            fs=fs,
        )

        if mic_labels is None:
            self.mic_labels = mic_labels_file
        else:
            if len(mic_labels) != self.mic_locs.shape[1]:
                breakpoint()
                raise ValueError(
                    f"Number of labels provided ({len(mic_labels)}) does not match the "
                    f"number of microphones ({self.mic_locs.shape[1]})"
                )
            self.mic_labels = mic_labels

        if source_labels is None:
            self.source_labels = src_labels_file
        else:
            if len(source_labels) != self.source_locs.shape[1]:
                raise ValueError(
                    f"Number of labels provided ({len(source_labels)}) does not match "
                    f"the number of sources ({self.source_locs.shape[1]})"
                )
            self.source_labels = source_labels

        self.interp_order = interp_order
        self.interp_n_points = interp_n_points

        self._ir_interp_cache = {}
        if interp_order is not None:
            self.interp_grid = GridSphere(
                cartesian_points=fibonacci_spherical_sampling(n_points=interp_n_points)
            )
        else:
            self.interp_grid = None

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

    def get_mic_position(self, measurement_id):
        """
        Get the position of source with id `measurement_id`

        Parameters
        ----------
        measurement_id: int or str
            The id of the source
        """
        mid = self._get_measurement_index(measurement_id, self.mic_labels)

        if not (0 <= mid < self.mic_locs.shape[1]):
            raise ValueError(f"Microphone id {mid} not found")

        return self.mic_locs[:, mid]

    def get_source_position(self, measurement_id):
        """
        Get the position of source with id `measurement_id`

        Parameters
        ----------
        measurement_id: int or str
            The id of the source
        """
        mid = self._get_measurement_index(measurement_id, self.source_labels)

        if not (0 <= mid < self.source_locs.shape[1]):
            raise ValueError(f"Source id {mid} not found")

        # convert to cartesian since the sources are stored by
        # default in spherical coordinates
        pos = spher2cart(*self.source_locs[:, mid])

        return pos

    def get_mic_directivity(self, measurement_id, orientation):
        """
        Get a directivity for a microphone

        Parameters
        ----------
        measurement_id: int or str
            The id of the microphone
        orientation: Rotation3D
            The orientation of the directivity pattern
        """
        mid = self._get_measurement_index(measurement_id, self.mic_labels)

        if not (0 <= mid < self.mic_locs.shape[1]):
            raise ValueError(f"Microphone id {mid} not found")

        # select the measurements corresponding to the mic id
        ir = self.impulse_responses[:, mid, :]
        src_grid = GridSphere(spherical_points=self.source_locs[:2])

        # interpolate the IR
        grid, ir = self._interpolate("mic", mid, src_grid, ir)

        dir_obj = MeasuredDirectivity(orientation, grid, ir, self.fs)
        return dir_obj

    def get_source_directivity(self, measurement_id, orientation):
        """
        Get a directivity for a source

        Parameters
        ----------
        measurement_id: int or str
            The id of the source
        orientation: Rotation3D
            The orientation of the directivity pattern
        """
        mid = self._get_measurement_index(measurement_id, self.source_labels)

        if not (0 <= mid < self.source_locs.shape[1]):
            raise ValueError(f"Source id {mid} not found")

        # select the measurements corresponding to the mic id
        ir = self.impulse_responses[mid, :, :]

        # here we need to swap the coordinate types
        mic_pos = np.array(cart2spher(self.mic_locs))
        mic_grid = GridSphere(spherical_points=mic_pos[:2])

        # interpolate the IR
        grid, ir = self._interpolate("source", mid, mic_grid, ir)

        dir_obj = MeasuredDirectivity(orientation, grid, ir, self.fs)
        return dir_obj
