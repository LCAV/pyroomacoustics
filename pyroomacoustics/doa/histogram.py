# This module provides a class to plot histograms of data on the sphere.
# Copyright (C) 2024  Robin Scheibler
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
A class to make collect histograms of data distributed on the sphere.
"""
import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree

from .doa import GridSphere


class SphericalHistogram:
    def __init__(self, n_bins, dim=3, enable_peak_finding=False):

        self._n_dim = 3
        self._n_bins = n_bins

        if self.n_dim == 3:
            self._grid = GridSphere(
                n_points=self.n_bins, enable_peak_finding=enable_peak_finding
            )
        else:
            raise NotImplementedError("Only 3D histogram has been implemented")

        # we need to know the area of each bin
        self._voronoi = SphericalVoronoi(self._grid.cartesian.T)
        self._areas = self._voronoi.calculate_areas()

        # now we also need a KD-tree to do nearest neighbor search
        self._kd_tree = cKDTree(self._grid.cartesian.T)

        # the counter variables for every bin
        self._bins = np.zeros(self.n_bins, dtype=np.int)

        # the total number of points in the histogram
        self._total_count = 0

        # we cache the histogram bins
        self._cache_dirty = False
        self._cache_histogram = np.zeros(self.n_bins)

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def n_bins(self):
        return self._n_bins

    @property
    def histogram(self):
        if self._cache_dirty:
            # if the cache is dirty, we need to recompute
            Z = np.sum(self._areas * self._bins)  # partitioning constant
            self._cache_histogram[:] = self._bins / Z
            self._cache_dirty = False

        return self._cache_histogram

    @property
    def raw_counts(self):
        return self._bins

    @property
    def total_count(self):
        return self._total_count

    def find_peak(self, *args, **kwargs):
        return self._grid.find_peaks(self, *args, **kwargs)

    def plot(self):
        self._grid.set_values(self.histogram)
        self._grid.plot_old()

    def push(self, points):
        """
        Add new data into the histogram

        Parameters
        ----------
        points: array_like, shape (n_dim, n_points)
            The points to add to the histogram
        """
        self._total_count += points.shape[1]
        self._cache_dirty = True

        _, matches = self._kd_tree.query(points.T)
        bin_indices, counts = np.unique(matches, return_counts=True)
        self._bins[bin_indices] += counts
