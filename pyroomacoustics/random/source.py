# Utilities for generating random source positions.
# Copyright (C) 2019  Eric Bezzam
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

import numpy as np
from pyroomacoustics.random.distribution import DiscreteDistribution, \
    UniformDistribution, Distribution
from pyroomacoustics.doa.utils import spher2cart


class RandomSource(object):
    """
    Object to randomly sample a source position relative to a certain position.

    Parameters
    -----------
    distance_distrib : Distribution
        Distribution to sample distance of source.
    azimuth_distrib : Distribution
        Distribution to sample azimuth angle of source.
    elevation_distrib : Distribution
        Distribution to sample elevation angle of source.

    """
    def __init__(self,
                 distance_distrib=None,
                 azimuth_distrib=None,
                 elevation_distrib=None):

        # set distance distribution
        if distance_distrib is None:
            self.distance_distrib = DiscreteDistribution(
                values=[1, 2, 3, 4, 5, 6, 7],
                prob=[15, 22, 29, 21, 8, 3, 0.5]
            )
        else:
            self.distance_distrib = distance_distrib
        assert isinstance(self.distance_distrib, Distribution)

        # set azimuth distribution
        if azimuth_distrib is None:
            self.azimuth_distrib = UniformDistribution(
                vals_range=[-180, 180]
            )
        else:
            self.azimuth_distrib = azimuth_distrib
        assert isinstance(self.azimuth_distrib, Distribution)

        # set elevation distribution
        if elevation_distrib is None:
            self.elevation_distrib = UniformDistribution(
                vals_range=[45, 135]
            )
        else:
            self.elevation_distrib = elevation_distrib
        assert isinstance(self.elevation_distrib, Distribution)

    def sample(self, cartesian=True):
        """
        Sample a source location according to specified distributions.

        Parameters
        ----------
        cartesian : bool, optional
            Whether to return coordinates in cartesian form. Default is True.
        """
        distance = self.distance_distrib.sample()
        azimuth = self.azimuth_distrib.sample()
        elevation = self.elevation_distrib.sample()

        if cartesian:
            return spher2cart(
                r=distance,
                azimuth=azimuth,
                colatitude=elevation
            )
        else:
            return np.array([distance, azimuth, elevation])




