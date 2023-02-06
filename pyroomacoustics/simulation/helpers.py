# Some helper functions for room simulation
# Copyright (C) 2023-2019  Robin Scheibler, Ivan Dokmanic, Sidney Barthe, Cyril Cadoux
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
import scipy.spatial as spatial

from .. import libroom
from ..parameters import eps


def wall_factory(corners, absorption, scattering, name=""):
    """Calls the correct method according to wall dimension"""
    if corners.shape[0] == 3:
        return libroom.Wall(corners, absorption, scattering, name)
    elif corners.shape[0] == 2:
        return libroom.Wall2D(corners, absorption, scattering, name)
    else:
        raise ValueError("Rooms can only be 2D or 3D")


def sequence_generation(volume, duration, c, fs, max_rate=10000):
    # repeated constant
    fpcv = 4 * np.pi * c**3 / volume

    # initial time
    t0 = ((2 * np.log(2)) / fpcv) ** (1.0 / 3.0)
    times = [t0]

    while times[-1] < t0 + duration:
        # uniform random variable
        z = np.random.rand()
        # rate of the point process at this time
        mu = np.minimum(fpcv * (t0 + times[-1]) ** 2, max_rate)
        # time interval to next point
        dt = np.log(1 / z) / mu

        times.append(times[-1] + dt)

    # convert from continuous to discrete time

    indices = (np.array(times) * fs).astype(np.int64)
    seq = np.zeros(indices[-1] + 1)
    seq[indices] = np.random.choice([1, -1], size=len(indices))

    return seq


def find_non_convex_walls(walls):
    """
    Finds the walls that are not in the convex hull

    Parameters
    ----------
    walls: list of Wall objects
        The walls that compose the room

    Returns
    -------
    list of int
        The indices of the walls no in the convex hull
    """

    all_corners = []
    for wall in walls[1:]:
        all_corners.append(wall.corners.T)
    X = np.concatenate(all_corners, axis=0)
    convex_hull = spatial.ConvexHull(X, incremental=True)

    # Now we need to check which walls are on the surface
    # of the hull
    in_convex_hull = [False] * len(walls)
    for i, wall in enumerate(walls):
        # We check if the center of the wall is co-linear or co-planar
        # with a face of the convex hull
        point = np.mean(wall.corners, axis=1)

        for simplex in convex_hull.simplices:
            if point.shape[0] == 2:
                # check if co-linear
                p0 = convex_hull.points[simplex[0]]
                p1 = convex_hull.points[simplex[1]]
                if libroom.ccw3p(p0, p1, point) == 0:
                    # co-linear point add to hull
                    in_convex_hull[i] = True

            elif point.shape[0] == 3:
                # Check if co-planar
                p0 = convex_hull.points[simplex[0]]
                p1 = convex_hull.points[simplex[1]]
                p2 = convex_hull.points[simplex[2]]

                normal = np.cross(p1 - p0, p2 - p0)
                if np.abs(np.inner(normal, point - p0)) < eps:
                    # co-planar point found!
                    in_convex_hull[i] = True

    return [i for i in range(len(walls)) if not in_convex_hull[i]]
