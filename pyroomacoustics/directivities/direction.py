# Some classes to apply rotate objects or indicate directions in 3D space.
# Copyright (C) 2020-2024  Robin Scheibler, Satvik Dixit, Eric Bezzam
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
Using directivities makes sources and microphones having a different response
depending on the location of other objects. This means that their orientation
in 3D space matters.

Some types of directivities such as ``CardioidFamily`` and derived classes
are defined only by a vector (i.e., direction). The response is then symmetric
around the axis defined by this vector.

However, in general, not all directivities are symmetric in this way.
For the general case, the orientation can be defined by `Euler angles <https://en.wikipedia.org/wiki/Euler_angles>`_.
This is implemented in the class :py:class:`pyroomacoustics.direction.Rotation3D`.
"""
import numpy as np


class DirectionVector(object):
    """
    Object for representing direction vectors in 3D, parameterized by an azimuth and colatitude
    angle.

    Parameters
    ----------
    azimuth : float
    colatitude : float, optional
        Default to PI / 2, only XY plane.
    degrees : bool
        Whether provided values are in degrees (True) or radians (False).
    """

    def __init__(self, azimuth, colatitude=None, degrees=True):
        if degrees is True:
            azimuth = np.radians(azimuth)
            if colatitude is not None:
                colatitude = np.radians(colatitude)
        self._azimuth = azimuth
        if colatitude is None:
            colatitude = np.pi / 2
        assert colatitude <= np.pi and colatitude >= 0

        self._colatitude = colatitude

        self._unit_v = np.array(
            [
                np.cos(self._azimuth) * np.sin(self._colatitude),
                np.sin(self._azimuth) * np.sin(self._colatitude),
                np.cos(self._colatitude),
            ]
        )

    def get_azimuth(self, degrees=False):
        if degrees:
            return np.degrees(self._azimuth)
        else:
            return self._azimuth

    def get_colatitude(self, degrees=False):
        if degrees:
            return np.degrees(self._colatitude)
        else:
            return self._colatitude

    @property
    def unit_vector(self):
        """Direction vector in cartesian coordinates."""
        return self._unit_v


def _make_rot_matrix(a, axis):
    """
    Compute the rotation matrix for a single rotation around the z-axis.

    .. reference:
        https://en.wikipedia.org/wiki/Rotation_matrix
    """
    if not 0 <= axis <= 2:
        raise ValueError("Axis must be 0, 1, or 2.")
    if axis == 2:
        return np.array(
            [
                [np.cos(a), -np.sin(a), 0],
                [np.sin(a), np.cos(a), 0],
                [0, 0, 1],
            ]
        )
    elif axis == 1:
        return np.array(
            [
                [np.cos(a), 0, np.sin(a)],
                [0, 1, 0],
                [-np.sin(a), 0, np.cos(a)],
            ]
        )
    else:
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(a), -np.sin(a)],
                [0, np.sin(a), np.cos(a)],
            ]
        )


_axis_map = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


class Rotation3D:
    """
    An object representing 3D rotations by their
    `Euler angles <https://en.wikipedia.org/wiki/Euler_angles>`_.

    A rotation in 3D space can be fully described by 3 angles (i.e., the Euler angles). Each rotation is applied
    around one of the three axes and there are 12 possible ways of pickinig the order or the rotations.

    This class can apply full or partial rotations to sets of points.

    The angles are provided as an array ``angles``. The axes of rotation for
    the angles are provided in ``rot_order`` as a string of three characters
    out of ``["x", "y", "z"]`` or a list of three integers out of ``[0, 1,
    2]``. Each axis can be repeated. To obtain full rotations, the same
    axis should not be used twice in a row.

    By default, the angles are specified in degrees. This can be changed by setting
    ``degrees=False``. In that case, the angles are assumed to be in radians.

    Parameters
    ----------
    angles : array_like
        An array containing between 0 and 3 angles.
    rot_order : str of List[int]
        The order of the rotations. The default is "zyx".
        The order indicates around which axis the rotation is performed.
        For example, "zyx" means that the rotation is first around the z-axis,
        then the y-axis, and finally the x-axis.
    degrees : bool
        Whether the angles are in degrees (True) or radians (False).
    """

    def __init__(self, angles, rot_order="zyx", degrees=True):
        angles = np.array(angles)

        if not 0 <= len(angles) <= 3:
            raise ValueError("The number of angles must be between 0 and 3.")

        if degrees:
            angles = np.radians(angles)

        self._angles = angles

        self._rot_order = []
        for ax in rot_order:
            if ax not in _axis_map:
                raise ValueError(
                    "Axis must be 'x', 'y', or 'z'. Alternatively, 0, 1, 2 can be used instead."
                )
            self._rot_order.append(_axis_map[ax])

        if len(angles) != len(self._rot_order):
            raise ValueError(
                f"The number of rotation angles ({len(angles)}) must the "
                f"same as that of rotations axes ({len(self._rot_order)})."
            )

        self._rot_matrix = self._compute_matrix()

    def _compute_matrix(self):
        """
        Compute the rotation matrix.
        """
        mat = np.eye(3)

        for angle, axis in zip(self._angles, self._rot_order):
            mat = np.dot(_make_rot_matrix(angle, axis), mat)

        return mat

    def rotate(self, points):
        """
        Rotate a set of points.

        Parameters
        ----------
        points : array_like (3, ...)
            The points to rotate. The first dimension must be 3.

        Returns
        -------
        rotated_points : np.ndarray
            The rotated points.
        """
        return np.dot(self._rot_matrix, points)

    def rotate_transpose(self, points):
        """
        Transposed rotations of a set of points.

        Parameters
        ----------
        points : array_like (3, ...)
            The points to rotate. The last dimension must be 3.

        Returns
        -------
        rotated_points : np.ndarray
            The rotated points.
        """
        return np.dot(self._rot_matrix.T, points)
