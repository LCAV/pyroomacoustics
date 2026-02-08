# Copyright (C) 2026  Erik Fleischhauer
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
This class can be used to set a Real Spherical Harmonics directivities.

Real Spherical Harmonics are uses in the `Ambisonic file format 
<https://en.wikipedia.org/wiki/Ambisonic_data_exchange_formats>`_,
which can be uses to represent and restore a sound field at a given point.

Here is a simple example of capturing room impulse response using real spherical harmonics directivities:

.. code-block:: python

    # Simple example recording the different Harmonics as different microphones.
    order = 2
    azimuth = np.deg2rad(50)

    room = pra.AnechoicRoom(fs=16000)
    room.add_source([np.cos(azimuth), np.sin(azimuth), 1.0])

    for m, n in zip(*get_mn_in_acn_order(order)):
        room.add_microphone(
            [0.0, 0.0, 1.0],
            directivity=pra.directivities.RealSphericalHarmonicsDirectivity(
                m, n),
        )

    room.compute_rir()

"""

import numpy as np
import scipy.special

from .base import Directivity


def get_mn_in_acn_order(degree):
    """
    Calculates the order-degree (m,n) pairs in the `Ambisonic Channel Number (ACN) 
    <https://en.wikipedia.org/wiki/Ambisonic_data_exchange_formats>`_ format up to a given degree.

    Parameters
    ----------
    degree : int
        Maximum degree of the spherical harmonics.

    Returns
    -------
    all_m : ndarray
        Array of orders m in ACN order.
    all_n : ndarray
        Array of degrees n in ACN order.
    """
    all_m = np.array([j - i for i in range(0, degree + 1) for j in range(0, 2 * i + 1)])
    all_n = np.array([i for i in range(0, degree + 1) for _ in range(0, 2 * i + 1)])
    return all_m, all_n


def real_sph_harm(n, m, theta, phi, condon_shortley_phase=False):
    """
    Calculates the real spherical harmonics.

    Parameters
    ----------
    n : int
        Degree of the spherical harmonic.
    m : int
        Order of the spherical harmonic.
    theta : array_like
        Polar (colatitudinal) coordinate in radians.
    phi : array_like
        Azimuthal coordinate in radians.
    condon_shortley_phase : bool, optional
        If True, includes the Condon-Shortley phase factor (-1)^m. Default is False.

    Returns
    -------
    y_real : ndarray
        Real spherical harmonics evaluated at the given angles.
    """
    n = np.asarray(n)
    m = np.asarray(m)

    assert np.all(n >= 0), "Degree n must be non-negative."
    assert np.all(np.abs(m) <= n), "Order m must satisfy |m| <= n."

    try:
        ysh_complx = scipy.special.sph_harm_y(n, m, theta, phi)
    except AttributeError:
        # Deprecated since scipy v1.15.0.
        ysh_complx = scipy.special.sph_harm(m, n, phi, theta)

    y_real = np.empty_like(ysh_complx, dtype=np.float64)
    y_real[(m >= 0)] = np.real(ysh_complx[(m >= 0)])
    y_real[(m < 0)] = np.imag(ysh_complx[(m < 0)])

    if not condon_shortley_phase:
        # Cancel Condon-Shortley Phase (term (-1) ** m) by multiplying with (-1) ** m
        # In Rafaely's book, this step is not done
        mask = m != 0
        if np.any(mask):
            y_real[mask] *= np.sqrt(2) * np.array([-1.0]) ** m[mask]
        # In the formula, for m<0, |m| is used. We consider this by the use of the constraint.
        y_real[np.logical_and(m < 0, (m % 2) == 0)] = -y_real[
            np.logical_and(m < 0, (m % 2) == 0)
        ]

    return y_real


class RealSphericalHarmonicsDirectivity(Directivity):
    """
    A class for real spherical harmonic directivity patterns.

    Parameters
    ----------
    m: int
        Order of the spherical harmonic.
    n: int
        Degree of the spherical harmonic.
    condon_shortley_phase: bool, optional
        If True, includes the Condon-Shortley phase factor (-1)^m. Default is False.
    """

    def __init__(self, m, n, condon_shortley_phase: bool = False):
        self.m = m
        self.n = n

        self.condon_shortley_phase = condon_shortley_phase

    @property
    def is_impulse_response(self):
        return True

    @property
    def filter_len_ir(self):
        return 1

    def get_response(
        self, azimuth, colatitude=None, magnitude=False, frequency=None, degrees=True
    ):

        if degrees:
            azimuth = np.radians(azimuth)
            colatitude = np.radians(colatitude)
        return real_sph_harm(
            self.n,
            self.m,
            colatitude,
            azimuth,
            condon_shortley_phase=self.condon_shortley_phase,
        )[:, np.newaxis]
