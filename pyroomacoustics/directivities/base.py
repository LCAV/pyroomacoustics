# Directivity module that provides routines to use analytic and mesured directional
# responses for sources and microphones.
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
The `Directivity` class is an abstract class that can be subclassed to create new
types of directivities. The class provides a common interface to access the
directivity patterns.

The class should implement the following methods:

- ``get_response_cartesian`` to get the response for a given direction..
- ``get_response`` to get the response for a given (colatitude, azimuth) pair.
- ``sample_rays`` returns rays and corresponding energy for the directivity.
- ``is_impulse_response`` to indicate whether the directivity is an impulse response or
  just band coefficients.
- ``filter_len_ir`` to return the length of the impulse response. This should return 1
  if the directivity is not an impulse response.
"""

import abc

from ..doa.utils import cart2spher, spher2cart


class Directivity(abc.ABC):
    """
    Abstract class for directivity patterns.

    Directivity can be of three different types in the way it treats frequency.
    Depending on this type, the shape returned by ``get_response`` and
    ``get_response_cartesian`` differ.

    1. Frequency flat: the response is the same at all frequencies. In this case,
       ``is_impulse_response == False`` and the returned shape is ``(n_directions,)``.
    2. Octave bands: the response is provided per octave bands. Then
       ``is_impulse_response == False`` and ``shape == (n_directions, n_bands)``.
    3. Impulse response: there is a rich frequency response provided as an
       impulse response. Then, ``is_impulse_response == True`` and ``shape
       == (n_directions, n_taps)``.

    For ray tracing, the ray energies are tracked only in octave bands. I.e.,
    `sample_rays` returns an energy array with ``shape == (n_rays, n_bands)``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Check if the get_response methods in the NEW class are the same objects
        # as the ones defined here in Directivity.
        overrides_cart = (
            cls.get_response_cartesian is not Directivity.get_response_cartesian
        )
        overrides_sph = cls.get_response_cartesian is not Directivity.get_response

        if not (overrides_cart or overrides_sph):
            raise TypeError(
                f"Can't define {cls.__name__}: you must override "
                "either 'get_response_cartesian' or 'get_response'."
            )

    @property
    @abc.abstractmethod
    def is_impulse_response(self):
        """
        Indicates whether the array contains coefficients for octave bands
        (returns ``False``) or is a full-size impulse response (returns
        ``True``).
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def filter_len_ir(self):
        """
        When ``is_impulse_response`` returns ``True``, this property returns the
        lengths of the impulse responses returned.
        All impulse responses are assumed to have the same length.
        """
        raise NotImplementedError

    def get_response_cartesian(self, directions, magnitude=False):
        """
        Get response for provided direction cartesian vectors.

        Parameters
        ----------
        directions: np.ndarray, (n_points, 3)
            The directions of the desired responses expressed as Cartesian
            unit vectors stacked in the rows of a matrix.
        magnitude: bool
            Ignored

        Returns
        -------
        resp : :py:class:`~numpy.ndarray`
            Response at provided directions. See the class docstring for the shape.
        """
        azimuth, colatitude, _ = cart2spher(directions.T)
        return self.get_response(
            azimuth, colatitude=colatitude, magnitude=magnitude, degrees=False
        )

    def get_response(self, azimuth, colatitude=None, magnitude=False, degrees=True):
        """
        Get response for provided ``(azimuth, colatitude)`` pairs.

        Parameters
        ----------
        azimuth : array_like, shape (n_directions,)
            Azimuth angles to compute the response at.
        colatitude : array_like, optional, shape (n_directions,)
            Corresponding colatitude angles. If ``None``, the default is
            ``colatitude = np.pi / 2.0``, i.e., all directions are on the xy-plane.
        magnitude : bool, optional
            Whether to return magnitude of response.
        degrees : bool, optional
            If ``True`` (default), ``azimuth`` and ``colatitude`` are
            interpreted as in degrees. Otherwise, they are in radians.

        Returns
        -------
        resp : :py:class:`~numpy.ndarray`
            Response at requested directions. See the class docstring for the shape.
        """
        cart = spher2cart(azimuth, colatitude, r=1, degrees=degrees)
        return self.get_response_cartesian(cart, magnitude=magnitude)

    @abc.abstractmethod
    def sample_rays(self, n_rays, rng=None):
        """
        This method samples unit vectors from the sphere according to
        the distribution of the source

        Parameters
        ----------
        n_rays: int
            The number of rays to sample
        rng: numpy.random.Generator or None, optional
            A random number generator object from numpy or None.
            If None is passed numpy.random.default_rng is used to create
            a Generator object.

        Returns
        -------
        ray_directions: numpy.ndarray, shape (n_rays, n_dim)
            An array containing the unit vectors in its columns
        energies: numpy.ndarray, shape (n_rays, n_bands)
            An energy carried per ray so that the expectation over all the rays
            is the energy of the band, i.e., np.mean(energies) == band energy.
        """
        raise NotImplementedError
