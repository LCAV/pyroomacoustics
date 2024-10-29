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

- ``get_response`` to get the response for a given angle and frequency
- ``is_impulse_response`` to indicate whether the directivity is an impulse response or
  just band coefficients
- ``filter_len_ir`` to return the length of the impulse response. This should return 1
  if the directivity is not an impulse response.
"""
import abc


class Directivity(abc.ABC):
    """
    Abstract class for directivity patterns.
    """

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

    @abc.abstractmethod
    def get_response(
        self, azimuth, colatitude=None, magnitude=False, frequency=None, degrees=True
    ):
        """
        Get response for provided angles.

        Parameters
        ----------
        azimuth : array_like
            Azimuth
        colatitude : array_like, optional
            Colatitude in degrees. Default is to be on XY plane.
        magnitude : bool, optional
            Whether to return magnitude of response.
        frequency : float, optional
            For which frequency to compute the response.
            If the response is frequency independent, this parameter is ignored.
        degrees : bool, optional
            If ``True``, ``azimuth`` and ``colatitude`` are in degrees.
            Otherwise, they are in radians.


        Returns
        -------
        resp : :py:class:`~numpy.ndarray`
            Response at provided angles.
        """
        raise NotImplementedError
