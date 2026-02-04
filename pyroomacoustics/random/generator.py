# Accessor function for a package wide random number generator.
# Copyright (C) 2026  Robin Scheibler
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
Access to a package-wide random number generator (RNG).

The simulation can be made deterministic by fixing a seed.

.. code-block:: python

    # Globally seed pyroomacoustics.
    pra.random.seed(42)

    # Seed the numpy and libroom RNGs separately
    pra.random.seed(numpy=42, libroom=43)

"""

import numpy as np

from .. import libroom

_rng = None


def _set_libroom_seed(seed=None):
    global _rng

    # Just a safeguard in case this is called directly. It shouldn't be, though.
    if _rng is None:
        _rng = np.random.default_rng()

    if seed is None:
        seed = _rng.integers(2**64 - 1, size=(), dtype=np.uint64)

    libroom.set_rng_seed(int(seed))


def get_rng():
    """Access to the package global RNG."""
    global _rng
    if _rng is None:
        seed()
    return _rng


def seed(numpy=None, libroom=None):
    """
    Sets the seeds of the numpy random generator and optionally of the libroom
    sub-package.

    Parameters
    ----------
    numpy: {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        Seed for the Numpy generator. Seed `Numpy doc <https://numpy.org/doc/2.1/reference/random/generator.html>`_
        for details.
    libroom: Unsigned 64 bit int.
        Seed for the libroom sub-package (integer between ``0`` and ``2 ** 64 - 1``).
        If it not provided, it is derived from the Numpy RNG.
    """
    global _rng
    _rng = np.random.default_rng(seed=numpy)
    _set_libroom_seed(seed=libroom)
