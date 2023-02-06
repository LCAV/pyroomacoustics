# Shoebox is a sub-class of rooms that are rectangular
# Copyright (C) 2022-2014  Frederike Duembgen
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
import warnings

import numpy as np

from ..parameters import Material
from .shoebox import ShoeBox


class AnechoicRoom(ShoeBox):
    """
    This class provides an API for creating an Anechoic "room" in 2D or 3D.

    Parameters
    ----------
    dim: int
        Dimension of the room (2 or 3).
    fs: int, optional
        The sampling frequency in Hz. Default is 8000.
    t0: float, optional
        The global starting time of the simulation in seconds. Default is 0.
    sigma2_awgn: float, optional
        The variance of the additive white Gaussian noise added during
        simulation. By default, none is added.
    sources: list of SoundSource objects, optional
        Sources to place in the room. Sources can be added after room creating
        with the `add_source` method by providing coordinates.
    mics: MicrophoneArray object, optional
        The microphone array to place in the room. A single microphone or
        microphone array can be added after room creation with the
        `add_microphone_array` method.
    temperature: float, optional
        The air temperature in the room in degree Celsius. By default, set so
        that speed of sound is 343 m/s.
    humidity: float, optional
        The relative humidity of the air in the room (between 0 and 100). By
        default set to 0.
    air_absorption: bool, optional
        If set to True, absorption of sound energy by the air will be
        simulated.
    """

    def __init__(
        self,
        dim=3,
        fs=8000,
        t0=0.0,
        sigma2_awgn=None,
        sources=None,
        mics=None,
        temperature=None,
        humidity=None,
        air_absorption=False,
    ):
        if not dim in [2, 3]:
            raise ValueError("Anechoic room dimension has to be either 2 or 3.")

        # Setting max_order to 0 emulates an anechoic room.
        max_order = 0

        # Ray tracing only makes sense in echoic rooms.
        ray_tracing = False

        # Create some dummy walls
        p = np.ones((dim,))

        # The materials are not actually used because max_order is set to 0 and ray-tracing to False.
        # Anyways, we use the energy_absorption and scattering corresponding to an anechoic room.
        materials = Material(energy_absorption=1.0, scattering=0.0)

        # Set deprecated parameter
        absorption = None

        ShoeBox.__init__(
            self,
            p=p,
            fs=fs,
            t0=t0,
            max_order=max_order,
            sigma2_awgn=sigma2_awgn,
            sources=sources,
            mics=mics,
            materials=materials,
            temperature=temperature,
            humidity=humidity,
            air_absorption=air_absorption,
            ray_tracing=ray_tracing,
        )

    def __str__(self):
        return "AnechoicRoom instance in {}D.".format(self.dim)

    def is_inside(self, p):
        """Overloaded function to eliminate testing if objects are inside "room"."""
        # always return True because we want the walls to have no effect.
        return True

    def get_bbox(self):
        """Returns a bounding box for the mics and sources, for plotting."""

        if (self.mic_array is None) and not self.sources:
            raise ValueError("Nothing to plot, the Anechoic Room is empty!")

        lower = np.inf * np.ones((self.dim,))
        upper = -np.inf * np.ones((self.dim,))

        if self.mic_array is not None:
            lower = np.min(np.r_[lower[None, :], self.mic_array.R], axis=0)
            upper = np.max(np.r_[upper[None, :], self.mic_array.R], axis=0)

        for i, source in enumerate(self.sources):
            lower = np.min(np.r_[lower[None, :], source.position[None, :]], axis=0)
            upper = np.max(np.c_[upper[None, :], source.position[None, :]], axis=0)

        return np.c_[lower, upper]

    def plot_walls(self, ax):
        """Overloaded function to eliminate wall plotting."""
        return 1

    def plot(self, **kwargs):
        """Overloaded function to issue warning when img_order is given."""
        if "img_order" in kwargs.keys():
            warnings.warn("Ignoring img_order argument for AnechoicRoom.", UserWarning)
        ShoeBox.plot(self, **kwargs)
