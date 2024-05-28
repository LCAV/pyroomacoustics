# Directivity module that provides routines to use analytic and mesured directional
# responses for sources and microphones.
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
Real-world microphones and sound sources usually exhibit directional responses.
That is, the impulse response (or frequency response) depends on the emission
or reception angle (for sources and microphones, respectively).
A concrete example is the human ear attached to the head. The left ear is
typically more sensitive to sounds coming from the left side than from the right.

This sub-module provides an interface to add such directional responses to
microphones and sources in the room impulse response simulation.

.. warning::
    The directional responses are currently only supported for the
    image source method based simulation.

.. warning::
    Directional responses are only supported for 3D rooms.


The directivities are described by an object of a class derived from :py:class:`~pyroomacoustics.directivities.base.Directivity`.

Let's dive right in with an example.
Here, we simulate a shoebox room with a cardioid source and a dummy head
receiver with two ears (i.e., microphones). This simulates a binaural response.

.. code-block:: python

    import pyroomacoustics as pra

    room = pra.ShoeBox(
        p=[5, 3, 3],
        materials=pra.Material(energy_absorption),
        fs=16000,
        max_order=40,
    )

    # add a cardioid source
    dir = pra.directivities.Cardioid(DirectionVector(azimuth=-65, colatitude=90) , gain=1.0)
    room.add_source([3.75, 2.13, 1.41], directivity=dir)

    # add a dummy head receiver from the MIT KEMAR database
    hrtf = MeasuredDirectivityFile(
        path="mit_kemar_normal_pinna.sofa",  # SOFA file is in the database
        fs=room.fs,
        interp_order=12,  # interpolation order
        interp_n_points=1000,  # number of points in the interpolation grid
    )

    # provide the head rotation
    orientation = Rotation3D([90.0, 30.0], "yz", degrees=True)

    # choose and interpolate the directivities
    dir_left = hrtf.get_mic_directivity("left", orientation=orientation)
    dir_right = hrtf.get_mic_directivity("right", orientation=orientation)

    # for a head-related transfer function, the microphone should be co-located
    mic_pos = [1.05, 1.74, 1.81]
    room.add_microphone(mic_pos, directivity=dir)
    room.add_microphone(mic_pos, directivity=dir)
"""
from .analytic import (
    Cardioid,
    CardioidFamily,
    FigureEight,
    HyperCardioid,
    Omnidirectional,
    SubCardioid,
    cardioid_func,
)
from .base import Directivity
from .direction import DirectionVector, Rotation3D
from .measured import MeasuredDirectivity, MeasuredDirectivityFile
