# Test of ray tracing energy
# Copyright (C) 2019  Robin Scheibler, Cyril Cadoux
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
from __future__ import division

import unittest
import numpy as np
import pyroomacoustics as pra


class SimpleHistogram(list):
    """ A Histogram class based on list """

    def __init__(self, bin_size):
        self.bin_size = bin_size
        list.__init__([])

    def add(self, delay, val):
        """
        Adds val to the bin corresponding to delay.
        Pad the histogram to correct size if too short.
        """
        pos = int(delay / self.bin_size)

        if pos >= len(self):
            while len(self) < pos:
                self.append(0)
            self.append(val)
        else:
            self[pos] += val


class TestRayEnergy(unittest.TestCase):
    def test_square_room(self):

        """
        This is a cubic room of 2x2x2 meters. The source is placed at [0.5,0.5, 1]
        and the receiver at [1.5, 1.5, 1]. A ray is launched towards [1, 0, 1] so that
        the first receiver hit is after travel distance of 2*sqrt(2) and each subsequent
        hit travels a further 4*sqrt(2) until the threshold energy is reached.
        """

        energy_absorption = 0.07
        round_trip = 4 * np.sqrt(2)
        energy_thresh = 1e-7
        detector_radius = 0.15
        hist_bin_size = 0.004  # resolution of histogram [s]

        histogram_gt = SimpleHistogram(hist_bin_size * pra.constants.get("c"))

        # Create the groundtruth list of energy and travel time
        initial_energy = 2.0  # defined in libroom.Room.get_rir_entries
        transmitted = 1.0 * (1.0 - energy_absorption) ** 2 * initial_energy
        distance = round_trip / 2.0

        while transmitted / distance > energy_thresh:
            r_sq = distance ** 2
            p_hit = 1.0 - np.sqrt(1.0 - detector_radius ** 2 / r_sq)
            histogram_gt.add(distance, transmitted / (r_sq * p_hit))
            transmitted *= (1.0 - energy_absorption) ** 4  # 4 wall hits
            distance += round_trip

        print("Creating the python room (polyhedral)")
        walls_corners = [
            np.array([[0, 2, 2, 0], [0, 0, 0, 0], [0, 0, 2, 2]]),  # left
            np.array([[0, 0, 2, 2], [2, 2, 2, 2], [0, 2, 2, 0]]),  # right
            np.array([[0, 0, 0, 0], [0, 2, 2, 0], [0, 0, 2, 2]]),  # front`
            np.array([[2, 2, 2, 2], [0, 0, 2, 2], [0, 2, 2, 0]]),  # back
            np.array([[0, 2, 2, 0], [0, 0, 2, 2], [0, 0, 0, 0],]),  # floor
            np.array([[0, 0, 2, 2], [0, 2, 2, 0], [2, 2, 2, 2],]),  # ceiling
        ]
        walls = [pra.wall_factory(c, [energy_absorption], [0.0]) for c in walls_corners]
        room_poly = pra.Room(walls, fs=16000)
        # room = pra.Room(walls, fs=16000)
        room_poly.add_source([0.5, 0.5, 1])
        room_poly.add_microphone_array(
            pra.MicrophoneArray(np.c_[[1.5, 1.5, 1.0]], room_poly.fs)
        )

        room_poly.room_engine.set_params(
            room_poly.c,
            0,
            energy_thresh,  # energy threshold for rays
            5.0,  # time threshold for rays
            detector_radius,  # detector radius
            hist_bin_size,  # resolution of histogram [s]
            False,  # is it hybrid model ?
        )

        print("Running ray tracing (polyhedral)")
        room_poly.room_engine.ray_tracing(
            np.c_[[-np.pi / 4.0, np.pi / 2.0]],
            room_poly.sources[0].position,  # source loc
        )

        print("Creating the python room (shoebox)")
        room_cube = pra.ShoeBox(
            [2, 2, 2], fs=16000, materials=pra.Material(energy_absorption)
        )
        # room = pra.Room(walls, fs=16000)
        room_cube.add_source([0.5, 0.5, 1])
        room_cube.add_microphone_array(
            pra.MicrophoneArray(np.c_[[1.5, 1.5, 1.0]], room_poly.fs)
        )

        room_cube.room_engine.set_params(
            room_poly.c,
            0,
            energy_thresh,  # energy threshold for rays
            5.0,  # time threshold for rays
            detector_radius,  # detector radius
            hist_bin_size,  # resolution of histogram [s]
            False,  # is it hybrid model ?
        )

        print("Running ray tracing (shoebox)")
        room_cube.room_engine.ray_tracing(
            np.c_[[-np.pi / 4.0, np.pi / 2.0]],
            room_cube.sources[0].position,  # source loc
        )

        h_poly = room_poly.room_engine.microphones[0].histograms[0].get_hist()
        h_cube = room_cube.room_engine.microphones[0].histograms[0].get_hist()
        histogram_rt_poly = np.array(h_poly[0])[: len(histogram_gt)]
        histogram_rt_cube = np.array(h_cube[0])[: len(histogram_gt)]

        self.assertTrue(np.allclose(histogram_rt_poly, histogram_gt))
        self.assertTrue(np.allclose(histogram_rt_cube, histogram_gt))


if __name__ == "__main__":
    unittest.main()
