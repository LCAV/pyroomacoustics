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

        absorption = 0.07
        round_trip = 4 * np.sqrt(2)
        energy_thresh = 1e-7
        detector_radius = 0.15
        hist_bin_size = 0.004  # resolution of histogram [s]

        histogram_gt = SimpleHistogram(hist_bin_size * pra.constants.get("c"))

        # Create the groundtruth list of energy and travel time
        initial_energy = 2.  # defined in libroom.Room.get_rir_entries
        transmitted = 1.0 * (1. - absorption) ** 2 * initial_energy
        distance = round_trip / 2.0

        while transmitted / distance > energy_thresh:
            r_sq = distance ** 2
            p_hit = 1. - np.sqrt(1. - detector_radius ** 2 / r_sq)
            histogram_gt.add(distance, transmitted / (r_sq * p_hit))
            transmitted *= (1. - absorption) ** 4     # 4 wall hits
            distance += round_trip

        print("Creating the python room")
        room = pra.ShoeBox([2, 2, 2], fs=16000, materials=pra.Material.make_freq_flat(absorption))
        # room = pra.Room(walls, fs=16000)
        room.add_source([0.5, 0.5, 1])
        room.add_microphone_array(pra.MicrophoneArray(np.c_[[1.5, 1.5, 1.0]], room.fs))

        print("Creating the cpp room")
        room.room_engine.set_params(
            room.c,
            0,
            energy_thresh,  # energy threshold for rays
            5.0,  # time threshold for rays
            detector_radius,  # detector radius
            hist_bin_size,  # resolution of histogram [s]
            False,  # is it hybrid model ?
        )

        print("Running ray tracing")
        room.room_engine.ray_tracing(
            np.c_[[-np.pi / 4.0, np.pi / 2.0]], room.sources[0].position  # source loc
        )

        h = room.room_engine.microphones[0].histograms[0].get_hist()
        histogram_rt = np.array(h[0])[:len(histogram_gt)]

        self.assertTrue(np.allclose(histogram_rt, histogram_gt))


if __name__ == "__main__":
    unittest.main()
