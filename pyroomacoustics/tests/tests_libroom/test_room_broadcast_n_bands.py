# Test of Room constructor
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

import numpy as np

import pyroomacoustics as pra

wall_corners = [
    np.array([[0, 3, 3, 0], [0, 0, 0, 0], [0, 0, 2, 2]]),  # left
    np.array([[0, 0, 6, 6], [8, 8, 8, 8], [0, 4, 4, 0]]),  # right
    np.array([[0, 0, 6, 3], [0, 8, 8, 0], [0, 0, 0, 0]]),  # floor
    np.array([[0, 3, 6, 0], [0, 0, 8, 8], [2, 2, 4, 4]]),  # ceiling
    np.array([[0, 0, 0, 0], [0, 0, 8, 8], [0, 2, 4, 0]]),  # back
    np.array([[3, 6, 6, 3], [0, 8, 8, 0], [0, 0, 4, 2]]),  # front
]

absorptions = [0.1, 0.25, 0.25, 0.25, 0.2, 0.15]
scatterings = [0.1, 0.05, 0.1, 0.02, 0.2, 0.3]


def room_construct_and_broadcast_bands():
    walls = [
        pra.wall_factory(c, [a], [s])
        for c, a, s in zip(wall_corners, absorptions, scatterings)
    ]
    obstructing_walls = []

    room = pra.libroom.Room(
        walls,
        obstructing_walls,
        [],
        pra.constants.get("c"),  # speed of sound
        3,  # ism order
        1e-7,  # energy_thres
        1.0,  # time_thres
        0.5,  # receiver_radius
        0.004,  # hist_bin_size
        True,  # a priori we will always use a hybrid model
    )
    room.add_mic([1.5, 2.3, 1.2])

    for wall in room.walls:
        assert wall.absorption.shape == (1,)
    for mic in room.microphones:
        for hist in mic.histograms:
            assert hist.get_hist().shape[0] == 1

    room.broadcast_bands_to(7)

    for wall in room.walls:
        assert wall.absorption.shape == (7,)
    for mic in room.microphones:
        for hist in mic.histograms:
            assert hist.get_hist().shape[0] == 7

    room.broadcast_bands_to(1)

    for wall in room.walls:
        assert wall.absorption.shape == (1,)
    for mic in room.microphones:
        for hist in mic.histograms:
            assert hist.get_hist().shape[0] == 1

    return room


def test_room_broadcast_n_bands():
    room_construct_and_broadcast_bands()


if __name__ == "__main__":
    room = room_construct_and_broadcast_bands()
