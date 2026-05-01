import unittest

import matplotlib

matplotlib.use("Agg")

from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import art3d

import pyroomacoustics as pra


def _has_wall_collection(ax):
    return any(
        isinstance(c, (PatchCollection, art3d.Poly3DCollection)) for c in ax.collections
    )


class TestRoomPlot(unittest.TestCase):
    def test_room_2d(self):
        room = pra.ShoeBox([3, 4], max_order=2)
        room.add_source([1.4, 2.2])
        room.add_microphone([2.3, 3.5])
        room.plot()

    def test_room_3d(self):
        room = pra.ShoeBox([3, 4, 5], max_order=2)
        room.add_source([1.4, 2.2, 4.3])
        room.add_microphone([2.3, 3.5, 2.7])
        room.plot()

    def test_plot_walls_2d(self):
        room = pra.ShoeBox([3, 4], max_order=2)
        room.add_source([1.4, 2.2])
        room.add_microphone([2.3, 3.5])

        fig, ax = room.plot()
        self.assertTrue(_has_wall_collection(ax))

        fig, ax = room.plot(plot_walls=False)
        self.assertFalse(_has_wall_collection(ax))

    def test_plot_walls_3d(self):
        room = pra.ShoeBox([3, 4, 5], max_order=2)
        room.add_source([1.4, 2.2, 4.3])
        room.add_microphone([2.3, 3.5, 2.7])

        fig, ax = room.plot()
        self.assertTrue(_has_wall_collection(ax))

        fig, ax = room.plot(plot_walls=False)
        self.assertFalse(_has_wall_collection(ax))

    def test_anechoic_room_no_walls_2d(self):
        room = pra.AnechoicRoom(dim=2)
        room.add_source([1.4, 2.2])
        room.add_microphone([2.3, 3.5])

        fig, ax = room.plot()
        self.assertFalse(_has_wall_collection(ax))

    def test_anechoic_room_no_walls_3d(self):
        room = pra.AnechoicRoom(dim=3)
        room.add_source([1.4, 2.2, 4.3])
        room.add_microphone([2.3, 3.5, 2.7])

        fig, ax = room.plot()
        self.assertFalse(_has_wall_collection(ax))
