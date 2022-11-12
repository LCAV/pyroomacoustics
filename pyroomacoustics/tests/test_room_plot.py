import pyroomacoustics as pra
import matplotlib

matplotlib.use("Agg")


def test_room_2d():
    room = pra.ShoeBox([3, 4], max_order=2)
    room.add_source([1.4, 2.2])
    room.add_microphone([2.3, 3.5])
    room.plot()


def test_room_3d():
    room = pra.ShoeBox([3, 4, 5], max_order=2)
    room.add_source([1.4, 2.2, 4.3])
    room.add_microphone([2.3, 3.5, 2.7])
    room.plot()
