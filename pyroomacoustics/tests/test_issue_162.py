import numpy as np
import pyroomacoustics as pra


def compute_rir(order):
    fromPos = np.zeros((3))
    toPos = np.ones((3, 1))
    roomSize = np.array([3, 3, 3])
    e_abs = 1.0 - (1.0 - 0.95) ** 2
    room = pra.ShoeBox(
        roomSize, fs=1000, materials=pra.Material(e_abs), max_order=order
    )
    room.add_source(fromPos)
    mics = pra.MicrophoneArray(toPos, room.fs)
    room.add_microphone_array(mics)
    room.compute_rir()


def test_issue_162_max_order_15():
    compute_rir(15)


def test_issue_162_max_order_31():
    compute_rir(31)


def test_issue_162_max_order_32():
    compute_rir(32)


def test_issue_162_max_order_50():
    compute_rir(50)


def test_issue_162_max_order_75():
    compute_rir(75)
