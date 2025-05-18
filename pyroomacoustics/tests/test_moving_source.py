import pyroomacoustics as pra
import numpy as np

def test_simulate_moving_source():
    fs = 16000
    signal = np.arange(10)

    rt60 = 0.5
    room_dim = [6, 3.5, 2.5]
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)

    room.add_microphone(np.array([3, 1, 0.95]), room.fs)

    move = room.simulate_moving_source(
        start_position=[1, 1, 1.47],
        end_position=[5, 1, 1.47],
        signal=signal,
        delay=0,
        fs=fs,
    )

    move = move / np.max(np.abs(move))  # normalize recording
    print(move)


if __name__ == "__main__":
    test_simulate_moving_source()
