import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra


def inspect_directional_receiver_pattern_anechoic():
    room_dim = [10, 10, 4]
    materials = pra.Material(1.0)  # fully absorbant

    materials = pra.make_materials(
        north=0.0,
        south=1.0,
        east=1.0,
        west=1.0,
        floor=1.0,
        ceiling=1.0,
    )
    max_order = -1  # deactivate ism

    room = pra.ShoeBox(room_dim, materials=materials, max_order=-1)
    room.set_ray_tracing(
        n_rays=100000, receiver_radius=0.1, time_thres=0.1, hist_bin_size=1.0
    )
    room.add_microphone([8, 5, 2])

    # add directional receiver
    grid = pra.doa.GridSphere(n_points=256)
    room.room_engine.microphones[0].set_directions(grid.cartesian.T)

    room.room_engine.ray_tracing(100000, [2, 5, 2])

    bins = []
    for hist in room.room_engine.microphones[0].histograms:
        bins.append(np.sum(hist.get_hist()))

    grid.set_values(bins)
    print(bins)

    grid.plot_old()


def inspect_directional_receiver_pattern_reverb():
    room_dim = [10, 10, 4]
    materials = pra.Material(0.15)  # fully absorbant
    max_order = -1  # deactivate ism

    room = pra.ShoeBox(room_dim, materials=materials, max_order=-1)
    room.set_ray_tracing(
        n_rays=100000, receiver_radius=0.1, time_thres=0.1, hist_bin_size=1.0
    )
    room.add_microphone([8, 5, 2])

    # add directional receiver
    grid = pra.doa.GridSphere(n_points=256)
    room.room_engine.microphones[0].set_directions(grid.cartesian.T)

    room.room_engine.ray_tracing(100000, [2, 5, 2])

    bins = []
    for hist in room.room_engine.microphones[0].histograms:
        bins.append(np.sum(hist.get_hist()))

    grid.set_values(bins)
    print(bins)

    grid.plot_old()


if __name__ == "__main__":

    mic = pra.libroom.Microphone([0.0, 0.0, 0.0], 1, 0.1, 1.0)
    grid = pra.doa.GridSphere(n_points=16)
    dirs = grid.cartesian.T.copy()
    mic.set_directions(dirs)

    test_dir = dirs[0]

    mic.log_histogram(0.5, [1.0], test_dir)

    print(mic.histograms[0].get_hist())
    print(mic.histograms[1].get_hist())

    inspect_directional_receiver_pattern_anechoic()
    inspect_directional_receiver_pattern_reverb()
    plt.show()
