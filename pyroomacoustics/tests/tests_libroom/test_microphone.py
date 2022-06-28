import pyroomacoustics as pra

if __name__ == "__main__":

    mic = pra.libroom.Microphone([0.0, 0.0, 0.0], 1, 0.1, 1.0)
    grid = pra.doa.GridSphere(n_points=16, enable_peak_finding=False)
    dirs = grid.cartesian.T.copy()
    mic.set_directions(dirs)

    test_dir = dirs[0]

    mic.log_histogram(0.5, [1.0], test_dir)

    print(mic.histograms[0].get_hist())
    print(mic.histograms[1].get_hist())
