import pyroomacoustics as pra
import numpy as np

# tolerance for test success (1%)
tol = 0.01

fdl = 81
fs = 16000

t0 = (2 * fdl + 0.1) / fs
t1 = (3 * fdl - 0.1) / fs
t2 = (4 * fdl + 0.45) / fs
t3 = (5 * fdl + 0.001) / fs
t4 = (6 * fdl + 0.999) / fs

times = np.array(
        [
            [ t0 , t1 + (1 / 40 / 16000), t2, ],
            [ t0, t1 + (10 / fs), 3 * t3, ],
            [ t0, t3, t4, ],
            ],
        )
alphas = np.array(
        [
            [ 1., 0.5, -0.1 ],
            [ 0.5, 0.3, 0.1 ],
            [ 0.3, 2., 0.1 ],
            ],
        )
visibilities = np.array(
        [
            [ 1, 1, 1,],
            [ 1, 1, 1,],
            [ 0, 1, 1,],
            ],
        dtype=np.int32,
        )


def build_rir(time, alpha, visibility, fs, fdl):

    # fractional delay length
    fdl = pra.constants.get('frac_delay_length')
    fdl2 = (fdl-1) // 2

    # the number of samples needed
    N = int(np.ceil(time.max() * fs) + fdl)

    ir_ref = np.zeros(N)
    ir_cython = np.zeros(N)

    try:
        from pyroomacoustics import build_rir
        # Try to use the Cython extension
        build_rir.fast_rir_builder(ir_cython, time, alpha, visibility, fs, fdl)

    except ImportError:
        return 0., 0.

    # fallback to pure Python implemenation
    for i in range(time.shape[0]):
        if visibility[i] == 1:
            time_ip = int(np.round(fs * time[i]))
            time_fp = (fs * time[i]) - time_ip
            ir_ref[time_ip-fdl2:time_ip+fdl2+1] += alpha[i] * pra.fractional_delay(time_fp)

    return ir_ref, ir_cython

def test_build_rir():

    for t, a, v in zip(times, alphas, visibilities):
        ir_ref, ir_cython = build_rir(times[0], alphas[0], visibilities[0], fs, fdl)
        assert np.max(np.abs(ir_ref - ir_cython)) < tol
    

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    for t, a, v in zip(times, alphas, visibilities):
        ir_ref, ir_cython = build_rir(times[0], alphas[0], visibilities[0], fs, fdl)

        print('Error:', np.max(np.abs(ir_ref - ir_cython)))

        plt.figure()
        plt.plot(ir_ref, label='ref')
        plt.plot(ir_cython, label='cython')
        plt.legend()

    plt.show()

