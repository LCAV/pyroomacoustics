from __future__ import division, print_function

import os
import time

import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.libroom import threaded_rir_builder

try:
    from pyroomacoustics import build_rir

    build_rir_available = True
except:
    print("build_rir not available")
    build_rir_available = False

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
        [
            t0,
            t1 + (1 / 40 / fs),
            t2,
        ],
        [
            t0,
            t1 + (10 / fs),
            3 * t3,
        ],
        [
            t0,
            t3,
            t4,
        ],
    ],
)
alphas = np.array(
    [
        [1.0, 0.5, -0.1],
        [0.5, 0.3, 0.1],
        [0.3, 2.0, 0.1],
    ],
)
visibilities = np.array(
    [
        [
            1,
            1,
            1,
        ],
        [
            1,
            1,
            1,
        ],
        [
            0,
            1,
            1,
        ],
    ],
    dtype=np.int32,
)


def build_rir_wrap(time, alpha, visibility, fs, fdl):

    # fractional delay length
    fdl = pra.constants.get("frac_delay_length")
    fdl2 = (fdl - 1) // 2

    # the number of samples needed
    N = int(np.ceil(time.max() * fs) + fdl)

    ir_ref = np.zeros(N)
    ir_cython = np.zeros(N)

    ir_cpp_f = ir_cython.astype(np.float32)

    # Try to use the Cython extension
    # build_rir.fast_rir_builder(ir_cython, time, alpha, visibility, fs, fdl)
    threaded_rir_builder(
        ir_cpp_f,
        time.astype(np.float32),
        alpha.astype(np.float32),
        visibility.astype(np.int32),
        fs,
        fdl,
        20,
        2,
    )

    # fallback to pure Python implemenation
    for i in range(time.shape[0]):
        if visibility[i] == 1:
            time_ip = int(np.round(fs * time[i]))
            time_fp = (fs * time[i]) - time_ip
            ir_ref[time_ip - fdl2 : time_ip + fdl2 + 1] += alpha[
                i
            ] * pra.fractional_delay(time_fp)

    return ir_ref, ir_cpp_f


def test_build_rir():

    if not build_rir_available:
        return

    for t, a, v in zip(times, alphas, visibilities):
        ir_ref, ir_cython = build_rir_wrap(
            times[0], alphas[0], visibilities[0], fs, fdl
        )
        assert np.max(np.abs(ir_ref - ir_cython)) < tol


def test_short():
    """Tests that an error is raised if a provided time goes below the zero index"""

    if not build_rir_available:
        return

    N = 100
    fs = 16000
    fdl = 81
    rir = np.zeros(N, dtype=np.float32)

    time = np.array([0.0], dtype=np.float32)
    alpha = np.array([1.0], dtype=np.float32)
    visibility = np.array([1], dtype=np.int32)

    try:
        threaded_rir_builder(
            rir,
            time,
            alpha,
            visibility,
            fs,
            fdl,
            20,
            2,
        )
        assert False, "Short time not caught"
    except RuntimeError:
        print("Ok, short times are caught")


def test_long():
    """Tests that an error is raised if a time falls outside the rir array"""

    if not build_rir_available:
        return

    N = 100
    fs = 16000
    fdl = 81
    rir = np.zeros(N, dtype=np.float32)

    time = np.array([(N - 1) / fs], dtype=np.float32)
    alpha = np.array([1.0], dtype=np.float32)
    visibility = np.array([1], dtype=np.int32)

    try:
        threaded_rir_builder(rir, time, alpha, visibility, fs, fdl, 20, 2)
        assert False
    except RuntimeError:
        print("Ok, long times are caught")


def test_errors():
    """Tests that errors are raised when array lengths differ"""

    if not build_rir_available:
        return

    N = 300
    fs = 16000
    fdl = 81
    rir = np.zeros(N, dtype=np.float32)

    time = np.array([100 / fs, 200 / fs], dtype=np.float32)
    alpha = np.array([1.0, 1.0], dtype=np.float32)
    visibility = np.array([1, 1], dtype=np.int32)

    try:
        threaded_rir_builder(rir, time, alpha[:1], visibility, fs, fdl, 20, 2)
        assert False
    except RuntimeError:
        print("Ok, alpha error occured")
        pass

    try:
        threaded_rir_builder(rir, time, alpha, visibility[:1], fs, fdl, 20, 2)
        assert False
    except RuntimeError:
        print("Ok, visibility error occured")
        pass

    try:
        threaded_rir_builder(rir, time, alpha, visibility, fs, 80, 20, 2)
        assert False
    except RuntimeError:
        print("Ok, fdl error occured")
        pass


def measure_runtime(dtype=np.float32, num_threads=4):

    n_repeat = 20
    n_img = 1000000
    T = 2.0
    fs = 16000
    fdl = 81
    time_arr = T * np.random.rand(n_img).astype(dtype) + (fdl // 2) / fs
    alpha = np.random.randn(n_img).astype(dtype)
    rir_len = int(np.ceil(time_arr.max() * fs) + fdl)
    rir = np.zeros(rir_len, dtype=dtype)
    visibility = np.ones(n_img, dtype=np.int32)

    tick = time.perf_counter()
    rir[:] = 0.0
    for i in range(n_repeat):
        threaded_rir_builder(rir, time_arr, alpha, visibility, fs, fdl, 20, 1)
    tock_1 = (time.perf_counter() - tick) / n_repeat

    tick = time.perf_counter()
    rir[:] = 0.0
    for i in range(n_repeat):
        threaded_rir_builder(rir, time_arr, alpha, visibility, fs, fdl, 20, num_threads)
    tock_8 = (time.perf_counter() - tick) / n_repeat

    if dtype == np.float64:
        tick = time.perf_counter()
        rir[:] = 0.0
        for i in range(n_repeat):
            build_rir.fast_rir_builder(rir, time_arr, alpha, visibility, fs, fdl)
        tock_old = (time.perf_counter() - tick) / n_repeat

    print("runtime:")
    print(f"  - 1 thread : {tock_1}")
    print(f"  - {num_threads} threads: {tock_8}")
    if dtype == np.float64:
        print(f"  - old: {tock_old}")
    print(f"speed-up vs single-thread: {tock_1 / tock_8:.2f}x")
    if dtype == np.float64:
        print(f"speed-up vs old: {tock_old / tock_8:.2f}x")
        print(f"speed-up single-threaded vs old: {tock_old / tock_1:.2f}x")


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    for t, a, v in zip(times, alphas, visibilities):
        ir_ref, ir_cython = build_rir_wrap(t, a, v, fs, fdl)

        print("Error:", np.max(np.abs(ir_ref - ir_cython)))

        plt.figure()
        plt.plot(ir_ref, label="ref")
        plt.plot(ir_cython, label="cython")
        plt.legend()

    test_short()
    test_long()
    test_errors()

    num_threads = os.cpu_count()
    measure_runtime(dtype=np.float32, num_threads=num_threads)
    measure_runtime(dtype=np.float64, num_threads=num_threads)

plt.show()
