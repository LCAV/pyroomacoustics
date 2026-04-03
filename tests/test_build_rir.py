from __future__ import division, print_function

import os
import time

import numpy as np
import pytest

import pyroomacoustics as pra
from pyroomacoustics import libroom

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


def build_rir_wrap(time, alpha, fs, fdl):
    # fractional delay length
    fdl = pra.constants.get("frac_delay_length")
    fdl2 = (fdl - 1) // 2

    # the number of samples needed
    N = int(np.ceil(time.max() * fs) + fdl)

    ir_ref = np.zeros(N)
    ir_cython = np.zeros(N)

    ir_cpp_f = ir_cython.astype(np.float32)

    # Try to use the Cython extension
    # build_rir.fast_rir_builder(ir_cython, time, alpha, fs, fdl)
    libroom.rir_builder(
        ir_cpp_f,
        time.astype(np.float32),
        alpha.astype(np.float32),
        fs,
        fdl,
        20,
        2,
    )

    # fallback to pure Python implemenation
    for i in range(time.shape[0]):
        time_ip = int(np.round(fs * time[i]))
        time_fp = (fs * time[i]) - time_ip
        ir_ref[time_ip - fdl2 : time_ip + fdl2 + 1] += alpha[i] * pra.fractional_delay(
            time_fp
        )

    return ir_ref, ir_cpp_f


def test_build_rir():
    if not build_rir_available:
        return

    for t, a, v in zip(times, alphas, visibilities):
        ir_ref, ir_cython = build_rir_wrap(times[0], alphas[0], fs, fdl)
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

    with pytest.raises(RuntimeError):
        libroom.rir_builder(
            rir,
            time,
            alpha,
            fs,
            fdl,
            20,
            2,
        )


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

    with pytest.raises(RuntimeError):
        libroom.rir_builder(rir, time, alpha, fs, fdl, 20, 2)


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

    with pytest.raises(RuntimeError):
        libroom.rir_builder(rir, time, alpha[:1], fs, fdl, 20, 2)

    with pytest.raises(RuntimeError):
        libroom.rir_builder(rir, time, alpha, fs, 80, 20, 2)


@pytest.mark.parametrize("dtype,tol", [(np.float32, 1e-5), (np.float64, 1e-7)])
def test_delay_sum(dtype, tol):
    n = 1000
    taps = 81
    n_threads = os.cpu_count()
    rir_len = 16000

    np.random.seed(0)
    irs = np.random.randn(n, taps).astype(dtype)
    delays = np.random.randint(0, rir_len - taps, size=n, dtype=np.int32)
    out1 = np.zeros(rir_len, dtype=dtype)
    out2 = np.zeros(rir_len, dtype=dtype)

    libroom.delay_sum(irs, delays, out1, n_threads)

    for i in range(n):
        out2[delays[i] : delays[i] + taps] += irs[i]

    error = abs(out1 - out2).max()
    assert error < tol

    with pytest.raises(RuntimeError):
        libroom.delay_sum(irs[:-1, :], delays, out1, n_threads)

    with pytest.raises(RuntimeError):
        libroom.delay_sum(irs, delays + rir_len, out1, n_threads)

    with pytest.raises(RuntimeError):
        libroom.delay_sum(irs, delays - rir_len, out1, n_threads)


@pytest.mark.parametrize("dtype,tol", [(np.float32, 0.005), (np.float64, 0.005)])
def test_fractional_delay(dtype, tol):
    n = 10000
    lut_size = 20
    n_threads = os.cpu_count()
    fdl = pra.constants.get("frac_delay_length")
    delays = (np.random.rand(n) - 0.5).astype(dtype)
    out1 = pra.fractional_delay(delays)
    out2 = np.zeros((n, fdl), dtype=dtype)
    libroom.fractional_delay(out2, delays, lut_size, n_threads)
    error = abs(out1 - out2).max()
    print(error)
    assert error < tol

    with pytest.raises(RuntimeError):
        libroom.fractional_delay(out2, delays[:-1], lut_size, n_threads)

    with pytest.raises(RuntimeError):
        libroom.fractional_delay(out2[0, :], delays, lut_size, n_threads)


def measure_runtime(dtype=np.float32, num_threads=4):
    n_repeat = 20
    n_img = 1000000
    T = 3.0
    fs = 16000
    fdl = 81
    time_arr = T * np.random.rand(n_img).astype(dtype) + (fdl // 2) / fs
    alpha = np.random.randn(n_img).astype(dtype)
    rir_len = int(np.ceil(time_arr.max() * fs) + fdl)
    rir = np.zeros(rir_len, dtype=dtype)

    tick = time.perf_counter()
    rir[:] = 0.0
    for i in range(n_repeat):
        libroom.rir_builder(rir, time_arr, alpha, fs, fdl, 20, 1)
    tock_1 = (time.perf_counter() - tick) / n_repeat

    tick = time.perf_counter()
    rir[:] = 0.0
    for i in range(n_repeat):
        libroom.rir_builder(rir, time_arr, alpha, fs, fdl, 20, num_threads)
    tock_8 = (time.perf_counter() - tick) / n_repeat

    tick = time.perf_counter()
    rir[:] = 0.0
    for i in range(n_repeat):
        tt = time_arr
        td = np.round(tt).astype(np.int32)
        tf = (tt - td).astype(dtype)
        irs = np.zeros((tt.shape[0], fdl), dtype=dtype)
        libroom.fractional_delay(irs, tf, 20, num_threads)
        irs *= alpha[:, None]
        libroom.delay_sum(irs, td, rir, num_threads)
    tock_2steps = (time.perf_counter() - tick) / n_repeat

    if dtype == np.float64:
        tick = time.perf_counter()
        rir[:] = 0.0
        for i in range(n_repeat):
            build_rir.fast_rir_builder(rir, time_arr, alpha, fs, fdl)
        tock_old = (time.perf_counter() - tick) / n_repeat

    print("runtime:")
    print(f"  - 1 thread : {tock_1}")
    print(f"  - {num_threads} threads: {tock_8}")
    print(f"  - {num_threads} threads, 2-steps: {tock_2steps}")
    if dtype == np.float64:
        print(f"  - old: {tock_old}")
    print(f"speed-up vs single-thread: {tock_1 / tock_8:.2f}x")
    print(f"speed-up 2-steps vs single-thread: {tock_1 / tock_2steps:.2f}x")
    if dtype == np.float64:
        print(f"speed-up vs old: {tock_old / tock_8:.2f}x")
        print(f"speed-up single-threaded vs old: {tock_old / tock_1:.2f}x")
        print(f"speed-up 2-steps vs old: {tock_old / tock_2steps:.2f}x")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_fractional_delay(np.float32, 2e-2)
    test_fractional_delay(np.float64, 2e-2)
    test_delay_sum(np.float32, 1e-4)

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
