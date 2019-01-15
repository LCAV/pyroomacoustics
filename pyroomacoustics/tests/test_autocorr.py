import numpy as np
from pyroomacoustics import autocorr
from unittest import TestCase
import time

N = 256
n_iter = 100
x = np.random.randn(N)
tol = 1e-12


def consistent_results(p, biased=True):

    r_time = autocorr(x, p, method='time', biased=biased)
    r_fft = autocorr(x, p, method='fft', biased=biased)
    r_np = autocorr(x, p, method='numpy', biased=biased)
    r_pra = autocorr(x, p, method='pra', biased=biased)

    err_fft = np.linalg.norm(r_time - r_fft)
    err_np = np.linalg.norm(r_time - r_np)
    err_pra = np.linalg.norm(r_time - r_pra)
    return err_fft, err_np, err_pra


class TestAutoCorr(TestCase):

    def test_consistent_low_p_biased(self):
        res = consistent_results(p=3)
        self.assertTrue(res[0] < tol)
        self.assertTrue(res[1] < tol)
        self.assertTrue(res[2] < tol)

    def test_consistent_high_p_biased(self):
        res = consistent_results(p=35)
        self.assertTrue(res[0] < tol)
        self.assertTrue(res[1] < tol)
        self.assertTrue(res[2] < tol)

    def test_consistent_low_p_unbiased(self):
        res = consistent_results(p=3, biased=False)
        self.assertTrue(res[0] < tol)
        self.assertTrue(res[1] < tol)
        self.assertTrue(res[2] < tol)

    def test_consistent_high_p_unbiased(self):
        res = consistent_results(p=35, biased=False)
        self.assertTrue(res[0] < tol)
        self.assertTrue(res[1] < tol)
        self.assertTrue(res[2] < tol)


def timing_and_error(p):
    print()
    print("-" * 40)
    print("p = {}".format(p))
    print("-" * 40)

    print("TIMING:")

    start_time = time.time()
    for n in range(n_iter):
        autocorr(x, p, method="time")
    proc_time = (time.time() - start_time) / n_iter * 1e6
    print("time   : {} us".format(proc_time))

    start_time = time.time()
    for n in range(n_iter):
        autocorr(x, p, method="fft")
    proc_time = (time.time() - start_time) / n_iter * 1e6
    print("fft    : {} us".format(proc_time))

    start_time = time.time()
    for n in range(n_iter):
        autocorr(x, p, method="numpy")
    proc_time = (time.time() - start_time) / n_iter * 1e6
    print("numpy  : {} us".format(proc_time))

    start_time = time.time()
    for n in range(n_iter):
        autocorr(x, p, method="pra")
    proc_time = (time.time() - start_time) / n_iter * 1e6
    print("pra    : {} us".format(proc_time))

    print()
    print("BIASED")

    res = consistent_results(p, biased=True)
    print("fft   error wrt time : {}".format(res[0]))
    print("numpy error wrt time : {}".format(res[1]))
    print("pra   error wrt time : {}".format(res[2]))

    print()
    print("UNBIASED")

    res = consistent_results(p, biased=False)
    print("fft   error wrt time : {}".format(res[0]))
    print("numpy error wrt time : {}".format(res[1]))
    print("pra   error wrt time : {}".format(res[2]))


if __name__ == '__main__':
    timing_and_error(p=3)
    timing_and_error(p=35)




