from __future__ import division, print_function
from unittest import TestCase
import numpy as np
import pyroomacoustics as pra

tol = -80  # dB
nfft = 128
D = 7
x = np.random.randn(nfft, D).astype('float32')
X_numpy = np.fft.rfft(x, axis=0).astype('complex64')
analysis_window = pra.hann(nfft)
synthesis_window = pra.hann(nfft)

eps = np.finfo(x.dtype).eps

try:
    import pyfftw
    pyfftw_available = True
except ImportError:
    pyfftw_available = False

def no_window(nfft, D, transform, axis=0):

    if D == 1:
        x_local = x[:,0]
        X_local = X_numpy[:,0]
    else:
        if axis == 0:
            x_local = x
            X_local = X_numpy
        else:
            x_local = x.T
            X_local = X_numpy.T

    # make object
    dft = pra.transform.DFT(nfft, D, transform=transform, axis=axis)

    # forward
    X = dft.analysis(x_local)
    err_fwd = pra.dB(np.linalg.norm(X_local-X) + eps)

    # backward
    x_r = dft.synthesis()
    err_bwd = pra.dB(np.linalg.norm(x_local-x_r) + eps)

    return err_fwd, err_bwd


def window(nfft, D, analysis_window, synthesis_window, axis=0):

    if D == 1:
        x_local = x[:,0]
        X_local = X_numpy[:,0]
    else:
        if axis == 0:
            x_local = x
            X_local = X_numpy
        else:
            x_local = x.T
            X_local = X_numpy.T

    # make object
    dft = pra.transform.DFT(nfft, D, axis=axis, analysis_window=analysis_window,
                            synthesis_window=synthesis_window)

    try:
        # forward
        X = dft.analysis(x_local)
        # backward
        x_r = dft.synthesis()
        return True
    except:
        return False


class TestDFT(TestCase):

    def test_no_window_mono(self):
        res = no_window(nfft, D=1, transform='numpy')
        self.assertTrue(res[0] < tol)
        self.assertTrue(res[1] < tol)
        if pyfftw_available:
            res = no_window(nfft, D=1, transform='fftw')
            self.assertTrue(res[0] < tol)
            self.assertTrue(res[1] < tol)
        res = no_window(nfft, D=1, transform='mkl')
        self.assertTrue(res[0] < tol)
        self.assertTrue(res[1] < tol)

    def test_no_window_multichannel_axis0(self):
        res = no_window(nfft, D=D, transform='numpy')
        self.assertTrue(res[0] < tol)
        self.assertTrue(res[1] < tol)
        if pyfftw_available:
            res = no_window(nfft, D=D, transform='fftw')
            self.assertTrue(res[0] < tol)
            self.assertTrue(res[1] < tol)
        res = no_window(nfft, D=D, transform='mkl')
        self.assertTrue(res[0] < tol)
        self.assertTrue(res[1] < tol)

    def test_no_window_multichannel_axis1(self):
        res = no_window(nfft, D=D, transform='numpy', axis=1)
        self.assertTrue(res[0] < tol)
        self.assertTrue(res[1] < tol)
        if pyfftw_available:
            res = no_window(nfft, D=D, transform='fftw', axis=1)
            self.assertTrue(res[0] < tol)
            self.assertTrue(res[1] < tol)
        res = no_window(nfft, D=D, transform='mkl', axis=1)
        self.assertTrue(res[0] < tol)
        self.assertTrue(res[1] < tol)

    def test_window(self):
        res = window(nfft, 1, analysis_window, synthesis_window)
        self.assertTrue(res)
        res = window(nfft, D, analysis_window, synthesis_window)
        self.assertTrue(res)
        res = window(nfft, D, analysis_window, synthesis_window, axis=1)
        self.assertTrue(res)


if __name__ == "__main__":

    print()
    print("1D")
    res = no_window(nfft, D=1, transform='numpy')
    print("numpy :", res)
    if pyfftw_available:
        res = no_window(nfft, D=1, transform='fftw')
        print("fftw :", res)
    res = no_window(nfft, D=1, transform='mkl')
    print("mkl :", res)

    print()
    print("2D, axis=0")
    res = no_window(nfft, D=D, transform='numpy')
    print("numpy :", res)
    if pyfftw_available:
        res = no_window(nfft, D=D, transform='fftw')
        print("fftw :", res)
    res = no_window(nfft, D=D, transform='mkl')
    print("mkl :", res)

    print()
    print("2D, axis=1")
    axis=1
    res = no_window(nfft, D=D, transform='numpy', axis=axis)
    print("numpy :", res)
    if pyfftw_available:
        res = no_window(nfft, D=D, transform='fftw', axis=axis)
        print("fftw :", res)
    res = no_window(nfft, D=D, transform='mkl', axis=axis)
    print("mkl :", res)

    print()
    print("Testing no error with windows...")
    res = window(nfft, 1, analysis_window, synthesis_window)
    print(res)
    res = window(nfft, D, analysis_window, synthesis_window)
    print(res)
    res = window(nfft, D, analysis_window, synthesis_window, axis=1)
    print(res)


