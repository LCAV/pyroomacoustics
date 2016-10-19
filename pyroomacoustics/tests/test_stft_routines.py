# @version: 1.0  date: 19/10/2016 by Robin Scheibler
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

from unittest import TestCase

import numpy as np
from scipy.signal import fftconvolve
import pyroomacoustics as pra

class TestSTFT(TestCase):

    def test_stft_nowindow(self):
        frames = 100
        fftsize = [128, 256, 512]
        hop_div = [1, 2]
        loops = 10

        for n in fftsize:
            for div in hop_div:
                for epoch in range(loops):
                    x = np.random.randn(frames * n // div + n - n // div)
                    X = pra.stft(x, n, n // div, transform=np.fft.rfft)
                    y = pra.istft(X, n, n // div, transform=np.fft.irfft)

                    # because of overlap, there is a scaling at reconstruction
                    y[n // div : -n // div] /= div
                    self.assertTrue(np.allclose(x, y))

    def test_overlap_add(self):

        size1 = [53, 1000, 10000]
        size2 = [1000, 53, 89]
        block = [53, 100, 200]

        for n1, n2, L in zip(size1, size2, block):

            x = np.random.randn(n1)
            y = np.random.randn(n2)

            self.assertTrue(np.allclose(pra.overlap_add(x, y, L), fftconvolve(x, y)))
