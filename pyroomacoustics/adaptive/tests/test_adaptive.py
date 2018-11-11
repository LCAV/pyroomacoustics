
from __future__ import division, print_function

from unittest import TestCase
import numpy as np
from scipy.signal import fftconvolve
import pyroomacoustics as pra

# fix RNG for a deterministic result
np.random.seed(0)

# parameters
length = 15        # the unknown filter length
n_samples = 2000   # the number of samples to run
tol = 1e-2         # tolerance reconstructed filter

# the unknown filter (unit norm)
w = np.random.randn(length)
w /= np.linalg.norm(w)

# create a known driving signal
x = np.random.randn(n_samples)

# convolve with the unknown filter
d_clean = fftconvolve(x, w)[:n_samples]

# a function to the adaptive filter on all the samples
def run_filter(algorithm, x, d):
    for i in range(x.shape[0]):
        algorithm.update(x[i], d[i])

class TestAdaptiveFilter(TestCase):

    def test_rls(self):
        rls = pra.adaptive.RLS(length, lmbd=1., delta=2.0)
        run_filter(rls, x, d_clean)
        error = np.linalg.norm(rls.w - w)
        print('RLS Reconstruction Error', error)
        self.assertTrue(error < tol)

    def test_block_rls(self):
        block_rls = pra.adaptive.BlockRLS(length, lmbd=1., delta=2.0)
        run_filter(block_rls, x, d_clean)
        error = np.linalg.norm(block_rls.w - w)
        print('Block RLS Reconstruction Error', error)
        self.assertTrue(error < tol)

    def test_nlms(self):
        nlms = pra.adaptive.NLMS(length, mu=0.5)
        run_filter(nlms, x, d_clean)
        error = np.linalg.norm(nlms.w - w)
        print('NLMS Reconstruction Error', error)
        self.assertTrue(error < tol)

    def test_block_lms(self):
        block_lms = pra.adaptive.BlockLMS(length, mu=1./length/2.)
        run_filter(block_lms, x, d_clean)
        error = np.linalg.norm(block_lms.w - w)
        print('Block LMS Reconstruction Error', error)
        self.assertTrue(error < tol)

