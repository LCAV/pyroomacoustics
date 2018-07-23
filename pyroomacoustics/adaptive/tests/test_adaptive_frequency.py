from __future__ import division, print_function

from unittest import TestCase
import numpy as np
from scipy.signal import fftconvolve
import pyroomacoustics as pra

# fix RNG for a deterministic result
np.random.seed(0)

# parameters
num_taps = 6        # the unknown filter length
n_samples = 20000   # the number of samples to run
tol = 1e-5         # tolerance reconstructed filter
fft_length = 128   # block size

# the unknown filters in the frequency domain
num_bands = fft_length//2+1
W = np.random.randn(num_taps,num_bands) + \
    1j*np.random.randn(num_taps,num_bands)
W /= np.linalg.norm(W, axis=0)

# create a known driving signal
x = np.random.randn(n_samples)

# take to STFT domain
window = pra.hann(fft_length)  # the analysis window
hop = fft_length//2
stft_in = pra.transform.STFT(fft_length, hop=hop,
                             analysis_window=window)

n = 0
num_blocks = 0
X_concat = np.zeros((num_bands,n_samples//hop),dtype=np.complex64)
while  n_samples - n > hop:

    stft_in.analysis(x[n:n+hop,])
    X_concat[:,num_blocks] = stft_in.X

    n += hop
    num_blocks += 1

# convolve in frequency domain with unknown filter
Y_concat = np.zeros((num_bands,num_blocks), dtype=np.complex64)
for k in range(num_bands):
    Y_concat[k,:] = fftconvolve(X_concat[k,:], W[:,k])[:num_blocks]

# run filters on each block
def run_filters(algorithm, X_concat, Y_concat):
    num_blocks = X_concat.shape[1]
    for n in range(num_blocks):
        algorithm.update(X_concat[:,n], Y_concat[:,n])


class TestAdaptiveFilterFrequencyDomain(TestCase):

    def test_subband_nlms(self):

        subband_nlms = pra.adaptive.SubbandLMS(num_taps=num_taps, 
            num_bands=num_bands, mu=0.5, nlms=True)
        run_filters(subband_nlms, X_concat, Y_concat)
        error_per_band = np.linalg.norm(subband_nlms.W.conj() - W, axis=0)
        error = np.max(abs(error_per_band))
        print('Subband NLMS Reconstruction Error', error)
        self.assertTrue(error < tol)




