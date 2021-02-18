"""
Make sure the GL phase reconstruction is working

2019 (c) Robin Scheibler
Part of the pyroomacoustics package
MIT License
"""

import numpy as np
from scipy.io import wavfile
import unittest
import pyroomacoustics as pra

test_tol = 1e-2

filename = "examples/input_samples/cmu_arctic_us_axb_a0004.wav"
fs, audio = wavfile.read(filename)

fft_size = 512
hop = fft_size // 4
win_a = np.hamming(fft_size)
win_s = pra.transform.compute_synthesis_window(win_a, hop)
n_iter = 200

engine = pra.transform.STFT(
    fft_size, hop=hop, analysis_window=win_a, synthesis_window=win_s
)
X = engine.analysis(audio)
X_mag = np.abs(X)
X_mag_norm = np.linalg.norm(X_mag) ** 2


def compute_error(X_mag, y):
    """ routine to compute the spectral distance """
    Y_2 = engine.analysis(y)
    return np.linalg.norm(X_mag - np.abs(Y_2)) ** 2 / X_mag_norm


np.random.seed(0)
ini = [None, "random", X]

# The actual test case
# We use deterministic phase initialization (to zero)
class TestGL(unittest.TestCase):
    def test_griffin_lim(self):
        rec = pra.phase.griffin_lim(X_mag, hop, win_a, n_iter=n_iter)
        error = compute_error(X_mag, rec)
        self.assertTrue(error < test_tol)

    def test_griffin_lim_rand(self):
        np.random.seed(0)
        rec = pra.phase.griffin_lim(X_mag, hop, win_a, n_iter=n_iter, ini="random")
        error = compute_error(X_mag, rec)
        self.assertTrue(error < test_tol)

    def test_griffin_lim_true(self):
        rec = pra.phase.griffin_lim(X_mag, hop, win_a, n_iter=n_iter, ini=X)
        error = compute_error(X_mag, rec)
        self.assertTrue(error < test_tol)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # monitor convergence
    errors = []

    # the callback to track the spectral distance convergence
    def cb(epoch, Y, y):
        if epoch % 10 == 0:
            errors.append(compute_error(X_mag, y))

    pra.phase.griffin_lim(X_mag, hop, win_a, n_iter=n_iter, callback=cb)

    plt.semilogy(np.arange(len(errors)) * 10, errors)
    plt.show()

    # also run the tests to make sure they pass
    unittest.main()
