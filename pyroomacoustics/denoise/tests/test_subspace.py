from __future__ import division, print_function
from unittest import TestCase
import numpy as np
from pyroomacoustics.denoise import Subspace

tol = 1e-7


def test_cov_computation(skip=1):
    """
    Seed is set because not every random signal will satisfy test.
    The input signal should have noise frames in each frame, otherwise
    the estimated covariance matrix of the class implementation will differ
    than if all samples are available.
    """
    np.random.seed(0)

    frame_len = 256
    lookback = 5
    threshold = 1
    data_type = 'float32'

    # random signal
    hop = frame_len//2
    noisy_signal = np.random.randn(hop*lookback+frame_len)

    # apply denoising to run covariance estimation
    scnr = Subspace(frame_len, lookback=lookback, skip=skip,
                    thresh=threshold, data_type=data_type)
    n = 0
    covs = []
    while noisy_signal.shape[0] - n >= hop:
        scnr.apply(noisy_signal[n:n + hop])
        n += hop
        covs.append((scnr._cov_sn[-1]))

    # compute covariance with all samples instead of update
    if data_type is 'float64':
        data_type = np.float64
    else:
        data_type = np.float32
    n_frames = lookback * (hop//skip)
    cov_sn_true = np.zeros((frame_len, frame_len), dtype=data_type)
    cov_n_true = np.zeros((frame_len, frame_len), dtype=data_type)
    n_noise = 0
    for i in range(lookback):
        for k in range(0, hop, skip):
            a = i*hop + k
            b = a + frame_len
            _noisy_signal = noisy_signal[a:b]
            new_cov = np.outer(_noisy_signal, _noisy_signal).astype(data_type)

            cov_sn_true += new_cov

            # noise cov
            if np.std(_noisy_signal) ** 2 < threshold:
                cov_n_true += new_cov
                n_noise += 1
    cov_sn_true /= n_frames
    cov_n_true /= n_noise

    cov_sn_test = np.linalg.norm(cov_sn_true - scnr.cov_sn) / cov_sn_true.size
    cov_n_test_1 = np.linalg.norm(cov_n_true - scnr.cov_n) / cov_n_true.size
    cov_n_test_2 = int(n_noise - sum(scnr.n_noise_frames))

    return cov_sn_test, cov_n_test_1, cov_n_test_2


class TestSubspace(TestCase):

    def test_cov_estimation_skip_1(self):
        res = test_cov_computation(skip=1)
        self.assertTrue(res[0] < tol)
        self.assertTrue(res[1] < tol)
        self.assertEqual(res[2], 0)

    def test_cov_estimation_skip_5(self):
        res = test_cov_computation(skip=5)
        self.assertTrue(res[0] < tol)
        self.assertTrue(res[1] < tol)
        self.assertEqual(res[2], 0)


if __name__ == "__main__":
    print("SKIP = 1")
    res = test_cov_computation(skip=1)
    print("COV_SN error: {}".format(res[0]))
    print("COV_N error: {}".format(res[1]))
    print("n_noise_frames error: {}".format(res[2]))

    print()
    print("SKIP = 5")
    res = test_cov_computation(skip=5)
    print("COV_SN error: {}".format(res[0]))
    print("COV_N error: {}".format(res[1]))
    print("n_noise_frames error: {}".format(res[2]))

