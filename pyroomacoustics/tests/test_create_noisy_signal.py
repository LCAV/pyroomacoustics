from __future__ import division, print_function
from unittest import TestCase
import numpy as np
import os
from pyroomacoustics import create_noisy_signal, rms, normalize
from scipy.io import wavfile

tol = 1e-5

signal_fp = os.path.join(os.path.dirname(__file__), '..', '..', 'examples',
                         'input_samples', 'cmu_arctic_us_aew_a0001.wav')

def white_noise(snr):
    np.random.seed(0)
    noisy_signal, signal, noise, fs = create_noisy_signal(signal_fp, snr=snr)
    _snr = 20 * np.log10(rms(signal) / rms(noise))
    err = abs(snr-_snr)
    return err, noisy_signal, fs


class TestCreateNoisySignal(TestCase):

    def test_snr_error(self):
        snr_test = 0
        res = white_noise(snr_test)
        self.assertTrue(res[0] < tol)


if __name__ == "__main__":
    snr = 0
    err, noisy_signal, fs = white_noise(snr)
    print("SNR error: {}".format(err))

    # write to WAV, need to cast to `np.float32` for float values
    # see table here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    wavfile.write("noisy_signal_{}dB.wav".format(snr), fs,
                  noisy_signal.astype(np.float32))
