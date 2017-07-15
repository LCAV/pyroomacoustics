
from unittest import TestCase
import numpy as np
from scipy.signal import fftconvolve

import pyroomacoustics as pra

# fix seed for repeatability
np.random.seed(0)

h_len = 30
x_len = 1000
SNR = 1000.  # decibels

h_lp = np.fft.irfft(np.ones(5), n=h_len)
h_rand = np.random.randn(h_len)
h_hann = pra.hann(h_len, flag='symmetric')

x = np.random.randn(x_len)
noise = np.random.randn(x_len + h_len - 1)

def generate_signals(SNR, x, h, noise):
    ''' run convolution '''

    # noise standard deviation
    sigma_noise = 10**(-SNR / 20.)

    y = fftconvolve(x, h) 
    y += sigma_noise * noise

    return y, sigma_noise

class TestDeconvolution(TestCase):

    def test_deconvolve_hann_noiseless(self):

        h = h_hann
        h_len = h_hann.shape[0]
        SNR = 1000.
        tol = 1e-7

        y, sigma_noise = generate_signals(SNR, x, h, noise)

        h_hat = pra.experimental.deconvolve(y, x, length=h_len)
        rmse = np.sqrt(np.linalg.norm(h_hat - h)**2 / h_len)

        print('rmse=', rmse, '(tol=', tol, ')')

        self.assertTrue(rmse < tol)

    def test_wiener_deconvolve_hann_noiseless(self):

        h = h_hann
        h_len = h_hann.shape[0]
        SNR = 1000.
        tol = 1e-7

        y, sigma_noise = generate_signals(SNR, x, h, noise)

        h_hat = pra.experimental.wiener_deconvolve(y, x, length=h_len, noise_variance=sigma_noise**2)
        rmse = np.sqrt(np.linalg.norm(h_hat - h)**2 / h_len)

        print('rmse=', rmse, '(tol=', tol, ')')

        self.assertTrue(rmse < tol)

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    h = h_hann
    y, sigma_noise = generate_signals(SNR, x, h, noise)

    h_hat1 = pra.experimental.deconvolve(y, x, length=h_len)
    res1 = np.linalg.norm(y - fftconvolve(x, h_hat1))**2 / y.shape[0]
    mse1 = np.linalg.norm(h_hat1 - h)**2 / h_len

    h_hat2 = pra.experimental.wiener_deconvolve(y, x, length=h_len, noise_variance=sigma_noise**2, let_n_points=15)
    res2 = np.linalg.norm(y - fftconvolve(x, h_hat2))**2 / y.shape[0]
    mse2 = np.linalg.norm(h_hat2 - h)**2 / h_len

    print('MSE naive: rmse=', np.sqrt(mse1), ' res=', pra.dB(res1, power=True))
    print('MSE Wiener: rmse=', np.sqrt(mse2), ' res=', pra.dB(res1, power=True))

    plt.plot(h)
    plt.plot(h_hat1)
    plt.plot(h_hat2)
    plt.legend(['Original', 'Naive', 'Wiener'])
    plt.show()
