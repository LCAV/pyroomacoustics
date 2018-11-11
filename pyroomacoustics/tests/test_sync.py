import numpy as np
from scipy.signal import fftconvolve
import pyroomacoustics as pra

def test_correlate():
    
    N = [100, 200, 50, 37]
    M = [47, 82, 151, 893]

    for n, m in zip(N,M):

        x = np.random.randn(n)
        y = np.random.randn(m)

        assert np.allclose(pra.correlate(x,y), np.correlate(x, y, mode='full'))

def test_tdoa_delay_int():

    N = [100, 200, 50, 1000]
    M = [47, 37, 12, 128]
    delays = [27, 4, 10, 347]

    for n,m,tau in zip(N,M,delays):

        pulse = np.random.randn(m)
        x1 = np.zeros(n)
        x1[:m] = pulse
        x2 = np.zeros(n)
        x2[tau:tau+m] = pulse

        d1 = pra.tdoa(x1, x2)
        d2 = pra.tdoa(x2, x1)

        assert int(d1) == -tau
        assert int(d2) == tau

def test_tdoa_delay_frac_phat():

    N = [14, 200, 70, 40, 28]
    M = [3, 78, 12, 10, 12]

    # important to start at zero and only use positive
    # delays due to fractional_delay_filter_bank function
    # subtracting minimum delay
    delays = [0.0, 0.3, 3.5, 10.1, 5.7]

    zero_delay_filter = pra.fractional_delay(0.)
    frac_filters = pra.fractional_delay_filter_bank(delays)

    # test without explicit sampling frequency
    for n,m,tau,fil in zip(N,M,delays,frac_filters):

        pulse = np.random.randn(m)
        signal = np.zeros(n)
        signal[:m] = pulse
        x1 = fftconvolve(zero_delay_filter, signal)
        x2 = fftconvolve(signal, fil)

        d1 = pra.tdoa(x1, x2, interp=20, phat=True)
        d2 = pra.tdoa(x2, x1, interp=20, phat=True)

        assert np.allclose(d1, -tau)
        assert np.allclose(d2, tau)

        fs = 100
        d3 = pra.tdoa(x1, x2, interp=20, phat=True, fs=fs)
        d4 = pra.tdoa(x2, x1, interp=20, phat=True, fs=fs)

        assert np.allclose(d3, -tau / fs)
        assert np.allclose(d4, tau / fs)





if __name__ == '__main__':

    test_correlate()
    test_tdoa_delay_int()
    test_tdoa_delay_frac_phat()
