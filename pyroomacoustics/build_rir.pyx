# cython: infer_types=True

import numpy as np
cimport cython

from libc.math cimport floor, ceil

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_rir_builder(
        double [:] rir,
        double [:] time,
        double [:] alpha,
        int [:] visibility,
        int fs,
        int fdl,
        int lut_gran=20,
        ):
    '''
    Fast impulse response builder. This function takes the image source delays
    and amplitude and fills the impulse response. It uses a linear interpolation
    of the sinc function for speed.

    Parameters
    ----------
    rir: ndarray (double)
        The array to receive the impulse response. It should be filled with
        zero and of the correct size
    time: ndarray (double)
        The array of delays for the image sources
    alpha: ndarray (double)
        The array of attenuations for the image sources
    visibility: ndarray (int)
        Contains 1 if the image source is visible, 0 if not
    fs: int
        The sampling frequency
    fdl: int
        The length of the fractional delay filter (should be odd)
    lut_gran: int
        The number of point per unit in the sinc interpolation table
    '''

    fdl2 = (fdl - 1) // 2
    n_times = time.shape[0]

    assert time.shape[0] == visibility.shape[0]
    assert time.shape[0] == alpha.shape[0]
    assert fdl % 2 == 1

    # check the size of the return array
    max_sample = ceil(fs * np.max(time)) + fdl2
    min_sample = floor(fs * np.min(time)) - fdl2
    assert min_sample >= 0
    assert max_sample < rir.shape[0]

    # create a look-up table of the sinc function and
    # then use linear interpolation
    cdef float delta = 1. / lut_gran
    cdef int lut_size = (fdl + 1) * lut_gran + 1
    n = np.linspace(-fdl2-1, fdl2 + 1, lut_size)

    cdef double [:] sinc_lut = np.sinc(n)
    cdef double [:] hann = np.hanning(fdl)
    cdef int lut_pos, i, f, time_ip
    cdef float x_off, x_off_frac, sample_frac

    for i in range(n_times):
        if visibility[i] == 1:
            # decompose integer and fractional delay
            sample_frac = fs * time[i]
            time_ip = int(floor(sample_frac))
            time_fp = sample_frac - time_ip

            # do the linear interpolation
            x_off_frac = (1. - time_fp) * lut_gran
            lut_gran_off = int(floor(x_off_frac))
            x_off = (x_off_frac - lut_gran_off)
            lut_pos = lut_gran_off
            k = 0
            for f in range(-fdl2, fdl2+1):
                rir[time_ip + f] += alpha[i] * hann[k] * (sinc_lut[lut_pos] 
                        + x_off * (sinc_lut[lut_pos+1] - sinc_lut[lut_pos]))
                lut_pos += lut_gran
                k += 1
