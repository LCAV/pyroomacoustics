# cython: infer_types=True

import numpy as np

cimport cython
from libc.math cimport ceil, floor


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

    #Total number of points in 81 fractional delay
    cdef int lut_size = (fdl + 1) * lut_gran + 1

    #equal space between -41 to +41 for 1641 length as each point between -40 to +40 represents 20 samples in the sinc
    n = np.linspace(-fdl2-1, fdl2 + 1, lut_size)

    cdef double [:] sinc_lut = np.sinc(n) #Sinc over linspace n
    cdef double [:] hann = np.hanning(fdl) #Hanning window of size 81
    cdef int lut_pos, i, f, time_ip
    cdef float x_off, x_off_frac, sample_frac
    g =[]
    print_filter=0
    pf=[]

    #Loop through each image source
    for i in range(n_times):
        if visibility[i] == 1:
            # decompose integer and fractional delay
            sample_frac = fs * time[i] #Samples in fraction eg 250.567 sample , actual time of arrival of the image source to the microphone
            time_ip = int(floor(sample_frac)) #Get the integer value of the sample eg 250th sample as int(250.567) = 250
            time_fp = sample_frac - time_ip #Get the fractional sample 250.567-250= 0.567 samples

            # do the linear interpolation
            x_off_frac = (1. - time_fp) * lut_gran #(1-0.567) *20 = 8.66 , as each point represents 20 samples in the sinc table
            lut_gran_off = int(floor(x_off_frac)) #int(8.66) = 8 sample in the sinc table
            x_off = (x_off_frac - lut_gran_off) #fractional in the sinc table 8.66-8=0.66
            lut_pos = lut_gran_off #lut_pos=8
            k = 0

            #Loop through -40 to 41 , which accounts for 81 , as it is the amount of fractional delay every dirac goes through, the sinc table helps spread the energy to -40 to +40 samples in the RIR.
            for f in range(-fdl2, fdl2+1):
                rir[time_ip + f] += alpha[i] * hann[k] * (sinc_lut[lut_pos]
                        + x_off * (sinc_lut[lut_pos+1] - sinc_lut[lut_pos]))
                pf.append(rir[time_ip+f])
                lut_pos += lut_gran
                k += 1

def fast_window_sinc_interpolator(double [:] vectorized_time_fp, int window_length, double [:,:] vectorized_interpolated_sinc): #Takes fractional part of the delay of IS k

    cdef double [:] hann_wd = np.hanning(window_length)
    cdef int fdl2 = (window_length - 1) // 2
    cdef int lut_gran=20
    cdef int lut_size = (window_length + 1) * lut_gran + 1
    n_ = np.linspace(-fdl2 - 1, fdl2 + 1, lut_size)
    cdef double [:] sinc_lut=np.sinc(n_)
    cdef int img_src = 0

    for time_fp in vectorized_time_fp:
        x_off_frac = (1 - time_fp) * lut_gran
        lut_gran_off = int(np.floor(x_off_frac))
        x_off = x_off_frac - lut_gran_off
        lut_pos = lut_gran_off
        filter_sample = 0

        for f in range(-fdl2, fdl2 + 1):
            vectorized_interpolated_sinc[img_src,filter_sample] = hann_wd[filter_sample]*(sinc_lut[lut_pos] + x_off * (sinc_lut[lut_pos + 1] - sinc_lut[lut_pos]))
            lut_pos += lut_gran
            filter_sample += 1

        img_src+=1

    return vectorized_interpolated_sinc

cdef int val_i
import multiprocessing

nthread = multiprocessing.cpu_count()

def fast_convolution_4 (
        double complex [:] a,
        double complex [:] b,
        double complex [:] c,
        double complex [:] d,
        int final_fir_IS_len,):

    cdef double complex [:] out = np.zeros(final_fir_IS_len,dtype=np.complex_)

    a=np.fft.fft(a)
    b=np.fft.fft(b)
    c=np.fft.fft(c)
    d=np.fft.fft(d)

    for val_i in range(final_fir_IS_len):
        out[val_i]=a[val_i]*b[val_i]*c[val_i]*d[val_i]

    out=np.fft.ifft(out)
    return out


def fast_convolution_3 (
        double complex [:] a,
        double complex [:] b,
        double complex [:] c,
        int final_fir_IS_len,):

    cdef double complex [:] out = np.zeros(final_fir_IS_len,dtype=np.complex_)

    a=np.fft.fft(a)
    b=np.fft.fft(b)
    c=np.fft.fft(c)

    for val_i in range(final_fir_IS_len):
        out[val_i]=a[val_i]*b[val_i]*c[val_i]

    out=np.fft.ifft(out)
    return out
