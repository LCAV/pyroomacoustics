from __future__ import division, print_function
import numpy as np
import pyroomacoustics as pra
import time

try:
    import pyfftw
    pyfftw_available = True
except ImportError:
    pyfftw_available = False

try:
    import mkl_fft
    mkl_available = True
except ImportError:
    mkl_available = False


n_trials = 1000
nfft = 128
D = 7
x = np.random.randn(nfft, D).astype('float32')

def timing(transform, n_trials):

    dft = pra.transform.DFT(nfft, D, transform=transform)
    start_time = time.time()
    for k in range(n_trials):
        dft.analysis(x)
    analysis_time = (time.time()-start_time)/n_trials * 1e6
    start_time = time.time()
    for k in range(n_trials):
        dft.synthesis()
    synthesis_time = (time.time()-start_time)/n_trials * 1e6

    print("avg %s : %f [1e-6 sec], (analysis, synthesis)=(%f, %f) [1e-6 sec]" % 
        (transform, analysis_time+synthesis_time, analysis_time, synthesis_time))


res = timing('numpy', n_trials)
if pyfftw_available:
    res = timing('fftw', n_trials)
if mkl_available:
    res = timing('mkl', n_trials)

"""
test against without using class
"""
print()
start_time = time.time()
for k in range(n_trials):
    X = np.fft.rfft(x)
analysis_time = (time.time()-start_time)/n_trials * 1e6
start_time = time.time()
for k in range(n_trials):
    x_r = np.fft.irfft(X)
synthesis_time = (time.time()-start_time)/n_trials * 1e6
print("avg numpy w/o class : %f [1e-6 sec], (analysis, synthesis)=(%f, %f) [1e-6 sec]" % 
        (analysis_time+synthesis_time, analysis_time, synthesis_time))

if pyfftw_available:

    # prepare
    a = pyfftw.empty_aligned([nfft, D], dtype='float32')
    b = pyfftw.empty_aligned([nfft//2+1, D], dtype='complex64')
    c = pyfftw.empty_aligned([nfft, D], dtype='float32')
    forward = pyfftw.FFTW(a, b, axes=(0, ))
    backward = pyfftw.FFTW(b, c, axes=(0, ), direction='FFTW_BACKWARD')

    start_time = time.time()
    for k in range(n_trials):
        forward()
    analysis_time = (time.time()-start_time)/n_trials * 1e6
    start_time = time.time()
    for k in range(n_trials):
        backward()
    synthesis_time = (time.time()-start_time)/n_trials * 1e6
    print("avg fftw w/o class : %f [1e-6 sec], (analysis, synthesis)=(%f, %f) [1e-6 sec]" % 
            (analysis_time+synthesis_time, analysis_time, synthesis_time))

if mkl_available:
    start_time = time.time()
    for k in range(n_trials):
        X = mkl_fft.rfft_numpy(x)
    analysis_time = (time.time()-start_time)/n_trials * 1e6
    start_time = time.time()
    for k in range(n_trials):
        x_r = mkl_fft.irfft_numpy(X)
    synthesis_time = (time.time()-start_time)/n_trials * 1e6
    print("avg mkl w/o class : %f [1e-6 sec], (analysis, synthesis)=(%f, %f) [1e-6 sec]" % 
            (analysis_time+synthesis_time, analysis_time, synthesis_time))
    
