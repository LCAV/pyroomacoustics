
'''
Adaptive Filters Example
========================

In this example, we will run adaptive filters for system identification.
'''
from __future__ import division, print_function

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# parameters
length = 15        # the unknown filter length
n_samples = 2000   # the number of samples to run
SNR = 15           # signal to noise ratio

# the unknown filter (unit norm)
w = np.random.randn(length)
w /= np.linalg.norm(w)

# create a known driving signal
x = np.random.randn(n_samples)

# convolve with the unknown filter
d_clean = fftconvolve(x, w)[:n_samples]

# add some noise to the reference signal
d = d_clean + np.random.randn(n_samples) * 10**(-SNR / 20.)

# create a bunch adaptive filters
adfilt = dict(
    nlms=dict(
        filter=pra.adaptive.NLMS(length, mu=0.5), 
        error=np.zeros(n_samples),
        ),
    blocklms=dict(
        filter=pra.adaptive.BlockLMS(length, mu=1./15./2.), 
        error=np.zeros(n_samples),
        ),
    rls=dict(
        filter=pra.adaptive.RLS(length, lmbd=1., delta=2.0),
        error=np.zeros(n_samples),
        ),
    blockrls=dict(
        filter=pra.adaptive.BlockRLS(length, lmbd=1., delta=2.0),
        error=np.zeros(n_samples),
        ),
    )

for i in range(n_samples):
    for algo in adfilt.values():
        algo['filter'].update(x[i], d[i])
        algo['error'][i] = np.linalg.norm(algo['filter'].w - w)

plt.plot(w)
for algo in adfilt.values():
    plt.plot(algo['filter'].w)
plt.title('Original and reconstructed filters')
plt.legend(['groundtruth'] + list(adfilt))

plt.figure()
for algo in adfilt.values():
    plt.semilogy(algo['error'])
plt.legend(adfilt)
plt.title('Convergence to unknown filter')
plt.show()
