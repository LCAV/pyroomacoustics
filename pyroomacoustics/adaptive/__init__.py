'''
Adaptive Filter Algorithms
==========================

This sub-package provides implementations of popular adaptive filter algorithms.

RLS
    | Recursive Least Squares
    | :py:obj:`pyroomacoustics.adaptive.rls` 
LMS
    | Least Mean Squares and Normalized Least Mean Squares
    | :py:obj:`pyroomacoustics.adaptive.lms`

All these classes derive from the base class
:py:obj:`pyroomacoustics.adaptive.adaptive_filter.AdaptiveFilter` that offer
a generic way of running an adaptive filter.

The above classes are applicable for time domain processing. For frequency 
domain adaptive filtering, there is the SubbandLMS class. After using a DFT or
STFT block, the SubbandLMS class can be used to used to apply LMS or NLMS to 
each frequency band. A shorter adaptive filter can be used on each band as 
opposed to the filter required in the time domain version. Roughly, a filter of
M taps applied to each band (total of B) corresponds to a time domain filter 
with N = M x B taps.

How to use the adaptive filter module
-------------------------------------

First, an adaptive filter object is created and all the relevant options
can be set (step size, regularization, etc). Then, the update function
is repeatedly called to provide new samples to the algorithm.

::

    # initialize the filter
    rls = pyroomacoustics.adaptive.RLS(30)

    # run the filter on a stream of samples
    for i in range(100):
        rls.update(x[i], d[i])

    # the reconstructed filter is available
    print('Reconstructed filter:', rls.w)


The SubbandLMS class has the same methods as the time domain 
approaches. However, the signal must be in the frequency domain. This
can be done with the STFT block in the `transform` sub-package of
`pyroomacoustics`.

::

    # initialize STFT and SubbandLMS blocks
    block_size = 128
    stft_x = pra.transform.STFT(N=block_size,
        hop=block_size//2, 
        analysis_window=pra.hann(block_size))
    stft_d = pra.transform.STFT(N=block_size,
        hop=block_size//2, 
        analysis_window=pra.hann(block_size))
    nlms = pra.adaptive.SubbandLMS(num_taps=6, 
        num_bands=block_size//2+1, mu=0.5, nlms=True)

    # preparing input and reference signals
    ...
    
    # apply block-by-block
    for n in range(num_blocks):

        # obtain block
        ...

        # to frequency domain
        stft_x.analysis(x_block)
        stft_d.analysis(d_block)
        nlms.update(stft_x.X, stft_d.X)

        # estimating input convolved with unknown response
        y_hat = stft_d.synthesis(np.diag(np.dot(nlms.W.conj().T,stft_x.X)))

        # AEC output
        E = stft_d.X - np.diag(np.dot(nlms.W.conj().T,stft_x.X))
        out = stft_d.synthesis(E)


Other Available Subpackages
---------------------------

:py:obj:`pyroomacoustics.adaptive.data_structures`
    this provides abstractions for computing functions on regular or irregular
    grids defined on circles and spheres with peak finding methods

:py:obj:`pyroomacoustics.adaptive.util`
    a few methods mainly to efficiently manipulate Toeplitz and Hankel matrices

Utilities
---------

:py:obj:`pyroomacoustics.adaptive.algorithms`
    a dictionary containing all the adaptive filter object subclasses availables indexed by
    keys ``['RLS', 'BlockRLS', 'BlockLMS', 'NLMS', 'SubbandLMS']``

'''

from .adaptive_filter import *
from .lms import *
from .rls import *
from .subband_lms import *
from .util import *
from .data_structures import *

# Create this dictionary as a shortcut to different algorithms
algorithms = {
        'RLS' : RLS,
        'BlockRLS' : BlockRLS,
        'NLMS' : NLMS,
        'BlockLMS' : BlockLMS,
        'SubbandLMS' : SubbandLMS
        }

