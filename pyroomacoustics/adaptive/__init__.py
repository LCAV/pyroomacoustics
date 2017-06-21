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
    keys ``['RLS', 'BlockRLS', 'BlockLMS', 'NLMS']``

'''

from .adaptive_filter import *
from .lms import *
from .rls import *
from .util import *
from .data_structures import *

# Create this dictionary as a shortcut to different algorithms
algorithms = {
        'RLS' : RLS,
        'BlockRLS' : BlockRLS,
        'NLMS' : NLMS,
        'BlockLMS' : BlockLMS,
        }

