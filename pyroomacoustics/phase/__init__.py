# encoding: utf-8
'''
Phase Processing
================

This sub-package contains algorithms related to phase specific process, such as phase reconstruction.

Griffin-Lim
    | Phase reconstruction by fixed-point iterations [1]_
    | :py:mod:`pyroomacoustics.phase.gl`

References
----------

.. [1] D. Griffin and J. Lim, “Signal estimation from modified short-time Fourier
    transform,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol.
    32, no. 2, pp. 236–243, 1984.

'''

from .gl import griffin_lim
