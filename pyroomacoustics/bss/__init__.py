# encoding: utf-8
'''
Blind Source Separation
=======================

Implementations of a few blind source separation (BSS) algorithms.

AuxIVA
    | Independent Vector Analysis [1]_
    | :py:mod:`pyroomacoustics.bss.auxiva`
Trinicon
    | Time-domain BSS [2]_
    | :py:mod:`pyroomacoustics.bss.trinicon`
ILRMA
    | Independent Low-Rank Matrix Analysis [3]_
    | :py:mod:`pyroomacoustics.bss.ilrma`
SparseAuxIVA
    | Sparse Independent Vector Analysis [4]_
    | :py:mod `pyroomacoustics.bss.sparseauxiva`


A few commonly used functions, such as projection back, can be found in
:py:mod:`pyroomacoustics.bss.common`.

References
----------

.. [1] N. Ono, *Stable and fast update rules for independent vector analysis based
    on auxiliary function technique,* Proc. IEEE, WASPAA, pp. 189-192, Oct. 2011.


.. [2] R. Aichner, H. Buchner, F. Yan, and W. Kellermann  *A real-time
    blind source separation scheme and its application to reverberant and noisy
    acoustic environments*,  Signal Processing, 86(6), 1260-1277.
    doi:10.1016/j.sigpro.2005.06.022, 2006.

.. [3] D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, *Determined blind
    source separation unifying independent vector analysis and nonnegative matrix
    factorization,* IEEE/ACM Trans. ASLP, vol. 24, no. 9, pp. 1626-1641, Sept. 2016

.. [4] J. Janský, Z. Koldovský, and N. Ono, *A computationally cheaper method
    for blind speech separation based on AuxIVA and incomplete demixing transform,*
    Proc. IEEE, IWAENC, pp. 1-5, Sept. 2016.

'''

from .trinicon import trinicon
from .auxiva import auxiva, f_contrasts
from .ilrma import ilrma
from .sparseauxiva import sparseauxiva
from .common import projection_back, sparir
