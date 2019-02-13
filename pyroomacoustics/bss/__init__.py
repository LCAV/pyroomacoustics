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

.. [1] N. Ono, *Stable and fast update rules for independent vector analysis
    based on auxiliary function technique*, Proc. IEEE, WASPAA, 2011.

.. [2] R. Aichner, H. Buchner, F. Yan, and W. Kellermann  *A real-time
    blind source separation scheme and its application to reverberant and noisy
    acoustic environments*,  Signal Processing, 86(6), 1260-1277.
    doi:10.1016/j.sigpro.2005.06.022, 2006.

.. [3] D. Kitamura, N. Ono, H. Sawada, H. Kameoka, and H. Saruwatari
    *Determined Blind Source Separation with Independent Low-Rank Matrix Analysis*,
    in Audio Source Separation, S. Makino, Ed. Springer, 2018, pp.  125-156.

.. [4] Janský, Jakub & Koldovský, Zbyněk & Ono, Nobutaka. (2016). A computationally
    cheaper method for blind speech separation based on AuxIVA and incomplete demixing
    transform. 1-5. 10.1109/IWAENC.2016.7602921.

'''

from .trinicon import trinicon
from .auxiva import auxiva, f_contrasts
from .ilrma import ilrma
from .sparseauxiva import sparseauxiva
from .common import projection_back, sparir
