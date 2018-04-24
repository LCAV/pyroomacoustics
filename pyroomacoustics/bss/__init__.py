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

'''

from .trinicon import trinicon
from .auxiva import auxiva, f_contrasts
from .common import projection_back
