"""
Single Channel Noise Reduction
==============================

Collection of single channel noise reduction (SCNR) algorithms for speech:

- :doc:`Spectral Subtraction <pyroomacoustics.denoise.spectral_subtraction>` [1]_
- :doc:`Subspace Approach <pyroomacoustics.denoise.subspace>` [2]_
- :doc:`Iterative Wiener Filtering <pyroomacoustics.denoise.iterative_wiener>` [3]_

At `this repository <https://github.com/santi-pdp/segan>`_, a deep learning approach in Python can be found.

References
----------

.. [1] M. Berouti, R. Schwartz, and J. Makhoul, *Enhancement of speech corrupted by acoustic noise,*
    ICASSP '79. IEEE International Conference on Acoustics, Speech, and Signal Processing, 1979, pp. 208-211.

.. [2] Y. Ephraim and H. L. Van Trees, *A signal subspace approach for speech enhancement,*
    IEEE Transactions on Speech and Audio Processing, vol. 3, no. 4, pp. 251-266, Jul 1995.

.. [3] J. Lim and A. Oppenheim, *All-Pole Modeling of Degraded Speech,*
    IEEE Transactions on Acoustics, Speech, and Signal Processing 26.3 (1978): 197-210.

"""

from .spectral_subtraction import *
from .subspace import *
from .iterative_wiener import *
