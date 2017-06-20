'''
Direction of Arrival Finding
============================

This sub-package provides implementations of popular direction of arrival findings algorithms.

MUSIC
    | Multiple Signal Classification [1]_ 
    | :py:obj:`pyroomacoustics.doa.music` 
SRP-PHAT 
    | Steered Response Power -- Phase Transform [2]_ 
    | :py:obj:`pyroomacoustics.doa.srp`
CSSM 
    | Coherent Signal Subspace Method [3]_
    | :py:obj:`pyroomacoustics.doa.cssm`
WAVES 
    | Weighted Average of Signal Subspaces [4]_ 
    | :py:obj:`pyroomacoustics.doa.waves`
TOPS 
    | Test of Orthogonality of Projected Subspaces [5]_
    | :py:obj:`pyroomacoustics.doa.tops`
FRIDA 
    | Finite Rate of Innovation Direction of Arrival [6]_ 
    | :py:obj:`pyroomacoustics.doa.frida`

All these classes derive from the abstract base class
:py:obj:`pyroomacoustics.doa.doa.DOA` that offers generic methods for finding
and visualizing the locations of acoustic sources.

The constructor can be called once to build the DOA finding object. Then, the
method :py:obj:`pyroomacoustics.doa.doa.DOA.locate_sources` performs DOA
finding based on time-frequency passed to it as an argument. Extra arguments
can be supplied to indicate which frequency bands should be used for
localization.

How to use the DOA module
-------------------------

Here ``R`` is a 2xQ ndarray that contains the locations of the Q microphones
in the columns, ``fs`` is the sampling frequency of the input signal, and
``nfft`` the length of the FFT used.

The STFT snapshots are passed to the localization methods in the X ndarray of
shape ``Q x (nfft // 2 + 1) x n_snapshots``, where ``n_snapshots`` is the
number of STFT frames to use for the localization. The option ``freq_bins``
can be provided to specify which frequency bins to use for the localization.

    >>> doa = pyroomacoustics.doa.MUSIC(R, fs, nfft)
    >>> doa.locate_sources(X, freq_bins=np.arange(20, 40))

Other Available Subpackages
---------------------------

:py:obj:`pyroomacoustics.doa.grid`
    this provides abstractions for computing functions on regular or irregular
    grids defined on circles and spheres with peak finding methods

:py:obj:`pyroomacoustics.doa.plotters`
    a few methods to plot functions and points on circles or spheres

:py:obj:`pyroomacoustics.doa.detect_peaks`
    1D peak detection routine from Marcos Duarte

:py:obj:`pyroomacoustics.doa.tools_frid_doa_plane`
    routines implementing FRIDA algorithm

Utilities
---------

:py:obj:`pyroomacoustics.doa.algorithms`
    a dictionary containing all the DOA objects subclass availables indexed by
    keys ``['MUSIC', 'SRP', 'CSSM', 'WAVES', 'TOPS', 'FRIDA']``

References
----------

.. [1] R. Schmidt, *Multiple emitter location and signal parameter estimation*, 
    IEEE Trans. Antennas Propag., Vol. 34, Num. 3, pp 276--280, 1986

.. [2] J. H. DiBiase, *A high-accuracy, low-latency technique for talker localization 
    in reverberant environments using microphone arrays*, PHD Thesis, Brown University, 2000

.. [3] H. Wang, M. Kaveh, *Coherent signal-subspace processing for the detection and 
    estimation of angles of arrival of multiple wide-band sources*, IEEE Trans. Acoust., 
    Speech, Signal Process., Vol. 33, Num. 4, pp 823--831, 1985

.. [4] E. D. di Claudio, R. Parisi, *WAVES: Weighted average of signal subspaces for 
    robust wideband direction finding*, IEEE Trans. Signal Process., Vol. 49, Num. 10, 
    2179--2191, 2001

.. [5] Y. Yeo-Sun, L. M. Kaplan, J. H. McClellan, *TOPS: New DOA estimator for wideband 
    signals*, IEEE Trans. Signal Process., Vol. 54, Num 6., pp 1977--1989, 2006

.. [6] H. Pan, R. Scheibler, E. Bezzam, I. Dokmanic, and M. Vetterli, *FRIDA:
    FRI-based DOA estimation for arbitrary array layouts*, Proc. ICASSP,
    pp 3186-3190, 2017

'''

from .doa import *
from .srp import *
from .music import *
from .cssm import *
from .waves import *
from .tops import *
from .frida import *
from .grid import *
from .utils import *

# Create this dictionary as a shortcut to different algorithms
algorithms = {
        'SRP' : SRP,
        'MUSIC' : MUSIC,
        'CSSM' : CSSM,
        'WAVES' : WAVES,
        'TOPS' : TOPS,
        'FRIDA' : FRIDA,
        }

