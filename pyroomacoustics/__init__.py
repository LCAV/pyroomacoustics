'''
pyroomacoustics
===============

Provides
  1. Room impulse simulations via the image source model
  2. Simulation of sound propagation using STFT engine
  3. Reference implementations of popular algorithms for

    * beamforming
    * direction of arrival
    * adaptive filtering
    * source separation
    * single channel denoising
    * etc

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`the pyroomacoustics readthedocs page <http://pyroomacoustics.readthedocs.io>`_.

We recommend exploring the docstrings using
`IPython <http://ipython.scipy.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `pyroomacoustics` has been imported as `pra`::

  >>> import pyroomacoustics as pra

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(pra.stft.STFT)
  ... # doctest: +SKIP

Available submodules
---------------------
:py:obj:`pyroomacoustics.acoustics`
    Acoustics and psychoacoustics routines, mel-scale, critcal bands, etc.

:py:obj:`pyroomacoustics.beamforming`
    Microphone arrays and beamforming routines.

:py:obj:`pyroomacoustics.geometry`
    Core geometry routine for the image source model.

:py:obj:`pyroomacoustics.metrics`
    Performance metrics like mean-squared error, median, Itakura-Saito, etc.

:py:obj:`pyroomacoustics.multirate`
    Rate conversion routines.

:py:obj:`pyroomacoustics.parameters`
    Global parameters, i.e. for physics constants.

:py:obj:`pyroomacoustics.recognition`
    Hidden Markov Model and TIMIT database structure.

:py:obj:`pyroomacoustics.room`
    Abstraction of room and image source model.

:py:obj:`pyroomacoustics.soundsource`
    Abstraction for a sound source.

:py:obj:`pyroomacoustics.stft`
    **Deprecated** Replaced by the methods in :py:obj:`pyroomacoustics.transform`

:py:obj:`pyroomacoustics.sync`
    A few routines to help synchronize signals.

:py:obj:`pyroomacoustics.utilities`
    A bunch of routines to do various stuff.

:py:obj:`pyroomacoustics.wall`
    Abstraction for walls of a room.

:py:obj:`pyroomacoustics.windows`
    Tapering windows for spectral analysis.

Available subpackages
---------------------

:py:obj:`pyroomacoustics.adaptive`
    Adaptive filter algorithms

:py:obj:`pyroomacoustics.bss`
    Blind source separation.

:py:obj:`pyroomacoustics.datasets`
    Wrappers around a few popular speech datasets

:py:obj:`pyroomacoustics.denoise`
    Single channel noise reduction methods

:py:obj:`pyroomacoustics.doa`
    Direction of arrival finding algorithms

:py:obj:`pyroomacoustics.phase`
    Phase-related processing

:py:obj:`pyroomacoustics.transform`
    Block frequency domain processing tools


Utilities
---------
__version__
    pyroomacoustics version string

'''

from .version import __version__

from . import libroom as libroom

from .room import *
from .beamforming import *
from .soundsource import *
from .parameters import *
from .stft import *
from .utilities import *
from .windows import *
from .sync import *
from .metrics import *
from .multirate import *
from .acoustics import *
from .recognition import *

from . import doa
from . import adaptive
from . import transform
from . import transform as realtime  # to be deprecated
from . import experimental
from . import datasets
from . import bss
from . import denoise
from . import phase

import warnings
warnings.warn(
    "Do not import the 'realtime' module as it will be deprecated soon. It will be replaced by the 'transform' module."
    , DeprecationWarning
)
