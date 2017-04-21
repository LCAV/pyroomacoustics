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

The docstring examples assume that `numpy` has been imported as `np`::

  >>> import pyroomacoustics as pra

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(np.sort)
  ... # doctest: +SKIP

For some objects, ``np.info(obj)`` may provide additional help.  This is
particularly true if you see the line "Help on ufunc object:" at the top
of the help() page.  Ufuncs are implemented in C, not Python, for speed.
The native Python help() does not know how to view their help, but our
np.info() function does.
To search for documents containing a keyword, do::

  >>> np.lookfor('keyword')
  ... # doctest: +SKIP

General-purpose documents like a glossary and help on the basic concepts
of numpy are available under the ``doc`` sub-module::

  >>> from numpy import doc
  >>> help(doc)
  ... # doctest: +SKIP

Available subpackages
---------------------
acoustics
    Acoustics and psychoacoustics routines, mel-scale, critcal bands, etc.
beamforming
    Microphone arrays and beamforming routines.
bss
    Blind source separation.
geometry
    Core geometry routine for the image source model.
metrics
    Performance metrics like mean-squared error, median, Itakura-Saito, etc.
multirate
    Rate conversion routines.
parameters
    Global parameters, i.e. for physics constants.
recognition
    Hidden Markov Model and TIMIT database structure.
room
    Abstraction of room and image source model.
soundsource
    Abstraction for a sound source.
stft
    STFT processing engine.
sync
    A few routines to help synchronize signals.
utilities
    A bunch of routines to do various stuff.
wall
    Abstraction for walls of a room.
windows
    Tapering windows for spectral analysis.

Utilities
---------
__version__
    pyroomacoustics version string

'''

__version__ = '1.1.0'

from . import c_package

from .room import *
from .beamforming import *
from .soundsource import *
from .parameters import *
from .stft import *
from .utilities import *
from .windows import *
from .sync import *
from .metrics import *
from .bss import *
from .multirate import *
from .acoustics import *
from .recognition import *
