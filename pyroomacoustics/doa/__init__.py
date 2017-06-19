'''
Direction of Arrival Finding
============================

This sub-package provides implementations of for direction of arrival findings algorithms.

The package has one main class `DOA` that provides generic methods for initialization and
direction findings. Sub-classes are implemented for specific algorithms such as MUSIC, CSSM, etc.
'''

from .doa import *
from .srp import *
from .music import *
from .cssm import *
from .waves import *
from .tops import *
from .frida import *
from .grid import *

# Create this dictionary as a shortcut to different algorithms
algos = {
        'SRP' : SRP,
        'MUSIC' : MUSIC,
        'CSSM' : CSSM,
        'WAVES' : WAVES,
        'TOPS' : TOPS,
        'FRIDA' : FRIDA,
        }

