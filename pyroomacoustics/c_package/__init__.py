''' 
This subpackage is a C extension to accelerate the computation of image
sources. It implements two main algorithms, one for rectangular rooms, and one
for arbitrary polyhedral rooms.

The C extension is used instead of the pure python code whenever available.
'''

import ctypes as _ctypes
import os
import glob

path = os.path.dirname(__file__)

try:
    from . import libroom
    libroom_available = True
except:
    libroom = False
    libroom_available = False

from .libroom_wrapper import *
