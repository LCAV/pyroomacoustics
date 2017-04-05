
import ctypes as _ctypes
import os

path = os.path.dirname(__file__)

try:
    libroom = _ctypes.cdll.LoadLibrary(path + "/libroom.so")
    libroom_available = True
except OSError:
    libroom = False
    libroom_available = False

from libroom_wrapper import *
