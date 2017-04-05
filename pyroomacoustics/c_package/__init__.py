
import ctypes as _ctypes
import os

path = os.path.dirname(__file__)
libroom = _ctypes.cdll.LoadLibrary(path + "/libroom.so")

from libroom_wrapper import *
