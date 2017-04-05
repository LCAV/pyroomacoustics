
import ctypes as _ctypes
import os

from ctypes.util import find_library
path = find_library("libroom.so")
print "found", path

path = os.path.dirname(__file__)
libroom = _ctypes.cdll.LoadLibrary(path + "/libroom.so")

from libroom_wrapper import *
