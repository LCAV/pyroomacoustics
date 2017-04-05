
import ctypes as _ctypes
import os

from ctypes.util import find_library
path = find_library("libroom.so")
print "found", path

libroom = _ctypes.cdll.LoadLibrary("libroom.so")

from libroom_wrapper import *
