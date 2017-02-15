
import ctypes
import os

path = os.path.dirname(__file__)
libroom = ctypes.cdll.LoadLibrary(path + "/libroom.so")

from libroom_wrapper import *
