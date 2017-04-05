
import ctypes as _ctypes
import os
import glob

path = os.path.dirname(__file__)

try:
    # we need the matching because python3 appends some os info to the name
    match_files = glob.glob(path + "/libroom*so")
    libroom = _ctypes.cdll.LoadLibrary(match_files[0])
    libroom_available = True
except:
    libroom = False
    libroom_available = False

from .libroom_wrapper import *
