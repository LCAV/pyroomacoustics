"""
Block based processing
======================

This subpackage contains routines for continuous *realtime-like* block
processing of signals with the STFT. The routines are written to take
advantage of fast FFT libraries like `pyfftw` or `mkl` when available.

DFT
    | A class for performing DFT of real signals
    | :py:obj:`pyroomacoustics.transform.dft`
STFT
    | A class for continuous STFT processing and frequency domain filtering
    | :py:obj:`pyroomacoustics.transform.stft`

"""


from . import stft
from .dft import DFT
from .stft import STFT
