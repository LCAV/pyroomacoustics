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


from .dft import DFT
from .stft import STFT
from . import stft


def analysis(*args, **kwargs):
    import warnings
    warnings.warn(
        "The `pyroomacoustics.transform.analysis` function is deprecated in favor of "
        "`pyroomacoustics.transform.stft.analysis`.",
        DeprecationWarning,
    )
    return stft.analysis(*args, **kwargs)


def synthesis(*args, **kwargs):
    import warnings
    warnings.warn(
        "The `pyroomacoustics.transform.synthesis` function is deprecated in favor of "
        "'pyroomacoustics.transform.stft.synthesis`.",
        DeprecationWarning,
    )
    return stft.synthesis(*args, **kwargs)


def compute_synthesis_window(*args, **kwargs):
    import warnings
    warnings.warn(
        "The `pyroomacoustics.transform.compute_synthesis_window` function is "
        "deprecated in favor of "
        "`pyroomacoustics.transform.stft.compute_synthesis_window`.",
        DeprecationWarning,
    )
    return stft.compute_synthesis_window(*args, **kwargs)
