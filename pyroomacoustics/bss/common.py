'''
Common Functions used in BSS algorithms

2018 (c) Robin Scheibler, MIT License
'''
import numpy as np

def projection_back(Y, ref, clip_up=None, clip_down=None):
    '''
    This function computes the frequency-domain filter that minimizes
    the squared error to a reference signal. This is commonly used
    to solve the scale ambiguity in BSS.

    Derivation of the projection
    ----------------------------

    The optimal filter `z` minimizes the squared error.
    
    .. math::

        \min E[|z^* y - x|^2]

    It should thus satsify the orthogonality condition
    and can be derived as follows

    .. math::

        0 & = E[y^*\\, (z^* y - x)]

        0 & = z^*\\, E[|y|^2] - E[y^* x]

        z^* & = \\frac{E[y^* x]}{E[|y|^2]}

        z & = \\frac{E[y x^*]}{E[|y|^2]}

    In practice, the expectations are replaced by the sample
    mean.

    Parameters
    ----------
    Y: array_like (n_frames, n_bins, n_channels)
        The STFT data to project back on the reference signal
    ref: array_like (n_frames, n_bins)
        The reference signal
    clip_up: float, optional
        Limits the maximum value of the gain (default no limit)
    clip_down: float, optional
        Limits the minimum value of the gain (default no limit)
    '''

    num = np.sum(np.conj(ref[:,:,None]) * Y, axis=0)
    denom = np.sum(np.abs(Y)**2, axis=0)

    c = np.ones(num.shape, dtype=np.complex)
    I = denom > 0.
    c[I] = num[I] / denom[I]

    if clip_up is not None:
        I = np.logical_and(np.abs(c) > clip_up, np.abs(c) > 0)
        c[I] *= clip_up / np.abs(c[I])

    if clip_down is not None:
        I = np.logical_and(np.abs(c) < clip_down, np.abs(c) > 0)
        c[I] *= clip_down / np.abs(c[I])

    return c
