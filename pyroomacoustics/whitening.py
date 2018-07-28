# @version: 1.0  2018, Juan Azcarreta
# Whitening algorithm as a preprocessing step for blind source separation

from __future__ import division

import numpy as np

#=========================================================================
# Whitening in the time-frequency domain.
#=========================================================================

def whitening(X):
    '''
    This function computes the time-frequency domain decorrelation (whitening)
    Withening based on the method presented in Section II of the following paper:
    Cichocki, Andrzej & Osowski, S & Siwek, Krzysztof. (2004). Prewhitening Algorithms of Signals
    in the Presence of White Noise.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the observed signal
        n_src: int, optional
        The number of sources or independent components
    Returns
    ----------
    Y: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the whitened signal
    -------
    '''

    [n_frames, n_freq, n_chan] = X.shape
    Y = np.zeros([n_frames, n_freq, n_chan], dtype=np.complex)
    fudge = 1E-16
    for f in range(n_freq):
        Xi = X[:,f,:].T
        cov_x = np.dot(Xi, np.conj(Xi).T)/n_frames
        [eigenvalues, eigenvectors] = np.linalg.eigh(cov_x)
        Y[:,f,:] = (np.dot(np.conj(eigenvectors).T, Xi).T/np.sqrt(eigenvalues) + fudge)
    return Y
