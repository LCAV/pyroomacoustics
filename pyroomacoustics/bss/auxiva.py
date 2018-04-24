'''
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

2018 (c) Robin Scheibler, MIT License
'''
import numpy as np

from pyroomacoustics import stft, istft
from .common import projection_back

# A few contrast functions
f_contrasts = {
        'norm' : { 'f' : (lambda r,c,m : c * r), 'df' : (lambda r,c,m : c) },
        'cosh' : { 'f' : (lambda r,c,m : m * np.log(np.cosh(c * r))), 'df' : (lambda r,c,m : c * m * np.tanh(c * r)) }
        }

def auxiva(X, n_src=None, n_iter=20, proj_back=True, W0=None,
        f_contrast=None, f_contrast_args=[],
        return_filters=False, callback=None):

    '''
    Implementation of AuxIVA algorithm for BSS presented in

    N. Ono, *Stable and fast update rules for independent vector analysis based
    on auxiliary function technique*, Proc. IEEE, WASPAA, 2011.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for demixing matrix
    f_contrast: dict of functions
        A dictionary with two elements 'f' and 'df' containing the contrast
        function taking 3 arguments This should be a ufunc acting element-wise
        on any array
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    '''

    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    # for now, only supports determined case
    assert n_chan == n_src

    # initialize the demixing matrices
    if W0 is None:
        W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)
    else:
        W = W0.copy()

    if f_contrast is None:
        f_contrast = f_contrasts['norm']
        f_contrast_args = [1, 1]

    I = np.eye(n_src,n_src)
    Y = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)
    V = np.zeros((n_freq, n_src, n_chan, n_chan), dtype=X.dtype)
    r = np.zeros((n_frames, n_src))
    G_r = np.zeros((n_frames, n_src))

    # Compute the demixed output
    def demix(Y, X, W):
        for f in range(n_freq):
            Y[:,f,:] = np.dot(X[:,f,:], np.conj(W[f,:,:]))

    for epoch in range(n_iter):

        demix(Y, X, W)

        if callback is not None and epoch % 10 == 0:
            if proj_back:
                z = projection_back(Y, X[:,:,0])
                callback(Y * np.conj(z[None,:,:]))
            else:
                callback(Y)

        # simple loop as a start
        # shape: (n_frames, n_src)
        r[:,:] = np.sqrt(np.sum(np.abs(Y * np.conj(Y)), axis=1))

        # Apply derivative of contrast function
        G_r[:,:] = f_contrast['df'](r, *f_contrast_args) / r  # shape (n_frames, n_src)

        # Compute Auxiliary Variable
        for f in range(n_freq):
            for s in range(n_src):
                V[f,s,:,:] =  (np.dot(G_r[None,:,s] * X[:,f,:].T, np.conj(X[:,f,:]))) / X.shape[0]

        # Update now the demixing matrix
        for f in range(n_freq):
            for s in range(n_src):
                WV = np.dot(np.conj(W[f,:,:].T), V[f,s,:,:])
                W[f,:,s] = np.linalg.solve(WV, I[:,s])
                W[f,:,s] /= np.inner(np.conj(W[f,:,s]), np.dot(V[f,s,:,:], W[f,:,s]))

    demix(Y, X, W)

    if proj_back:
        z = projection_back(Y, X[:,:,0])
        Y *= np.conj(z[None,:,:])

    if return_filters:
        return Y, W
    else:
        return Y
