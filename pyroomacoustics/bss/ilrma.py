'''
Blind Source Separation using Independent Low-Rank Matrix Analysis (ILRMA)

2018 (c) Juan Azcarreta, Robin Scheibler, MIT License
'''
import numpy as np
from .common import projection_back

def ilrma(X, n_src=None, n_iter=20, proj_back=False, W0=None,
          n_components=2,
          return_filters=0,
          callback=None):
    '''
    Implementation of ILRMA algorithm without partitioning function for BSS presented in

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, *Determined blind
    source separation unifying independent vector analysis and nonnegative matrix
    factorization,* IEEE/ACM Trans. ASLP, vol. 24, no. 9, pp. 1626-1641, Sept. 2016

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, and H. Saruwatari *Determined
    Blind Source Separation with Independent Low-Rank Matrix Analysis,* in
    Audio Source Separation, S. Makino, Ed. Springer, 2018, pp. 125-156.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the observed signal n_src: int, optional
        The number of sources or independent components
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for demixing matrix
    n_components: int
        Number of components in the non-negative spectrum
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix W (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    '''
    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    # Only supports determined case
    assert n_chan == n_src

    # initialize the demixing matrices
    # The demixing matrix has the following dimensions (nfrequencies, nchannels, nsources),
    if W0 is None:
        W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)
    else:
        W = W0.copy()

    # initialize the nonnegative matrixes with random values
    T = np.array(np.random.rand(n_freq, n_components, n_src))
    V = np.array(np.random.rand(n_components, n_frames, n_src))
    Y = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)
    R = np.zeros((n_freq, n_frames, n_src))
    I = np.eye(n_src, n_src)
    U = np.zeros((n_freq, n_src, n_chan, n_chan), dtype=X.dtype)
    product = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    lambda_aux = np.zeros(n_src)
    machine_epsilon = np.finfo(float).eps

    for n in range(0, n_src):
        R[:,:,n] = np.dot(T[:,:,n], V[:,:,n])

    # Compute the demixed output
    def demix(Y, X, W):
        for f in range(n_freq):
            Y[:,f,:] = np.dot(X[:,f,:], np.conj(W[f,:,:]))

    demix(Y, X, W)

    P = np.power(abs(Y), 2.)

    for epoch in range(n_iter):
        if callback is not None and epoch % 10 == 0:
            if proj_back:
                z = projection_back(Y, X[:,:,0])
                callback(Y * np.conj(z[None,:,:]))
            else:
                callback(Y)

        # simple loop as a start
        for s in range(n_src):
            iR = 1 / R[:,:,s]
            T[:,:,s] *= np.sqrt(np.dot(P[:,:,s].T * iR ** 2, V[:,:,s].T) / np.dot(iR, V[:,:,s].T))
            T[T < machine_epsilon] = machine_epsilon

            R[:,:,s] = np.dot(T[:,:,s], V[:,:,s])

            iR = 1 / R[:,:,s]
            V[:,:,s] *= np.sqrt(np.dot(T[:,:,s].T, P[:,:,s].T * iR ** 2) / np.dot(T[:,:,s].T, iR))
            V[V < machine_epsilon] = machine_epsilon

            R[:,:,s] = np.dot(T[:,:,s], V[:,:,s])

            # Compute Auxiliary Variable and update the demixing matrix
            for f in range(n_freq):
                U[f,s,:,:] = np.dot(X[:,f,:].T, np.conj(X[:,f,:]) / R[f,:,None,s]) / n_frames
                product[f,:,:] = np.dot(np.conj(W[f,:,:].T), U[f,s,:,:])
                W[f,:,s] = np.linalg.solve(product[f,:,:], I[s,:])
                w_Unorm = np.inner(np.conj(W[f,:,s]), np.dot(U[f,s,:,:], W[f,:,s]))
                W[f,:,s] /= np.sqrt(w_Unorm)

        demix(Y, X, W)
        P = np.abs(Y) ** 2

        for s in range(n_src):
            lambda_aux[s] = 1 / np.sqrt(np.mean(P[:,:,s]))

            W[:,:,s] *= lambda_aux[s]
            P[:,:,s] *= lambda_aux[s] ** 2
            R[:,:,s] *= lambda_aux[s] ** 2
            T[:,:,s] *= lambda_aux[s] ** 2

    if proj_back:
        z = projection_back(Y, X[:,:,0])
        Y *= np.conj(z[None,:,:])

    if return_filters:
        return Y, W
    else:
        return Y
