'''
Blind Source Separation using Independent Low-Rank Matrix Factorization (ILRM)

2018 (c) Juan Azcarreta Ortiz, MIT License
'''
import numpy as np
from .common import projection_back

def ilrma(X, n_src=None, n_iter=20, proj_back=False, W0=None,
        n_components=2,
        return_filters=0,
        callback=None):

    '''
    Implementation of ILRMA algorithm for BSS presented in

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, and H. Saruwatari *Determined Blind Source Separation
    with Independent Low-Rank Matrix Analysis*, in Audio Source Separation, S. Makino, Ed. Springer, 2018, pp.  125-156.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the observed signal
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
        function taking 3 arguments. This should be a ufunc acting element-wise
        on any array
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
    X = np.transpose(X, (1, 0, 2))
    n_freq, n_frames, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    # Only supports determined case
    assert n_chan == n_src

    # initialize the demixing matrices
    if W0 is None:
        W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)
    else:
        W = W0.copy()

    # initialize the nonnegative matrixes with random values
    T = abs(np.array(np.random.rand(n_freq, n_components, n_src)))
    V = abs(np.array(np.random.rand(n_components, n_frames, n_src)))
    Y = np.zeros((n_freq, n_frames, n_src), dtype=X.dtype)
    R = np.zeros((n_freq, n_frames, n_src))
    I = np.eye(n_src, n_src)
    U = np.zeros((n_freq, n_src, n_chan, n_chan))
    product = np.zeros((n_freq, n_chan, n_chan))
    lambda_aux = np.zeros(n_src)
    machine_epsilon = np.finfo(float).eps

    for n in range(0, n_src):
        R[:, :, n] = np.dot(T[:, :, n], V[:, :, n])

    # Compute the demixed output
    def demix(Y, X, W):
        for f in range(n_freq):
            Y[f,:,:] = np.dot(X[f,:,:], np.conj(W[f,:,:]).T)

    demix(Y, X, W)
    P = np.power(abs(Y), 2.)

    for epoch in range(n_iter):

        if callback is not None and epoch % 10 == 0:
            print("Iteration: " + str(epoch))

            if proj_back:
                z = projection_back(Y, X[:,:,0])
                callback(Y * np.conj(z[None,:,:]))
            else:
                callback(Y)

        # simple loop as a start
        for s in range(n_src):
            T[:, :, s] = np.multiply(T[:, :, s], np.power(np.divide(np.dot(np.multiply(P[:, :, s],
                        np.power(R[:,:,s], -2.)), V[:,:,s].transpose()), np.dot(np.power(R[:,:,s], -1.),
                        np.transpose(V[:,:,s]))), 0.5))
            T[T < machine_epsilon] = machine_epsilon

            R[:, :, s] = np.dot(T[:, :, s], V[:, :, s])

            V[:, :, s] = np.multiply(V[:, :, s], np.power(np.divide(np.dot(np.transpose(T[:, :, s]), np.multiply(P[:, :, s],
                        np.power(R[:, :, s], -2.))), np.dot(np.transpose(T[:, :, s]), np.power(R[:, :, s], -1))), 0.5))
            V[V < machine_epsilon] = machine_epsilon

            R[:, :, s] = np.dot(T[:, :, s], V[:, :, s])

            # Compute Auxiliary Variable and update the demixing matrix
            for f in range(n_freq):
                U[f, s, :, :] = np.transpose(np.dot(np.conjugate(X[f, :, :].T), np.multiply(X[f, :, :],
                               np.dot(np.power(np.reshape(R[f, :, s], (n_frames, 1)), -1), np.ones([1, n_chan]))))/n_frames)

                product[f, :, :] = np.dot(W[f, :, :], U[f, s, :, :])
                W[f, s, :] = np.dot(np.linalg.inv(product[f, :, :] + machine_epsilon * I), I[s, :])
                W[f, s, :] = np.dot(W[f, s, :], np.power(np.dot(np.dot(np.conjugate(W[f, s, :]).T, U[f, s, :, :]), W[f, s, :]), 0.5))

        demix(Y, X, W)
        P = np.power(abs(Y), 2.)

        for s in range(n_src):
            lambda_aux[s] = np.sqrt(np.sum(np.sum(P[:, :, s], axis=0), axis=0)/(n_freq*n_frames))

            W[:, s, :] = np.dot(W[:, s, :], np.power(lambda_aux[s], -1))
            P[:, :, s] = np.dot(P[:, :, s], np.power(lambda_aux[s], -2))
            R[:, :, s] = np.dot(R[:, :, s], np.power(lambda_aux[s], -2))
            T[:, :, s] = np.dot(T[:, :, s], np.power(lambda_aux[s], -2))

    if proj_back:
        z = projection_back(Y, X[:, :, 0])
        Y *= np.conj(z[None, :, :])

    Y = np.transpose(Y, [1, 0, 2])
    if return_filters:
        return Y, W
    else:
        return Y
