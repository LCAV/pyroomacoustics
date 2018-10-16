
from __future__ import division, print_function
import numpy as np
from scipy import fftpack

import scipy.linalg as la
try:
    import mkl_fft as fft
    has_mklfft = True
except ImportError:
    import numpy.fft as fft
    has_mklfft = False

def autocorr(x):
    """ Fast autocorrelation computation using the FFT """
    
    X = fft.rfft(x, n=2*x.shape[0])
    r = fft.irfft(np.abs(X)**2)
    
    return r[:x.shape[0]] / (x.shape[0] - 1)

def toeplitz_multiplication(c, r, A, **kwargs):
    """ 
    Compute numpy.dot(scipy.linalg.toeplitz(c,r), A) using the FFT.

    Parameters
    ----------
    c: ndarray
        the first column of the Toeplitz matrix
    r: ndarray
        the first row of the Toeplitz matrix
    A: ndarray
        the matrix to multiply on the right
    """
    
    m = c.shape[0]
    n = r.shape[0]

    fft_len = int(2**np.ceil(np.log2(m+n-1)))
    zp = fft_len - m - n + 1
    
    if A.shape[0] != n:
        raise ValueError('A dimensions not compatible with toeplitz(c,r)')
    
    x = np.concatenate((c, np.zeros(zp, dtype=c.dtype), r[-1:0:-1]))
    xf = np.fft.rfft(x, n=fft_len)
    
    Af = np.fft.rfft(A, n=fft_len, axis=0)
    
    return np.fft.irfft((Af.T*xf).T, n=fft_len, axis=0)[:m,]

def hankel_multiplication(c, r, A, mkl=True, **kwargs):
    '''
    Compute numpy.dot(scipy.linalg.hankel(c,r=r), A) using the FFT.

    Parameters
    ----------
    c: ndarray
        the first column of the Hankel matrix
    r: ndarray
        the last row of the Hankel matrix
    A: ndarray
        the matrix to multiply on the right
    mkl: bool, optional
        if True, use the mkl_fft package if available
    '''
    
    if mkl and has_mklfft:
        fmul = mkl_toeplitz_multiplication
    else:
        fmul = toeplitz_multiplication
        A = A[:r.shape[0],:]

    return fmul(c[::-1], r, A, **kwargs)[::-1,]


def mkl_toeplitz_multiplication(c, r, A, A_padded=False, out=None, fft_len=None):
    """ 
    Compute numpy.dot(scipy.linalg.toeplitz(c,r), A) using the FFT from the mkl_fft package.

    Parameters
    ----------
    c: ndarray
        the first column of the Toeplitz matrix
    r: ndarray
        the first row of the Toeplitz matrix
    A: ndarray
        the matrix to multiply on the right
    A_padded: bool, optional
        the A matrix can be pre-padded with zeros by the user, if this is the case
        set to True
    out: ndarray, optional
        an ndarray to store the output of the multiplication
    fft_len: int, optional
        specify the length of the FFT to use
    """

    if not has_mklfft:
        raise ValueError('Import mkl_fft package unavailable. Install from https://github.com/LCAV/mkl_fft')

    m = c.shape[0]
    n = r.shape[0]

    if fft_len is None:
        fft_len = int(2**np.ceil(np.log2(m+n-1)))
    zp = fft_len - m - n + 1
    
    if (not A_padded and A.shape[0] != n) or (A_padded and A.shape[0] != fft_len):
        raise ValueError('A dimensions not compatible with toeplitz(c,r)')
    
    x = np.concatenate((c, np.zeros(zp, dtype=c.dtype), r[-1:0:-1]))
    xf = fft.rfft(x, n=fft_len)
    
    if out is not None:
        fft.rfft(A, n=fft_len, axis=0, out=out)
    else:
        out = fft.rfft(A, n=fft_len, axis=0)

    out *= xf[:,None]

    if A_padded:
        fft.irfft(out, n=fft_len, axis=0, out=A)
    else:
        A = fft.irfft(out, n=fft_len, axis=0)
    
    return A[:m,]


def naive_toeplitz_multiplication(c, r, A):
    """ 
    Compute numpy.dot(scipy.linalg.toeplitz(c,r), A)

    Parameters
    ----------
    c: ndarray
        the first column of the Toeplitz matrix
    r: ndarray
        the first row of the Toeplitz matrix
    A: ndarray
        the matrix to multiply on the right
    """

    return np.dot(la.toeplitz(c,r),A)

def hankel_stride_trick(x, shape):
    ''' 
    Make a Hankel matrix from a vector using stride tricks 
    
    Parameters
    ----------
    x: ndarray
        a vector that contains the concatenation of the first column
        and first row of the Hankel matrix to build *without* repetition
        of the lower left corner value of the matrix
    shape: tuple
        the shape of the Hankel matrix to build, it must satisfy ``x.shape[0] == shape[0] + shape[1] - 1``
    '''

    if x.shape[0] != shape[0] + shape[1] - 1:
        raise ValueError('Inconsistent dimensions')

    strides = (x.itemsize, x.itemsize)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

def toeplitz_strang_circ_approx(r, matrix=False):
    '''
    Circulant approximation to a symetric Toeplitz matrix
    by Gil Strang

    Parameters
    ----------
    r: ndarray
        the first row of the symmetric Toeplitz matrix
    matrix: bool, optional
        if True, the full symetric Toeplitz matrix is returned,
        otherwise, only the first column
    '''

    n = r.shape[0]
    c = r.copy()
    m = n // 2 if n % 2 == 0 else (n - 1) // 2
    c[-m:] = r[m:0:-1]

    if matrix:
        return la.circulant(c)
    else:
        return c
    

def toeplitz_opt_circ_approx(r, matrix=False):
    ''' 
    Optimal circulant approximation of a symmetric Toeplitz matrix
    by Tony F. Chan

    Parameters
    ----------
    r: ndarray
        the first row of the symmetric Toeplitz matrix
    matrix: bool, optional
        if True, the full symetric Toeplitz matrix is returned,
        otherwise, only the first column
    '''

    n = r.shape[0]

    r_rev = np.zeros(r.shape)
    r_rev[1:] = r[:0:-1]

    i = np.arange(n)
    c = (i * r_rev + (n - i) * r) / n
    c[1:] = c[:0:-1]

    if matrix:
        return la.circulant(c)
    else:
        return c


if __name__ == "__main__":

    import time 

    try:
        import mkl as mkl_service
        mkl_service.set_num_threads(7)
    except ImportError:
        pass

    n_iter = 1

    m = 10
    n = 100
    o = 10

    dtype_in = np.float32
    dtype_out = np.complex128 if dtype_in == np.float64 else np.complex64
    dorder = 'C'
    fft_len = int(2 ** np.ceil( np.log2(m + n - 1) ))

    c = dtype_in(np.random.randn(m))
    r = dtype_in(np.random.randn(n))

    A = np.asfortranarray(dtype_in(np.random.randn(n, o)))

    Apad = np.zeros((fft_len, o), order=dorder, dtype=dtype_in)
    Apad[:n,:] = A
    out = np.zeros((fft_len//2+1, o), dtype=dtype_out, order=dorder)

    print('Start...')
    start_time = time.time()
    for i in range(n_iter):
        C1 = naive_toeplitz_multiplication(c, r, A)
    nai_time = time.time() - start_time
    print("naive --- %s seconds ---" % (nai_time))

    '''
    start_time = time.time()
    for i in range(n_iter):
        C = toeplitz_multiplication(c, r, A)
    npy_time = time.time() - start_time
    print("numpy   --- %s seconds ---" % (npy_time))
    '''

    start_time = time.time()
    for i in range(n_iter):
        #C2 = mkl_toeplitz_multiplication(c, r, Apad, A_padded=True, out=out, fft_len=fft_len)
        C2 = mkl_toeplitz_multiplication(c, r, A, A_padded=False, out=None, fft_len=None)
    mkl_time = time.time() - start_time

    print("mkl     --- %s seconds ---" % (mkl_time))
    print("Result matching:", np.allclose(C1, C2))

    print('Stop.')
