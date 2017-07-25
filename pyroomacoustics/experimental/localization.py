from __future__ import division, print_function

import numpy as np
from scipy import linalg as la
from .point_cloud import PointCloud

try:
    import mklfft as fft
except ImportError:
    import numpy.fft as fft

def tdoa_loc(R, tdoa, c, x0=None):
    '''
    TDOA based localization

    Parameters
    ----------
    R : ndarray
        A 3xN array of 3D points
    tdoa : ndarray
        A length N array of tdoa
    c : float
        The speed of sound

    Reference
    ---------
    Steven Li, TDOA localization
    '''
    tau = tdoa - tdoa[0]

    # eliminate 0 tdoa
    I = tau != 0.
    I[0] = True  # keep mic 0! (reference, tdoa always 0)
    tau = tau[I]
    R = R[:,I]

    # Need two ref points
    r0 = R[:,0:1]
    r1 = R[:,1:2]
    rm = R[:,2:]

    n0 = la.norm(r0)**2
    n1 = la.norm(r1)**2
    nm = la.norm(rm, axis=0)**2

    # Build system matrices
    # Steven Li's equations
    ABC = 2 * ( rm - r0 ) / (c * tau[2:]) - 2 * (r1 - r0) / (c * tau[1])
    D = c * tau[1] - c * tau[2:] + (nm - n0) / (c * tau[2:]) - (n1 - n0) / (c * tau[1])

    loc = la.lstsq(ABC.T, D)[0]
    '''

    from scipy.optimize import leastsq

    def f(r, *args):

        R = args[0]
        c = args[1]
        tdoa = args[2]

        res = la.norm(R - r[:3,None], axis=0) - (r[3] + c * tau)

        return res

    def Jf(r, *args):

        R = args[0]
        c = args[1]
        tdoa = args[2]

        delta = r[:3,None] - R
        norm = la.norm(delta, axis=0)

        J = np.zeros((R.shape[0]+1, R.shape[1]))
        J[:3,:] = (delta / norm)
        J[3,:] = -1.

        return J

    init = f(x0, R[:,1:], c, tdoa[1:])
    sol = leastsq(f, x0, args=(R[:,1:],c,tdoa[1:]), Dfun=Jf, full_output=True, maxfev=10000, col_deriv=True)
    print sol[2]['nfev']
    print sol[1]
    print np.sum(f(sol[0], R[:,1:], c, tdoa[1:])**2) / np.sum(init**2)

    loc = sol[0][:3]
    print 'distance offset',sol[0][3]
    '''

    return loc

def tdoa(x1, x2, interp=1, fs=1, phat=True):
    '''
    This function computes the time difference of arrival (TDOA)
    of the signal at the two microphones. This in turns is used to infer
    the direction of arrival (DOA) of the signal.
    
    Specifically if s(k) is the signal at the reference microphone and
    s_2(k) at the second microphone, then for signal arriving with DOA
    theta we have
    
    s_2(k) = s(k - tau)
    
    with
    
    tau = fs*d*sin(theta)/c
    
    where d is the distance between the two microphones and c the speed of sound.
    
    We recover tau using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)
    method. The reference is
    
    Knapp, C., & Carter, G. C. (1976). The generalized correlation method for estimation of time delay. 
    
    Parameters
    ----------
    x1 : nd-array
        The signal of the reference microphone
    x2 : nd-array
        The signal of the second microphone
    interp : int, optional (default 1)
        The interpolation value for the cross-correlation, it can
        improve the time resolution (and hence DOA resolution)
    fs : int, optional (default 44100 Hz)
        The sampling frequency of the input signal
        
    Return
    ------
    theta : float
        the angle of arrival (in radian (I think))
    pwr : float
        the magnitude of the maximum cross correlation coefficient
    delay : float
        the delay between the two microphones (in seconds)
    '''

    # zero padded length for the FFT
    n = (x1.shape[0]+x2.shape[0]-1)
    if n % 2 != 0:
        n += 1

    # Generalized Cross Correlation Phase Transform
    # Used to find the delay between the two microphones
    # up to line 71
    X1 = fft.rfft(np.array(x1, dtype=np.float32), n=n)
    X2 = fft.rfft(np.array(x2, dtype=np.float32), n=n)

    if phat:
        X1 /= np.abs(X1)
        X2 /= np.abs(X2)

    cc = fft.irfft(X1*np.conj(X2), n=interp*n)

    # maximum possible delay given distance between microphones
    t_max = n // 2 + 1

    # reorder the cross-correlation coefficients
    cc = np.concatenate((cc[-t_max:],cc[:t_max]))

    # pick max cross correlation index as delay
    tau = np.argmax(np.abs(cc))
    pwr = np.abs(cc[tau])
    tau -= t_max  # because zero time is at the center of the array

    return tau / (fs*interp)


def edm_line_search(R, tdoa, bounds, steps):
    '''
    We have a number of points of know locations and have the TDOA measurements
    from an unknown location to the known point.
    We perform an EDM line search to find the unknown offset to turn TDOA to TOA.

    Parameters
    ----------
    R : ndarray
        An ndarray of 3xN where each column is the location of a point
    tdoa : ndarray
        A length N vector containing the tdoa measurements from uknown location to known ones
    bounds : ndarray
        Bounds for the line search
    step : float
        Step size for the line search
    '''

    dim = R.shape[0]

    pc = PointCloud(X=R)

    # use point 0 as reference
    dif = tdoa - tdoa.min()

    # initialize EDM
    D = np.zeros((pc.m+1, pc.m+1))
    D[:-1,:-1] = pc.EDM()

    # distance offset to search
    d = np.linspace(bounds[0], bounds[1], steps)

    # sum of eigenvalues that should be zero
    #cost = np.zeros((d.shape[0], D.shape[0]))
    cost = np.zeros(*d.shape)

    for i in range(d.shape[0]):
        D[-1,:-1] = D[:-1,-1] = (dif + d[i])**2
        w = np.sort(np.abs(la.eigh(D, eigvals_only=True)))
        #w = la.eigh(D, eigvals_only=True, eigvals=(D.shape[0]-6,D.shape[0]-6))
        cost[i] = np.sum(w[:D.shape[0]-5])

    return cost, d
