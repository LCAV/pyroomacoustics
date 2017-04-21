# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

'''A collection of windowing functions.'''

import numpy as np

# cosine window function
def cosine(N, flag='asymmetric', length='full'):

    # first choose the indexes of points to compute
    if (length == 'left'):     # left side of window
        t = np.arange(0, N / 2)
    elif(length == 'right'):   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if (flag == 'symmetric' or flag == 'mdct'):
        t = t / float(N - 1)
    else:
        t = t / float(N)

    w = np.cos(np.pi * (t - 0.5)) ** 2

    # make the window respect MDCT condition
    if (flag == 'mdct'):
        w **= 2
        d = w[:N / 2] + w[N / 2:]
        w[:N / 2] *= 1. / d
        w[N / 2:] *= 1. / d

    # compute window
    return w


# triangular window function
def triang(N, flag='asymmetric', length='full'):

    # first choose the indexes of points to compute
    if (length == 'left'):     # left side of window
        t = np.arange(0, N / 2)
    elif(length == 'right'):   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if (flag == 'symmetric' or flag == 'mdct'):
        t = t / float(N - 1)
    else:
        t = t / float(N)

    w = 1. - np.abs(2. * t - 1.)

    # make the window respect MDCT condition
    if (flag == 'mdct'):
        d = w[:N / 2] + w[N / 2:]
        w[:N / 2] *= 1. / d
        w[N / 2:] *= 1. / d

    # compute window
    return w


# hann window function
def hann(N, flag='asymmetric', length='full'):

    # first choose the indexes of points to compute
    if (length == 'left'):     # left side of window
        t = np.arange(0, N / 2)
    elif(length == 'right'):   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if (flag == 'symmetric' or flag == 'mdct'):
        t = t / float(N - 1)
    else:
        t = t / float(N)

    w = 0.5 * (1 - np.cos(2 * np.pi * t))

    # make the window respect MDCT condition
    if (flag == 'mdct'):
        d = w[:N / 2] + w[N / 2:]
        w[:N / 2] *= 1. / d
        w[N / 2:] *= 1. / d

    # compute window
    return w


# Blackman-Harris window
def blackman_harris(N, flag='asymmetric', length='full'):

    # coefficients
    a = np.array([.35875, .48829, .14128, .01168])

    # first choose the indexes of points to compute
    if (length == 'left'):     # left side of window
        t = np.arange(0, N / 2)
    elif(length == 'right'):   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if (flag == 'symmetric'):
        t = t / float(N - 1)
    else:
        t = t / float(N)

    pi = np.pi
    w = a[0] - a[1]*np.cos(2*pi*t) + a[2]*np.cos(4*pi*t) + a[3]*np.cos(6*pi*t)

    return w

# Rectangular window function
def rect(N):
    return np.ones(N)
