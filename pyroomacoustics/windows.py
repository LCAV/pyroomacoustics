# coding=utf-8
#
# MIT License
#
# Window functions Copyright (C) 2015-2019 Taishi Nakashima, Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# You should have received a copy of the MIT License along with this program. If
# not, see <https://opensource.org/licenses/MIT>.

r"""
Window Functions
================

This is a collection of many popular window functions used in signal processing.

A few options are provided to correctly construct the required window function.
The ``flag`` keyword argument can take the following values.

``asymmetric``
  This way, many of the functions will sum to one when their left part is added
  to their right part. This is useful for overlapped transforms such as the STFT.

``symmetric``
  With this flag, the window is perfectly symmetric. This might be more
  suitable for analysis tasks.
``mdct``
  Available for only some of the windows. The window is modified to satisfy
  the perfect reconstruction condition of the MDCT transform.

Often, we would like to get the full window function, but on some occasions, it is useful
to get only the left (or right) part. This can be indicated via the keyword argument
``length`` that can take values ``full`` (default), ``left``, or ``right``.
"""

import numpy as np
from scipy import special
pi = np.pi


# Bartlett window
def bart(N, flag='asymmetric', length='full'):
    r'''
    The Bartlett window function

    .. math::

        w[n] = 2 / (M-1) ((M-1)/2 - |n - (M-1)/2|) , n=0,\ldots,N-1


    Parameters
    ----------
    N: int
        the window length
    flag: string, optional
        Possible values

        - *asymmetric*: asymmetric windows are used
          for overlapping transforms (:math:`M=N`)
        - *symmetric*: the window is symmetric (:math:`M=N-1`)
        - *mdct*: impose MDCT condition on the window (:math:`M=N-1` and
          :math:`w[n]^2 + w[n+N/2]^2=1`)

    length: string, optional
        Possible values

        - *full*: the full length window is computed
        - *right*: the right half of the window is computed
        - *left*: the left half of the window is computed
    '''

    # first choose the indexes of points to compute
    if length == 'left':     # left side of window
        t = np.arange(0, N / 2)
    elif length == 'right':   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if flag in ['symmetric', 'mdct']:
        t = t / float(N - 1)
    else:
        t = t / float(N)

    w = 2/(N-1) * ((N-1)/2 - np.abs(t - (N-1)/2))

    # make the window respect MDCT condition
    if flag == 'mdct':
        w **= 2
        d = w[:N / 2] + w[N / 2:]
        w[:N / 2] *= 1. / d
        w[N / 2:] *= 1. / d

    # compute window
    return w


# Modified Bartlett--Hann window
def bart_hann(N, flag='asymmetric', length='full'):
    r'''
    The modified Bartlett--Hann window function

    .. math::

        w[n] = 0.62 - 0.48|(n/M-0.5)| + 0.38 \cos(2\pi(n/M-0.5)),
        n=0,\ldots,N-1

    Parameters
    ----------
    N: int
        the window length
    flag: string, optional
        Possible values

        - *asymmetric*: asymmetric windows are used
          for overlapping transforms (:math:`M=N`)
        - *symmetric*: the window is symmetric (:math:`M=N-1`)
        - *mdct*: impose MDCT condition on the window (:math:`M=N-1` and
          :math:`w[n]^2 + w[n+N/2]^2=1`)

    length: string, optional
        Possible values

        - *full*: the full length window is computed
        - *right*: the right half of the window is computed
        - *left*: the left half of the window is computed
    '''

    # first choose the indexes of points to compute
    if length == 'left':     # left side of window
        t = np.arange(0, N / 2)
    elif length == 'right':   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if flag in ['symmetric', 'mdct']:
        t = t / float(N - 1)
    else:
        t = t / float(N)

    w = 0.62 - 0.48 * np.abs(t/N - 0.5) + 0.38 * np.cos(2*pi*(t/N - 0.5))

    # make the window respect MDCT condition
    if flag == 'mdct':
        w **= 2
        d = w[:N / 2] + w[N / 2:]
        w[:N / 2] *= 1. / d
        w[N / 2:] *= 1. / d

    # compute window
    return w


# Blackman window
def blackman(N, flag='asymmetric', length='full'):
    r'''
    The Blackman window function

    .. math::

        w[n] = 0.42 - 0.5\cos(2\pi n/(M-1)) + 0.08\cos(4\pi n/(M-1)),
        n = 0, \ldots, M-1


    Parameters
    ----------
    N: int
        the window length
    flag: string, optional
        Possible values

        - *asymmetric*: asymmetric windows are used
          for overlapping transforms (:math:`M=N`)
        - *symmetric*: the window is symmetric (:math:`M=N-1`)
        - *mdct*: impose MDCT condition on the window (:math:`M=N-1` and
          :math:`w[n]^2 + w[n+N/2]^2=1`)

    length: string, optional
        Possible values

        - *full*: the full length window is computed
        - *right*: the right half of the window is computed
        - *left*: the left half of the window is computed
    '''

    # first choose the indexes of points to compute
    if length == 'left':     # left side of window
        t = np.arange(0, N / 2)
    elif length == 'right':   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if flag in ['symmetric', 'mdct']:
        t = t / float(N - 1)
    else:
        t = t / float(N)

    w = 0.42 - 0.5*np.cos(2*pi*t/(N-1)) + 0.08*np.cos(4*pi*t/(N-1))

    # make the window respect MDCT condition
    if flag == 'mdct':
        w **= 2
        d = w[:N / 2] + w[N / 2:]
        w[:N / 2] *= 1. / d
        w[N / 2:] *= 1. / d

    # compute window
    return w


# Blackman-Harris window
def blackman_harris(N, flag='asymmetric', length='full'):
    r'''
    The Hann window function

    .. math::

        w[n] = a_0 - a_1 \cos(2\pi n/M)
        + a_2 \cos(4\pi n/M) + a_3 \cos(6\pi n/M), n=0,\ldots,N-1

    Parameters
    ----------
    N: int
        the window length
    flag: string, optional
        Possible values

        - *asymmetric*: asymmetric windows are used
          for overlapping transforms (:math:`M=N`)
        - *symmetric*: the window is symmetric (:math:`M=N-1`)

    length: string, optional
        Possible values

        - *full*: the full length window is computed
        - *right*: the right half of the window is computed
        - *left*: the left half of the window is computed
    '''

    # coefficients
    a = np.array([.35875, .48829, .14128, .01168])

    # first choose the indexes of points to compute
    if length == 'left':     # left side of window
        t = np.arange(0, N / 2)
    elif length == 'right':   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if flag == 'symmetric':
        t = t / float(N - 1)
    else:
        t = t / float(N)

    w = a[0] - a[1]*np.cos(2*pi*t) + a[2]*np.cos(4*pi*t) + a[3]*np.cos(6*pi*t)

    return w


# Bohman window function
def bohman(N, flag='asymmetric', length='full'):
    r'''
    The Bohman window function

    .. math::

        w[n] = (1-|x|) \cos(\pi |x|) + \pi / |x| \sin(\pi |x|), -1\leq x\leq 1

    Parameters
    ----------
    N: int
        the window length
    flag: string, optional
        Possible values

        - *asymmetric*: asymmetric windows are used
          for overlapping transforms (:math:`M=N`)
        - *symmetric*: the window is symmetric (:math:`M=N-1`)
        - *mdct*: impose MDCT condition on the window (:math:`M=N-1` and
          :math:`w[n]^2 + w[n+N/2]^2=1`)

    length: string, optional
        Possible values

        - *full*: the full length window is computed
        - *right*: the right half of the window is computed
        - *left*: the left half of the window is computed
    '''

    # first choose the indexes of points to compute
    if length == 'left':     # left side of window
        t = np.arange(0, N / 2)
    elif length == 'right':   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if flag in ['symmetric', 'mdct']:
        t = t / float(N - 1)
    else:
        t = t / float(N)

    x = np.abs(np.linspace(-1, 1, N)[1:-1])
    w = (1 - x) * np.cos(pi * x) + 1.0 / pi * np.sin(pi * x)
    w = np.r_[0, w, 0]

    # make the window respect MDCT condition
    if flag == 'mdct':
        d = w[:N / 2] + w[N / 2:]
        w[:N / 2] *= 1. / d
        w[N / 2:] *= 1. / d

    # compute window
    return w


# cosine window function
def cosine(N, flag='asymmetric', length='full'):
    r'''
    The cosine window function

    .. math::

        w[n] = \cos(\pi (n/M - 0.5))^2

    Parameters
    ----------
    N: int
        the window length
    flag: string, optional
        Possible values

        - *asymmetric*: asymmetric windows are used
          for overlapping transforms (:math:`M=N`)
        - *symmetric*: the window is symmetric (:math:`M=N-1`)
        - *mdct*: impose MDCT condition on the window (:math:`M=N-1` and
          :math:`w[n]^2 + w[n+N/2]^2=1`)

    length: string, optional
        Possible values

        - *full*: the full length window is computed
        - *right*: the right half of the window is computed
        - *left*: the left half of the window is computed
    '''

    # first choose the indexes of points to compute
    if length == 'left':     # left side of window
        t = np.arange(0, N / 2)
    elif length == 'right':   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if flag in ['symmetric', 'mdct']:
        t = t / float(N - 1)
    else:
        t = t / float(N)

    w = np.cos(pi * (t - 0.5)) ** 2

    # make the window respect MDCT condition
    if flag == 'mdct':
        w **= 2
        d = w[:N / 2] + w[N / 2:]
        w[:N / 2] *= 1. / d
        w[N / 2:] *= 1. / d

    # compute window
    return w


# Flattop window
def flattop(N, flag='asymmetric', length='full'):
    r'''
    The flat top weighted window function

    .. math::

        w[n] = a_0 - a_1 \cos(2\pi n/M) + a_2 \cos(4\pi n/M)
        + a_3 \cos(6\pi n/M) + a_4 \cos(8\pi n/M), n=0,\ldots,N-1

    where

    .. math::
        a0 = 0.21557895
        a1 = 0.41663158
        a2 = 0.277263158
        a3 = 0.083578947
        a4 = 0.006947368

    Parameters
    ----------
    N: int
        the window length
    flag: string, optional
        Possible values

        - *asymmetric*: asymmetric windows are
          used for overlapping transforms (:math:`M=N`)
        - *symmetric*: the window is symmetric (:math:`M=N-1`)

    length: string, optional
        Possible values

        - *full*: the full length window is computed
        - *right*: the right half of the window is computed
        - *left*: the left half of the window is computed
    '''

    # coefficients
    a = np.array([.21557895, .41663158, .277263158, .083578947, .006947368])

    # first choose the indexes of points to compute
    if length == 'left':     # left side of window
        t = np.arange(0, N / 2)
    elif length == 'right':   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if flag == 'symmetric':
        t = t / float(N - 1)
    else:
        t = t / float(N)

    w = a[0] - a[1]*np.cos(2*pi*t) + a[2]*np.cos(4*pi*t)\
        + a[3]*np.cos(6*pi*t) + a[4]*np.cos(8*pi*t)

    return w


# Gaussian window
def gaussian(N, std, flag='asymmetric', length='full'):
    r'''
    The flat top weighted window function

    .. math::
        w[n] = e^{ -\frac{1}{2}\left(\frac{n}{\sigma}\right)^2 }

    Parameters
    ----------
    N: int
        the window length
    std: float
        the standard deviation
    flag: string, optional
        Possible values

        - *asymmetric*: asymmetric windows are used
          for overlapping transforms (:math:`M=N`)
        - *symmetric*: the window is symmetric (:math:`M=N-1`)

    length: string, optional
        Possible values

        - *full*: the full length window is computed
        - *right*: the right half of the window is computed
        - *left*: the left half of the window is computed
    '''

    # first choose the indexes of points to compute
    if length == 'left':     # left side of window
        t = np.arange(0, N / 2)
    elif length == 'right':   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if flag == 'symmetric':
        t = t / float(N - 1)
    else:
        t = t / float(N)

    n = np.arange(0, N) - (N - 1.0) / 2.0
    sig2 = 2 * std**2
    w = np.exp(-n**2 / sig2)

    return w


# hamming window function
def hamming(N, flag='asymmetric', length='full'):
    r'''
    The Hamming window function

    .. math::

        w[n] = 0.54  - 0.46 \cos(2 \pi n / M), n=0,\ldots,N-1

    Parameters
    ----------
    N: int
        the window length
    flag: string, optional
        Possible values

        - *asymmetric*: asymmetric windows are used
          for overlapping transforms (:math:`M=N`)
        - *symmetric*: the window is symmetric (:math:`M=N-1`)
        - *mdct*: impose MDCT condition on the window (:math:`M=N-1` and
          :math:`w[n]^2 + w[n+N/2]^2=1`)

    length: string, optional
        Possible values

        - *full*: the full length window is computed
        - *right*: the right half of the window is computed
        - *left*: the left half of the window is computed
    '''

    # first choose the indexes of points to compute
    if length == 'left':     # left side of window
        t = np.arange(0, N / 2)
    elif length == 'right':   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if flag in ['symmetric', 'mdct']:
        t = t / float(N - 1)
    else:
        t = t / float(N)

    w = 0.54 - 0.46*np.cos(2*pi*t)

    # make the window respect MDCT condition
    if flag == 'mdct':
        d = w[:N / 2] + w[N / 2:]
        w[:N / 2] *= 1. / d
        w[N / 2:] *= 1. / d

    # compute window
    return w


# hann window function
def hann(N, flag='asymmetric', length='full'):
    r'''
    The Hann window function

    .. math::

        w[n] = 0.5 (1 - \cos(2 \pi n / M)), n=0,\ldots,N-1

    Parameters
    ----------
    N: int
        the window length
    flag: string, optional
        Possible values

        - *asymmetric*: asymmetric windows are used
          for overlapping transforms (:math:`M=N`)
        - *symmetric*: the window is symmetric (:math:`M=N-1`)
        - *mdct*: impose MDCT condition on the window (:math:`M=N-1` and
          :math:`w[n]^2 + w[n+N/2]^2=1`)

    length: string, optional
        Possible values

        - *full*: the full length window is computed
        - *right*: the right half of the window is computed
        - *left*: the left half of the window is computed
    '''

    # first choose the indexes of points to compute
    if length == 'left':     # left side of window
        t = np.arange(0, N / 2)
    elif length == 'right':   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if flag in ['symmetric', 'mdct']:
        t = t / float(N - 1)
    else:
        t = t / float(N)

    w = 0.5 * (1 - np.cos(2 * pi * t))

    # make the window respect MDCT condition
    if flag == 'mdct':
        d = w[:N / 2] + w[N / 2:]
        w[:N / 2] *= 1. / d
        w[N / 2:] *= 1. / d

    # compute window
    return w


# Kaiser window function
def kaiser(N, beta, flag='asymmetric', length='full'):
    r'''
    The Kaiser window function

    .. math::
       w[n] = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}} \right)/I_0(\beta)

    with

    .. math::
       \quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2},

    where :math:`I_0` is the modified zeroth-order Bessel function.

    Parameters
    ----------
    N: int
        the window length
    beta: float
        Shape parameter, determines trade-off between main-lobe width and
        side lobe level. As beta gets large, the window narrows.
    flag: string, optional
        Possible values

        - *asymmetric*: asymmetric windows are used
          for overlapping transforms (:math:`M=N`)
        - *symmetric*: the window is symmetric (:math:`M=N-1`)
        - *mdct*: impose MDCT condition on the window (:math:`M=N-1` and
          :math:`w[n]^2 + w[n+N/2]^2=1`)

    length: string, optional
        Possible values

        - *full*: the full length window is computed
        - *right*: the right half of the window is computed
        - *left*: the left half of the window is computed
    '''

    # first choose the indexes of points to compute
    if length == 'left':     # left side of window
        t = np.arange(0, N / 2)
    elif length == 'right':   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if flag in ['symmetric', 'mdct']:
        t = t / float(N - 1)
    else:
        t = t / float(N)

    n = np.arange(0, N)
    alpha = (N - 1) / 2.0
    w = (special.i0(beta * np.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
         special.i0(beta))

    # make the window respect MDCT condition
    if flag == 'mdct':
        d = w[:N / 2] + w[N / 2:]
        w[:N / 2] *= 1. / d
        w[N / 2:] *= 1. / d

    # compute window
    return w


# Rectangular window function
def rect(N):
    r'''
    The rectangular window

    .. math::

        w[n] = 1, n=0,\ldots,N-1

    Parameters
    ----------
    N: int
        the window length
    '''

    return np.ones(N)


# triangular window function
def triang(N, flag='asymmetric', length='full'):
    r'''
    The triangular window function

    .. math::

        w[n] = 1 - | 2 n / M - 1 |, n=0,\ldots,N-1

    Parameters
    ----------
    N: int
        the window length
    flag: string, optional
        Possible values

        - *asymmetric*: asymmetric windows are used
          for overlapping transforms (:math:`M=N`)
        - *symmetric*: the window is symmetric (:math:`M=N-1`)
        - *mdct*: impose MDCT condition on the window (:math:`M=N-1` and
          :math:`w[n]^2 + w[n+N/2]^2=1`)

    length: string, optional
        Possible values

        - *full*: the full length window is computed
        - *right*: the right half of the window is computed
        - *left*: the left half of the window is computed
    '''

    # first choose the indexes of points to compute
    if length == 'left':     # left side of window
        t = np.arange(0, N / 2)
    elif length == 'right':   # right side of window
        t = np.arange(N / 2, N)
    else:                   # full window by default
        t = np.arange(0, N)

    # if asymmetric window, denominator is N, if symmetric it is N-1
    if flag in ['symmetric', 'mdct']:
        t = t / float(N - 1)
    else:
        t = t / float(N)

    w = 1. - np.abs(2. * t - 1.)

    # make the window respect MDCT condition
    if flag == 'mdct':
        d = w[:N / 2] + w[N / 2:]
        w[:N / 2] *= 1. / d
        w[N / 2:] *= 1. / d

    # compute window
    return w
