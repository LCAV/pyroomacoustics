
import numpy as np
import os
import platform

from scipy.stats import binom as _binom
from scipy.stats import norm as _norm

from .stft import stft

def median(x, alpha=None, axis=-1, keepdims=False):
    '''
    Computes 95% confidence interval for the median.

    Parameters
    ----------
    x: array_like
        the data array
    alpha: float, optional
        the confidence level of the interval, confidence intervals are only computed
        when this argument is provided
    axis: int, optional
        the axis of the data on which to operate, by default the last axis

    :returns: A tuple (m, [le, ue]). The confidence interval is [m-le, m+ue].
    '''

    # place the axis on which to compute median in first position
    xsw = np.moveaxis(x, axis, 0)

    # sort the array
    xsw = np.sort(xsw, axis=0)
    n = xsw.shape[0]

    if n % 2 == 1:
        # if n is odd, take central element
        m = xsw[n//2,];
    else:
        # if n is even, average the two central elements
        m = 0.5*(xsw[n//2-1,] + xsw[n//2,]);

    if alpha is None:
        if keepdims:
            m = np.moveaxis(m[np.newaxis,], 0, axis)
        return m

    else:
        # bound for using the large n approximation
        clt_bound = max(10 / alpha, 10 / (2 - alpha))

        if n < clt_bound:
            # Get the bounds of the CI from the binomial distribution
            b = _binom(n, 0.5)
            j,k = int(b.ppf(alpha/2)-1), int(b.ppf(1 - alpha/2)-1)

            if b.cdf(k) - b.cdf(j) < 1 - alpha:
                k += 1

            # sanity check
            assert b.cdf(k) - b.cdf(j) >= 1 - alpha

            if j < 0:
                raise ValueError('Warning: Sample size is too small. No confidence interval found.')
            else:
                ci = np.array([xsw[j,]-m, xsw[k,]-m])

        else:
            # we use the Normal approximation for large sets
            norm = _norm()
            eta = norm.ppf(1 - alpha / 2)
            j = int(np.floor(0.5*n - 0.5 * eta * np.sqrt(n))) - 1
            k = int(np.ceil(0.5*n + 0.5 * eta * np.sqrt(n)))
            ci = np.array([xsw[j,]-m,xsw[k,]-m])

        if keepdims:
            m = np.moveaxis(m[np.newaxis,], 0, axis)
            if axis < 0:
                ci = np.moveaxis(ci[:,np.newaxis,], 1, axis)
            else:
                ci = np.moveaxis(ci[:,np.newaxis,], 1, axis+1)


        return m, ci


# Simple mean squared error function
def mse(x1, x2):
    '''
    A short hand to compute the mean-squared error of two signals.

    .. math::

       MSE = \\frac{1}{n}\sum_{i=0}^{n-1} (x_i - y_i)^2


    :arg x1: (ndarray)
    :arg x2: (ndarray)
    :returns: (float) The mean of the squared differences of x1 and x2.
    '''
    return (np.abs(x1-x2)**2).sum()/len(x1)


# Itakura-Saito distance function
def itakura_saito(x1, x2, sigma2_n, stft_L=128, stft_hop=128):

  P1 = np.abs(stft(x1, stft_L, stft_hop))**2
  P2 = np.abs(stft(x2, stft_L, stft_hop))**2

  VAD1 = P1.mean(axis=1) > 2*stft_L**2*sigma2_n
  VAD2 = P2.mean(axis=1) > 2*stft_L**2*sigma2_n
  VAD = np.logical_or(VAD1, VAD2)

  if P1.shape[0] != P2.shape[0] or P1.shape[1] != P2.shape[1]:
    raise ValueError("Error: Itakura-Saito requires both array to have same length")

  R = P1[VAD,:]/P2[VAD,:]

  IS = (R - np.log(R) - 1.).mean(axis=1)

  return np.median(IS)

def snr(ref, deg):

    return np.sum(ref**2)/np.sum((ref-deg)**2)

# Perceptual Evaluation of Speech Quality for multiple files using multiple threads
def pesq(ref_file, deg_files, Fs=8000, swap=False, wb=False, bin='./bin/pesq'):
    '''
    pesq_vals = pesq(ref_file, deg_files, sample_rate=None, bin='./bin/pesq'):
    Computes the perceptual evaluation of speech quality (PESQ) metric of a degraded
    file with respect to a reference file.  Uses the utility obtained from ITU
    P.862 http://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en

    :arg ref_file:    The filename of the reference file.
    :arg deg_files:   A list of degraded sound files names.
    :arg sample_rate: Sample rates of the sound files [8kHz or 16kHz, default 8kHz].
    :arg swap:        Swap byte orders (whatever that does is not clear to me) [default: False].
    :arg wb:          Use wideband algorithm [default: False].
    :arg bin:         Location of pesq executable [default: ./bin/pesq].

    :returns: (ndarray size 2xN) ndarray containing Raw MOS and MOS LQO in rows 0 and 1, 
        respectively, and has one column per degraded file name in deg_files.
    '''

    if isinstance(deg_files, str):
        deg_files = [deg_files]

    if platform.system() is 'Windows':
        bin = bin + '.exe'

    if not os.path.isfile(ref_file):
        raise ValueError('Some file did not exist')
    for f in deg_files:
        if not os.path.isfile(f):
            raise ValueError('Some file did not exist')

    if Fs not in (8000, 16000):
        raise ValueError('sample rate must be 8000 or 16000')

    args = [ bin, '+%d' % int(Fs) ]

    if swap is True:
        args.append('+swap')

    if wb is True:
        args.append('+wb')

    args.append(ref_file)

    # array to receive all output values
    pesq_vals = np.zeros((2,len(deg_files)))

    # launch pesq for each degraded file in a different process
    import subprocess
    pipes = [ subprocess.Popen(args+[deg], stdout=subprocess.PIPE) for deg in deg_files ]
    states = np.ones(len(pipes), dtype=np.bool)

    # Recover output as the processes finish
    while states.any():

        for i,p in enumerate(pipes):
            if states[i] == True and p.poll() is not None:
                states[i] = False
                out = p.stdout.readlines()
                last_line = out[-1][:-2]

                if wb is True:
                    if not last_line.startswith('P.862.2 Prediction'):
                        raise ValueError(last_line)
                    pesq_vals[:,i] =  np.array([0, float(last_line.split()[-1])])
                else:
                    if not last_line.startswith('P.862 Prediction'):
                        raise ValueError(last_line)
                    pesq_vals[:,i] = np.array(map(float, last_line.split()[-2:]))

    return pesq_vals
