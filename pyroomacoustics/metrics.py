
import numpy as np
import os
import platform

from .stft import stft

def median(x, axis=-1, keepdims=False):
    '''
    Computes 95% confidence interval for the median.

    :arg x: (ndarray 1D)

    :returns: A tuple (m, [le, ue]). The confidence interval is [m-le, m+ue].
    '''

    # place the axis on which to compute median in first position
    xsw = np.swapaxes(x, axis, 0)

    # sort the array
    xsw = np.sort(xsw, axis=0)
    n = xsw.shape[0]

    if n % 2 == 1:
        # if n is odd, take central element
        m = xsw[n/2,];
    else:
        # if n is even, average the two central elements
        m = 0.5*(xsw[n/2,] + xsw[n/2+1,]);

    # This table is taken from the Performance Evaluation lecture notes by J-Y Le Boudec
    # available at: http://perfeval.epfl.ch/lectureNotes.htm
    CI = [[1,6],  [1,7],  [1,7],  [2,8],  [2,9],  [2,10], [3,10], [3,11], [3,11],[4,12], \
          [4,12], [5,13], [5,14], [5,15], [6,15], [6,16], [6,16], [7,17], [7,17],[8,18], \
          [8,19], [8,20], [9,20], [9,21], [10,21],[10,22],[10,22],[11,23],[11,23], \
          [12,24],[12,24],[13,25],[13,26],[13,27],[14,27],[14,28],[15,28],[15,29], \
          [16,29],[16,30],[16,30],[17,31],[17,31],[18,32],[18,32],[19,33],[19,34], \
          [19,35],[20,35],[20,36],[21,36],[21,37],[22,37],[22,38],[23,39],[23,39], \
          [24,40],[24,40],[24,40],[25,41],[25,41],[26,42],[26,43],[26,44],[27,44]];
    CI = np.array(CI)

    # Table assumes indexing starting at 1, adjust to indexing from 0
    CI -= 1

    if n < 6:
        # If we have less than 6 samples, we cannot have a confidence interval
        ci = np.zeros((2,) + m.shape)
    elif n <= 70:
        # For 6 <= n <= 70, we use exact values from the table
        j = CI[n-6,0]
        k = CI[n-6,1]
        ci = np.array([xsw[j,]-m ,xsw[k,]-m])
    else:
        # For 70 < n, we use the approximation for large sets
        j = np.floor(0.5*n - 0.98*np.sqrt(n))
        k = np.ceil(0.5*n + 1 + 0.98*np.sqrt(n))
        ci = np.array([xsw[j,]-m,xsw[k,]-m])

    if keepdims:
        m = np.swapaxes(m[np.newaxis,], 0, axis)
        if axis < 0:
            ci = np.swapaxes(ci[:,np.newaxis,], 1, axis)
        else:
            ci = np.swapaxes(ci[:,np.newaxis,], 1, axis+1)

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
