
from __future__ import division

import numpy as np
from scipy.fftpack import dct
from .stft import stft

def binning(S, bands):
    '''
    This function computes the sum of all columns of S in the subbands
    enumerated in bands
    '''
    B = np.zeros((S.shape[0], len(bands)), dtype=S.dtype)
    for i,b in enumerate(bands):
        B[:,i] = np.mean(S[:,b[0]:b[1]], axis=1)

    return B


def octave_bands(fc=1000, third=False):
    '''
    Create a bank of octave bands

    Parameters
    ----------
    fc : float, optional
        The center frequency
    third : bool, optional
        Use third octave bands (default False)
    '''

    div = 1
    if third == True:
        div = 3

    # Octave Bands
    fcentre = fc * ((2.0) ** (np.arange(-6*div,4*div + 1) / div))
    fd = 2**(0.5 / div);
    bands = np.array([ [f / fd, f*fd] for f in fcentre ])
    
    return bands, fcentre


def critical_bands():
    '''
    Compute the Critical bands as defined in the book:
    Psychoacoustics by Zwicker and Fastl. Table 6.1 p. 159
    '''

    # center frequencies
    fc = [50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850,
          2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500];
    # boundaries of the bands (e.g. the first band is from 0Hz to 100Hz 
    # with center 50Hz, fb[0] to fb[1], center fc[0]
    fb = [0,  100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
          2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500];

    # now just make pairs
    bands = [ [fb[j], fb[j+1]] for j in range(len(fb)-1) ]

    return np.array(bands), fc


def bands_hz2s(bands_hz, Fs, N, transform='dft'):
    '''
    Converts bands given in Hertz to samples with respect to a given sampling
    frequency Fs and a transform size N an optional transform type is used to
    handle DCT case.
    '''

    # set the bin width
    if (transform == 'dct'):
        B = Fs/2/N
    else:
        B = Fs/N

    # upper limit of the frequency range
    limit = min(Fs/2, bands_hz[-1,1])

    # Convert from Hertz to samples for all bands
    bands_s = [ np.around(band/B)  for band in bands_hz if band[0] <= limit]

    # Last band ends at N/2
    bands_s[-1][1] = N/2

    # remove duplicate, if any, (typically, if N is small and Fs is large)
    j = 0
    while (j < len(bands_s)-1):
        if (bands_s[j][0] == bands_s[j+1][0]):
            bands_s.pop(j)
        else:
            j += 1

    return np.array(bands_s, dtype=np.int)

def melscale(f):
    ''' Converts f (in Hertz) to the melscale defined according to Huang-Acero-Hon (2.6) '''
    return 1125.*np.log(1+f/700.)

def invmelscale(b):
    ''' Converts from melscale to frequency in Hertz according to Huang-Acero-Hon (6.143) '''
    return 700.*(np.exp(b/1125.)-1)

def melfilterbank(M, N, fs=1, fl=0., fh=0.5):
    '''
    Returns a filter bank of triangular filters spaced according to mel scale

    We follow Huang-Acera-Hon 6.5.2

    Parameters
    ----------
    M : (int)
        The number of filters in the bank
    N : (int)
        The length of the DFT
    fs : (float) optional
        The sampling frequency (default 8000)
    fl : (float)
        Lowest frequency in filter bank as a fraction of fs (default 0.)
    fh : (float)
        Highest frequency in filter bank as a fraction of fs (default 0.5)

    Returns
    -------
    An M times int(N/2)+1 ndarray that contains one filter per row
    '''

    # all center frequencies of the filters
    f = (N/fs)*invmelscale( melscale(fl*fs) + (
            np.arange(M+2)*(melscale(fh*fs)-melscale(fl*fs))/(M+1) )
        )

    # Construct the triangular filter bank
    H = np.zeros((M, N//2+1))
    k = np.arange(N//2+1)
    for m in range(1,M+1):
        I = np.where(np.logical_and(f[m-1] < k, k < f[m]))
        H[m-1,I] = 2 * (k[I]-f[m-1]) / ((f[m+1]-f[m-1]) * (f[m]-f[m-1]))
        I = np.where(np.logical_and(f[m] <= k, k < f[m+1]))
        H[m-1,I] = 2 * (f[m+1]-k[I]) / ((f[m+1]-f[m-1]) * (f[m+1]-f[m]))

    return H


def mfcc(x, L=128, hop=64, M=14, fs=8000, fl=0., fh=0.5):
    '''
    Computes the Mel-Frequency Cepstrum Coefficients (MFCC) according
    to the description by Huang-Acera-Hon 6.5.2 (2001)
    The MFCC are features mimicing the human perception usually
    used for some learning task.

    This function will first split the signal into frames, overlapping
    or not, and then compute the MFCC for each frame.

    Parameters
    ----------
    x : (nd-array)
        Input signal
    L : (int)
        Frame size (default 128)
    hop : (int)
        Number of samples to skip between two frames (default 64)
    M : (int)
        Number of mel-frequency filters (default 14)
    fs : (int)
        Sampling frequency (default 8000)
    fl : (float)
        Lowest frequency in filter bank as a fraction of fs (default 0.)
    fh : (float)
        Highest frequency in filter bank as a fraction of fs (default 0.5)

    Return
    ------
    The MFCC of the input signal
    '''

    # perform STFT, X contains frames in rows
    X = stft(x, L, hop, transform=np.fft.rfft)

    # get and apply the mel filter bank
    # and compute log energy
    H = melfilterbank(M, L, fs=fs, fl=fl, fh=fh)
    S = np.log(np.dot(H, np.abs(X.T)**2))

    # Now take DCT of the result
    C = dct(S, type=2, n=M, axis=0)

    return C

