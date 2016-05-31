
import numpy as np

def binning(S, bands):
    B = np.zeros((S.shape[0], len(bands)), dtype=S.dtype)
    for i,b in enumerate(bands):
        B[:,i] = np.mean(S[:,b[0]:b[1]], axis=1)

    return B


def octave_bands(fc=1000, third=False):
    ''' Create a bank of octave bands '''

    div = 1
    if third == True:
        div = 3

    # Octave Bands
    fcentre = fc * ((2.0) ** (np.arange(-6*div,4*div+1)/float(div)))
    fd = (2**(0.5/div));
    bands = np.array([ [f/fd, f*fd] for f in fcentre ])
    
    return bands, fcentre


def critical_bands():
    '''
    Compute the Critical bands as defined in the book:
    Psychoacoustics by Zwicker and Fastl. Table 6.1 p. 159
    '''

    # center frequencies
    fc = [50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850,
        2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500];
    # boundaries of the bands (e.g. the first band is from 0Hz to 100Hz with center 50Hz, fb[0] to fb[1], center fc[0]
    fb = [0,  100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
        2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500];

    # now just make pairs
    bands = []
    for j in range(len(fb)-1):
        bands.append([fb[j],fb[j+1]])

    return np.array(bands), fc


def bands_hz2s(bands_hz, Fs, N, transform='dft'):
    '''
    Converts bands given in Hertz to samples with respect to a given sampling
    frequency Fs and a transform size N an optional transform type is used to
    handle DCT case.
    '''

    # set the bin width
    if (transform == 'dct'):
        B = float(Fs)/2./N
    else:
        B = float(Fs)/N

    bands_s = []
    for i in range(bands_hz.shape[0]):
        bands_s.append(np.around(bands_hz[i,]/B))
        if bands_hz[i,1] >= min(Fs/2, bands_hz[-1,1]):
            break

    bands_s[i][1] = N/2


    # remove duplicate, if any, (typically, if N is small and Fs is large)
    j = 0
    while (j < i):
        if (bands_s[j][0] == bands_s[j+1][0]):
            bands_s.pop(j)
            i -= 1
        else:
            j += 1

    return np.array(bands_s, dtype=np.int)



