# Author: Eric Bezzam
# Date: July 15, 2016
from __future__ import division, print_function

from .music import *

class CSSM(MUSIC):
    """
    Class to apply the Coherent Signal-Subspace method [CSSM]_ for Direction of
    Arrival (DoA) estimation.

    .. note:: Run locate_sources() to apply the CSSM algorithm.

    Parameters
    ----------
    L: numpy array
        Microphone array positions. Each column should correspond to the 
        cartesian coordinates of a single microphone.
    fs: float
        Sampling frequency.
    nfft: int
        FFT length.
    c: float
        Speed of sound. Default: 343 m/s
    num_src: int
        Number of sources to detect. Default: 1
    mode: str
        'far' or 'near' for far-field or near-field detection 
        respectively. Default: 'far'
    r: numpy array
        Candidate distances from the origin. Default: np.ones(1)
    azimuth: numpy array
        Candidate azimuth angles (in radians) with respect to x-axis.
        Default: np.linspace(-180.,180.,30)*np.pi/180
    colatitude: numpy array
        Candidate colatitude angles (in radians) with respect to z-axis.
        Default is x-y plane search: np.pi/2*np.ones(1)
    num_iter: int
        Number of iterations for CSSM. Default: 5

    References
    ----------

    .. [CSSM] H. Wang, M. Kaveh, *Coherent signal-subspace processing for the detection and 
        estimation of angles of arrival of multiple wide-band sources*, IEEE Trans. Acoust., 
        Speech, Signal Process., Vol. 33, Num. 4, pp 823--831, 1985
    """
    def __init__(self, L, fs, nfft, c=343.0, num_src=1, mode='far', r=None,
        azimuth=None, colatitude=None, num_iter=5, **kwargs):

        MUSIC.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_src=num_src, 
            mode=mode, r=r, azimuth=azimuth, colatitude=colatitude, **kwargs)

        self.iter = num_iter

    def _process(self, X):
        """
        Perform CSSM for given frame in order to estimate steered response 
        spectrum.
        """

        # compute empirical cross correlation matrices
        C_hat = self._compute_correlation_matrices(X)

        # compute initial estimates
        beta = []
        invalid = []

        # Find number of spatial spectrum peaks at each frequency band.
        # If there are less peaks than expected sources, leave the band out
        # Otherwise, store the location of the peaks.
        for k in range(self.num_freq):

            self.grid.set_values(1 / self._compute_spatial_spectrum(C_hat[k,:,:],
                                 self.freq_bins[k]))
            idx = self.grid.find_peaks(k=self.num_src)

            if len(idx) < self.num_src:    # remove frequency
                invalid.append(k)
            else:
                beta.append(idx)

        # Here we remove the bands that had too few peaks
        self.freq_bins = np.delete(self.freq_bins, invalid)
        self.num_freq = self.num_freq - len(invalid)

        # compute reference frequency (take bin with max amplitude)
        f0 = np.argmax(np.sum(np.sum(abs(X[:,self.freq_bins,:]), axis=0),
            axis=1))
        f0 = self.freq_bins[f0]

        # iterate to find DOA, maximum number of iterations is 20
        i = 0

        # while(i < self.iter or (len(self.src_idx) < self.num_src and i < 20)):
        while(i < self.iter):

            # coherent sum
            R = self._coherent_sum(C_hat, f0, beta)

            # subspace decomposition
            Es, En, ws, wn = self._subspace_decomposition(R)

            # compute spatial spectrum
            cross = np.dot(En,np.conjugate(En).T)

            # cross = np.identity(self.M) - np.dot(Es, np.conjugate(Es).T)
            self.grid.set_values(self._compute_spatial_spectrum(cross,f0))
            idx = self.grid.find_peaks(k=self.num_src)
            beta = np.tile(idx, (self.num_freq, 1))

            i += 1

    def _coherent_sum(self, C_hat, f0, beta):

        R = np.zeros((self.M,self.M))

        # coherently sum frequencies
        for j in range(len(self.freq_bins)):
            k = self.freq_bins[j]

            Aj = self.mode_vec[k,:,beta[j]].T
            A0 = self.mode_vec[f0,:,beta[j]].T

            B = np.concatenate((np.zeros([self.M-len(beta[j]), len(beta[j])]), 
                np.identity(self.M-len(beta[j]))), axis=1).T

            Tj = np.dot(np.c_[A0, B], np.linalg.inv(np.c_[Aj, B]))

            R = R + np.dot(np.dot(Tj,C_hat[j,:,:]),np.conjugate(Tj).T)

        return R
