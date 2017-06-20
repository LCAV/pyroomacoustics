# Author: Eric Bezzam
# Date: July 15, 2016
from __future__ import division, print_function

from .doa import *

class SRP(DOA):
    """
    Class to apply Steered Response Power (SRP) direction-of-arrival (DoA) for 
    a particular microphone array.

    .. note:: Run locate_source() to apply the SRP-PHAT algorithm.

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
        Candidate elevation angles (in radians) with respect to z-axis.
        Default is x-y plane search: np.pi/2*np.ones(1)
    """
    def __init__(self, L, fs, nfft, c=343.0, num_src=1, mode='far', r=None, 
        azimuth=None, colatitude=None, **kwargs):

        DOA.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_src=num_src, 
            mode=mode, r=r, azimuth=azimuth, colatitude=colatitude, **kwargs)

        self.num_pairs = self.M*(self.M-1)/2

        #self.mode_vec = np.conjugate(self.mode_vec)

    def _process(self, X):
        """
        Perform SRP-PHAT for given frame in order to estimate steered response 
        spectrum.
        """

        ones = np.ones(self.L.shape[1])

        srp_cost = np.zeros(self.grid.n_points)

        # apply PHAT weighting
        absX = np.abs(X)
        absX[absX < tol] = tol
        pX = X / absX

        CC = []
        for k in self.freq_bins:
            CC.append( np.dot(pX[:,k,:], np.conj(pX[:,k,:]).T) )
        CC = np.array(CC)

        for n in range(self.grid.n_points):

            # get the mode vector axis: (frequency, microphones)
            mode_vec = self.mode_vec[self.freq_bins,:,n]

            # compute the outer product along the microphone axis
            mode_mat = np.conj(mode_vec[:,:,None]) * mode_vec[:,None,:]

            # multiply covariance by mode vectors and sum over the frequencies
            R = np.sum(CC * mode_mat, axis=0)

            # Now sum over all distince microphone pairs
            sum_val = np.inner(ones, np.dot(np.triu(R, 1), ones))

            # Finally normalize
            srp_cost[n] = np.abs(sum_val) / self.num_snap/self.num_freq/self.num_pairs

        self.grid.set_values(srp_cost)
