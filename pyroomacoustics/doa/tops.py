# Author: Eric Bezzam
# Date: July 15, 2016

from .music import *
from scipy.linalg import svdvals

from scipy import linalg

class TOPS(MUSIC):
    """
    Class to apply Test of Orthogonality of Projected Subspaces (TOPS) for 
    Direction of Arrival (DoA) estimation.

    .. note:: Run locate_source() to apply the TOPS algorithm.

    :param L: Microphone array positions. Each column should correspond to the 
    cartesian coordinates of a single microphone.
    :type L: numpy array
    :param fs: Sampling frequency.
    :type fs: float
    :param nfft: FFT length.
    :type nfft: int
    :param c: Speed of sound. Default: 343 m/s
    :type c: float
    :param num_src: Number of sources to detect. Default: 1
    :type num_src: int
    :param mode: 'far' or 'near' for far-field or near-field detection 
    respectively. Default: 'far'
    :type mode: str
    :param r: Candidate distances from the origin. Default: np.ones(1)
    :type r: numpy array
    :param azimuth: Candidate azimuth angles (in radians) with respect to x-axis.
    Default: np.linspace(-180.,180.,30)*np.pi/180
    :type azimuth: numpy array
    :param colatitude: Candidate elevation angles (in radians) with respect to z-axis.
    Default is x-y plane search: np.pi/2*np.ones(1)
    :type colatitude: numpy array
    """
    def __init__(self, L, fs, nfft, c=343.0, num_src=1, mode='far', r=None,
        azimuth=None, colatitude=None, **kwargs):

        MUSIC.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_src=num_src, 
            mode=mode, r=r, azimuth=azimuth, colatitude=colatitude, **kwargs)

    def _process(self, X):
        """
        Perform TOPS for a given frame in order to estimate steered response 
        spectrum.
        """

        # need more than 1 frequency band
        if self.num_freq < 2:
            raise ValueError('Need more than one frequency band!')

        # select reference frequency (largest power)
        max_bin = np.argmax(np.sum(np.sum(abs(X[:,self.freq_bins,:]),axis=0),
            axis=1))
        f0 = self.freq_bins[max_bin]
        freq = list(self.freq_bins)
        freq.remove(f0)

        # compute empirical cross correlation matrices
        C_hat = self._compute_correlation_matrices(X)

        # compute signal and noise subspace for each frequency band
        F = np.zeros((self.num_freq,self.M,self.num_src), dtype='complex64')
        W = np.zeros((self.num_freq,self.M,self.M-self.num_src), 
            dtype='complex64')
        for k in range(self.num_freq):
            # subspace decomposition
            F[k,:,:], W[k,:,:], ws, wn = \
                self._subspace_decomposition(C_hat[k,:,:])

        # create transformation matrix
        f = 1.0/self.nfft/self.c*1j*2*np.pi*self.fs*(np.linspace(0,self.nfft/2,
            self.nfft/2+1)-f0)

        Phi = np.zeros((len(f),self.M,self.grid.n_points), dtype='complex64')

        for n in range(self.grid.n_points):
            p_s = self.grid.cartesian[:self.grid.dim,n]
            proj = np.dot(p_s,self.L[:self.grid.dim,:])
            for m in range(self.M):
                Phi[:,m,n] = np.exp(f*proj[m])

        # determine direction using rank test
        for n in range(self.grid.n_points):
            # form D matrix
            D = np.zeros((self.num_src,(self.M-self.num_src)*
                (self.num_freq-1)), dtype='complex64')
            for k in range(self.num_freq-1):
                Uk = np.conjugate(np.dot(np.diag(Phi[k,:,n]), 
                    F[max_bin,:,:])).T
                    # F[max_bin,:,:])).T
                idx = range(k*(self.M-self.num_src),(k+1)*(self.M-self.num_src))
                D[:,idx] = np.dot(Uk,W[k,:,:])
            s = svdvals(D)
            self.grid.values[n] = 1.0/s[-1]

