# Author: Eric Bezzam
# Date: July 15, 2016

from .music import *

class WAVES(MUSIC):
    """
    Class to apply Weighted Average of Signal Subspaces [WAVES]_ for Direction of
    Arrival (DoA) estimation.

    .. note:: Run locate_sources() to apply the WAVES algorithm.

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
    num_iter: int
        Number of iterations for CSSM. Default: 5

    References
    ----------
    .. [WAVES] E. D. di Claudio, R. Parisi, *WAVES: Weighted average of signal subspaces for 
        robust wideband direction finding*, IEEE Trans. Signal Process., Vol. 49, Num. 10, 
        2179--2191, 2001

    """
    def __init__(self, L, fs, nfft, c=343.0, num_src=1, mode='far', r=None,
        azimuth=None, colatitude=None, num_iter=5, **kwargs):

        MUSIC.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_src=num_src, 
            mode=mode, r=r, azimuth=azimuth, colatitude=colatitude, **kwargs)

        self.iter = num_iter
        self.Z = None

    def _process(self, X):
        """
        Perform WAVES for given frame in order to estimate steered response 
        spectrum.
        """

        # compute empirical cross correlation matrices
        C_hat = self._compute_correlation_matrices(X)

        # compute initial estimates
        beta = []
        invalid = []

        for k in range(self.num_freq):

            self.grid.set_values(1 / self._compute_spatial_spectrum(C_hat[k,:,:],
                self.freq_bins[k]))
            idx = self.grid.find_peaks(k=self.num_src)

            if len(idx) < self.num_src:    # remove frequency
                invalid.append(k)
            else:
                beta.append(idx)

        self.freq_bins = np.delete(self.freq_bins, invalid)
        self.num_freq = self.num_freq - len(invalid)

        # compute reference frequency (take bin with max amplitude)
        f0 = np.argmax(np.sum(np.sum(abs(X[:,self.freq_bins,:]), axis=0), 
            axis=1))
        f0 = self.freq_bins[f0]

        # iterate to find DOA (but max 20)
        i = 0
        self.Z = np.empty((self.M,len(self.freq_bins)*self.num_src), 
            dtype='complex64')

        # while(i < self.iter or (len(self.src_idx) < self.num_src and i < 20)):
        while(i < self.iter):

            # construct waves matrix
            self._construct_waves_matrix(C_hat, f0, beta)

            # subspace decomposition with svd
            u,s,v = np.linalg.svd(self.Z)
            Un = u[:,self.num_src:]

            # compute spatial spectrum
            cross = np.dot(Un,np.conjugate(Un).T)
            self.grid.set_values(self._compute_spatial_spectrum(cross,f0))
            idx = self.grid.find_peaks(k=self.num_src)
            beta = np.tile(idx, (self.num_freq, 1))

            i += 1

    def _construct_waves_matrix(self, C_hat, f0, beta):
        for j in range(len(self.freq_bins)):
            k = self.freq_bins[j]
            Aj = self.mode_vec[k,:,beta[j]].T
            A0 = self.mode_vec[f0,:,beta[j]].T
            B = np.concatenate((np.zeros([self.M-len(beta[j]), len(beta[j])]), 
                np.identity(self.M-len(beta[j]))), axis=1).T
            Tj = np.dot(np.c_[A0, B], np.linalg.inv(np.c_[Aj, B]))
            # estimate signal subspace
            Es, En, ws, wn = self._subspace_decomposition(C_hat[j,:,:])
            P = (ws-wn[-1])/np.sqrt(ws*wn[-1]+1)
            # form WAVES matrix
            idx1 = j*self.num_src
            idx2 = (j+1)*self.num_src
            self.Z[:,idx1:idx2] = np.dot(Tj, Es*P)

