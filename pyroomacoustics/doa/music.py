# Author: Eric Bezzam
# Date: July 15, 2016

from .doa import *

class MUSIC(DOA):
    """
    Class to apply MUltiple SIgnal Classication (MUSIC) direction-of-arrival 
    (DoA) for a particular microphone array.

    .. note:: Run locate_source() to apply the MUSIC algorithm.

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

        self.Pssl = None

    def _process(self, X):
        """
        Perform MUSIC for given frame in order to estimate steered response 
        spectrum.
        """

        # compute steered response
        self.Pssl = np.zeros((self.num_freq,self.grid.n_points))
        num_freq = self.num_freq

        C_hat = self._compute_correlation_matrices(X)

        for i in range(self.num_freq):
            k = self.freq_bins[i]

            # subspace decomposition
            Es, En, ws, wn = self._subspace_decomposition(C_hat[i,:,:])

            # compute spatial spectrum
            # cross = np.dot(En,np.conjugate(En).T)
            cross = np.identity(self.M) - np.dot(Es, np.conjugate(Es).T) 
            self.Pssl[i,:] = self._compute_spatial_spectrum(cross,k)

        self.grid.set_values(np.sum(self.Pssl, axis=0)/num_freq)

    def plot_individual_spectrum(self):
        """
        Plot the steered response for each frequency.
        """

        # check if matplotlib imported
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings
            warnings.warn('Matplotlib is required for plotting')
            return

        # only for 2D
        if self.grid.dim == 3:
            pass
        else:
            import warnings
            warnings.warn('Only for 2D.')
            return

        # plot
        for k in range(self.num_freq):

            freq = float(self.freq_bins[k])/self.nfft*self.fs
            azimuth = self.grid.azimuth * 180 / np.pi

            plt.plot(azimuth, self.Pssl[k,0:len(azimuth)])

            plt.ylabel('Magnitude')
            plt.xlabel('Azimuth [degrees]')
            plt.xlim(min(azimuth),max(azimuth))
            plt.title('Steering Response Spectrum - ' + str(freq) + ' Hz')
            plt.grid(True)

    def _compute_spatial_spectrum(self,cross,k):

        P = np.zeros(self.grid.n_points)

        for n in range(self.grid.n_points):
            Dc = np.array(self.mode_vec[k,:,n],ndmin=2).T
            Dc_H = np.conjugate(np.array(self.mode_vec[k,:,n],ndmin=2))
            denom = np.dot(np.dot(Dc_H,cross),Dc)
            P[n] = 1/abs(denom)

        return P

    def _compute_correlation_matrices(self, X):
        C_hat = np.zeros([self.num_freq,self.M,self.M], dtype=complex)
        for i in range(self.num_freq):
            k = self.freq_bins[i]
            for s in range(self.num_snap):
                C_hat[i,:,:] = C_hat[i,:,:] + np.outer(X[:,k,s], 
                    np.conjugate(X[:,k,s]))
        return C_hat/self.num_snap

    def _subspace_decomposition(self, R):

        # eigenvalue decomposition!
        w,v = np.linalg.eig(R)

        # sort out signal and noise subspace
        # Signal comprises the leading eigenvalues
        # Noise takes the rest
        eig_order = np.flipud(np.argsort(abs(w)))
        sig_space = eig_order[:self.num_src]
        noise_space = eig_order[self.num_src:]

        # eigenvalues
        ws = w[sig_space]
        wn = w[noise_space]

        # eigenvectors
        Es = v[:,sig_space]
        En = v[:,noise_space]

        return Es, En, ws, wn

