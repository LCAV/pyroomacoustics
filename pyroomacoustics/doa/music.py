# Author: Eric Bezzam
# Date: July 15, 2016

import numpy as np
from .doa import DOA


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
        C_hat = self._compute_correlation_matricesvec(X)
        # subspace decomposition
        Es, En, ws, wn = self._subspace_decompositionvec(C_hat[None,...])
        # compute spatial spectrum
        identity = np.zeros((self.num_freq,self.M,self.M))
        identity[:,list(np.arange(self.M)),list(np.arange(self.M))] = 1
        cross = identity - np.matmul(Es,np.moveaxis(np.conjugate(Es),-1,-2))
        self.Pssl = self._compute_spatial_spectrumvec(cross)
        self.grid.set_values(np.squeeze(np.sum(self.Pssl, axis=1)/self.num_freq))

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

    def _compute_spatial_spectrumvec(self,cross):
        mod_vec = np.transpose(np.array(self.mode_vec[self.freq_bins,:,:]),axes=[2,0,1])
        # timeframe, frequ, no idea
        denom = np.matmul(np.conjugate(mod_vec[...,None,:]),np.matmul(cross,mod_vec[...,None]))
        return np.squeeze(1/abs(denom))

    def _compute_spatial_spectrum(self,cross,k):

        P = np.zeros(self.grid.n_points)

        for n in range(self.grid.n_points):
            Dc = np.array(self.mode_vec[k,:,n],ndmin=2).T
            Dc_H = np.conjugate(np.array(self.mode_vec[k,:,n],ndmin=2))
            denom = np.dot(np.dot(Dc_H,cross),Dc)
            P[n] = 1/abs(denom)

        return P

    # non-vectorized version
    def _compute_correlation_matrices(self, X):
        C_hat = np.zeros([self.num_freq,self.M,self.M], dtype=complex)
        for i in range(self.num_freq):
            k = self.freq_bins[i]
            for s in range(self.num_snap):
                C_hat[i,:,:] = C_hat[i,:,:] + np.outer(X[:,k,s],
                    np.conjugate(X[:,k,s]))
        return C_hat/self.num_snap


    # vectorized version
    def _compute_correlation_matricesvec(self, X):
        # change X such that time frames, frequency microphones is the result
        X = np.transpose(X,axes=[2,1,0])
        # select frequency bins
        X = X[...,list(self.freq_bins),:]
        # Compute PSD and average over time frame
        C_hat = np.matmul(X[...,None],np.conjugate(X[...,None,:]))
        # Average over time-frames
        C_hat = np.mean(C_hat,axis=0)
        return C_hat

    # non-vectorized version
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
    # vectorized versino
    def _subspace_decompositionvec(self,R):
        # eigenvalue decomposition!
        w,v = np.linalg.eig(R)
        # sort out signal and noise subspace
        # Signal comprises the leading eigenvalues
        # Noise takes the rest

        eig_order = np.argsort(abs(w),axis=-1)[...,::-1]


        sig_space = eig_order[...,:self.num_src]
        noise_space = eig_order[...,self.num_src:]

        # eigenvalues
        # broadcasting for fancy indexing 
        b = np.asarray(np.arange(w.shape[0]))[:,None,None]
        c = np.asarray(np.arange(w.shape[1]))[None,:,None]
        d = np.asarray(np.arange(w.shape[2]))[None,None,:,None]
        ws = w[b,c,sig_space]
        wn = w[b,c,noise_space]
        # eigenvectors
        Es = v[b[...,None],c[...,None],d,sig_space[...,None,:]]
        En = v[b[...,None],c[...,None],d,noise_space[...,None,:]]
        return (Es, En, ws, wn)
