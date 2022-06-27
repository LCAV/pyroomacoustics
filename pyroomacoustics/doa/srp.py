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

    def __init__(
        self,
        L,
        fs,
        nfft,
        c=343.0,
        num_src=1,
        mode="far",
        r=None,
        azimuth=None,
        colatitude=None,
        **kwargs
    ):

        DOA.__init__(
            self,
            L=L,
            fs=fs,
            nfft=nfft,
            c=c,
            num_src=num_src,
            mode=mode,
            r=r,
            azimuth=azimuth,
            colatitude=colatitude,
            **kwargs
        )

        self.num_pairs = self.M * (self.M - 1) / 2

    def _process(self, X):
        """
        Perform SRP-PHAT for given frame in order to estimate steered response
        spectrum.
        """

        srp_cost = np.zeros(self.grid.n_points)

        # apply PHAT weighting
        absX = np.abs(X)
        absX[absX < tol] = tol
        pX = X / absX

        CC = []
        for k in self.freq_bins:
            CC.append(np.dot(pX[:, k, :], np.conj(pX[:, k, :]).T))
        CC = np.array(CC)

        M = self.L.shape[1]
        ar = np.arange(M)
        av = ar[:, None]

        # the mask here allow to select all the coefficients above
        # the main diagonal of the covariance matrix
        # It can be applied on the flattened last two dimensions of the
        # stack of cov. matrices
        mask_triu = (av < av.T).flatten()

        # Flatten the covariance matrices and use the above mask to
        # select the upper triangular part
        CC_flat = CC.reshape((-1, CC.shape[-2] * CC.shape[-2]))[:, mask_triu]

        # The DC offset is the sum of all the diagonal coefficients
        # Due to the normalization in SRP-PHAT, they are all ones,
        # and we end up with the product:
        # <number of frames> x <number of microphones> x <number of freqs>
        # the number of frames appears because the covariance matrix
        # is not normalized
        DC_offset = pX.shape[-1] * self.L.shape[1] * len(self.freq_bins)

        for n in range(self.grid.n_points):

            # In the loop, this is just a fancy way of computing
            # the quadratic form:
            # mode_vec^H @ CC @ mode_vec

            # get the mode vector axis: (frequency, microphones)
            mode_vec = self.mode_vec[self.freq_bins, :, n]

            # compute the outer product along the microphone axis
            mode_mat = np.conj(mode_vec[:, :, None]) * mode_vec[:, None, :]

            # First, we flatten the mode vector covariance and select the
            # terms above the main diagonal
            mode_mat_flat = mode_mat.reshape((-1, M * M))[:, mask_triu]
            # Then we compute manually the real part of the element-wise product
            # This is equivalent to `np.real(CC_flat * mode_mat_flat)
            # but avoids computing the imaginary part that we end up discarding
            R = CC_flat.real * mode_mat_flat.real - CC_flat.imag * mode_mat_flat.imag
            # Finally, we sum up and add the DC offset to make the cost non-negative
            sum_val = 2.0 * np.sum(R) + DC_offset

            # Finally normalize
            srp_cost[n] = sum_val / self.num_snap / self.num_freq / self.num_pairs

        self.grid.set_values(srp_cost)
