from __future__ import division, print_function

from scipy import linalg as la

# import numpy as np

from .doa import *
from .tools_fri_doa_plane import pt_src_recon_multiband, extract_off_diag, cov_mtx_est, polar2cart, make_G, \
    make_GtG_and_inv

class FRIDA(DOA):
    '''
    Implements the FRI-based direction of arrival finding algorithm [FRIDA]_.

    .. note:: Run locate_sources() to apply the CSSM algorithm.

    Parameters
    ----------
    L: ndarray
        Contains the locations of the microphones in the columns
    fs: int or float
        Sampling frequency
    nfft: int
        FFT size
    max_four: int
        Maximum order of the Fourier or spherical harmonics expansion
    c: float, optional
        Speed of sound
    num_src: int, optional
        The number of sources to recover (default 1)
    G_iter: int
        Number of mapping matrix refinement iterations in recovery algorithm (default 1)
    max_ini: int, optional
        Number of random initializations to use in recovery algorithm (default 5)
    n_rot: int, optional
        Number of random rotations to apply before recovery algorithm (default 10)
    noise_level: float, optional
        Noise level in the visibility measurements, if available (default 1e-10)
    stopping: str, optional
        Stopping criteria for the recovery algorithm. Can be max iterations or noise level (default max_iter)
    stft_noise_floor: float
        The noise floor in the STFT measurements, if available (default 0)
    stft_noise_margin: float
        When this, along with stft_noise_floor is set, we only pick frames with at least
        stft_noise_floor * stft_noise_margin power
    signal_type: str
        Which type of measurements to use:

        - 'visibility': Cross correlation measurements
        - 'raw': Microphone signals
    use_lu: bool, optional
        Whether to use LU decomposition for efficiency
    verbose: bool, optional
        Whether to output intermediate result for debugging purposes
    symb: bool, optional
        Whether enforce the symmetry on the reconstructed uniform samples of sinusoids b

    References
    ----------

    .. [FRIDA] H. Pan, R. Scheibler, E. Bezzam, I. Dokmanic, and M. Vetterli, *FRIDA:
        FRI-based DOA estimation for arbitrary array layouts*, Proc. ICASSP,
        pp 3186-3190, 2017
    '''

    def __init__(self, L, fs, nfft, max_four=None, c=343.0, num_src=1,
                 G_iter=None, max_ini=5, n_rot=1, max_iter=50, noise_level=1e-10,
                 low_rank_cleaning=False, stopping='max_iter',
                 stft_noise_floor=0., stft_noise_margin=1.5, signal_type='visibility',
                 use_lu=True, verbose=False, symb=True, use_cache=False, **kwargs):

        DOA.__init__(self, L, fs, nfft, c=c, num_src=num_src, mode='far', **kwargs)

        # intialize some attributes
        self.visi_noisy_all = None
        self.alpha_recon = np.array(num_src, dtype=float)

        # These might be used to select high SNR frames for cross correlation computation
        self.stft_noise_floor = stft_noise_floor
        self.stft_noise_margin = stft_noise_margin

        # Set the number of updates of the mapping matrix
        self.max_four = max_four if max_four is not None else num_src + 1
        self.update_G = True if G_iter is not None and G_iter > 1 else False
        self.G_iter = G_iter if self.update_G else 1
        self.noise_level = noise_level
        self.max_ini = max_ini
        self.max_iter = max_iter
        self.low_rank_cleaning = low_rank_cleaning  # This will impose low rank on corr matrix
        self.stop_cri = stopping
        self.n_rot = n_rot
        self.use_lu = use_lu
        self.verbose = verbose
        self.symb = symb

        self.G = None

        self.mapping_dict = dict(use_cache=use_cache)

        # The type of measurement to use, can be 'visibility' (default) for the covariance
        # matrix, or 'raw' to use microphone signals directly
        self.signal_type = signal_type

    def _process(self, X):
        '''
        Parameters
        ----------
        X: ndarray
            The STFT frames
        '''

        if self.signal_type == 'visibility':
            visi_noisy_all = self._visibilities(X)

            # stack as columns (NOT SUBTRACTING NOISELESS)
            self.visi_noisy_all = np.column_stack(visi_noisy_all)

            signal = self.visi_noisy_all

        elif self.signal_type == 'raw':
            signal = self._raw_average(X)

        else:
            raise ValueError("Signal type can be 'visibility' or 'raw'.")

        # loop over all subbands
        self.num_freq = self.freq_bins.shape[0]

        if self.dim == 2:

            # build the G matrix if necessary
            if self.G is None:
                self.G = make_G(
                    self.L[0, :], self.L[1, :],
                    2 * np.pi * self.freq_hz, self.c,
                    self.max_four,
                    signal_type=self.signal_type
                )
                self.GtG, self.GtG_inv = make_GtG_and_inv(self.G)

            # reconstruct point sources with FRI
            self.azimuth_recon, self.alpha_recon = \
                pt_src_recon_multiband(
                    signal,
                    self.L[0, :], self.L[1, :],
                    2 * np.pi * self.freq_hz, self.c,
                    self.num_src, self.max_four,
                    self.noise_level, self.max_ini,
                    max_iter=self.max_iter,
                    update_G=self.update_G,
                    G_iter=self.G_iter,
                    verbose=False,
                    signal_type=self.signal_type,
                    G_lst=self.G,
                    GtG_lst=self.GtG,
                    GtG_inv_lst=self.GtG_inv
                )

        elif self.dim == 3:

            raise ValueError('3D reconstruction is not yet available with FRIDA.')

    def _raw_average(self, X):
        ''' Correct the time rotation and average the raw microphone signal '''
        phaser = np.exp(-1j * 2 * np.pi * self.freq_hz[:, None] * np.arange(X.shape[2]) * self.nfft / self.fs)

        return np.mean(X[:, self.freq_bins, :] * phaser, axis=2)

    def _visibilities(self, X):

        visi_noisy_all = []
        for band_count in range(self.num_freq):
            # Estimate the covariance matrix and extract off-diagonal entries
            fn = self.freq_bins[band_count]
            energy = np.var(X[:, fn, :], axis=0)
            I = np.where(energy > self.stft_noise_margin * self.stft_noise_floor)
            R = cov_mtx_est(X[:, fn, I[0]])

            # impose low rank constraint
            if self.low_rank_cleaning:
                w, vl = la.eig(R)
                order = np.argsort(w)
                sig = order[-self.num_src:]
                sigma = (np.trace(R) - np.sum(w[sig])) / (R.shape[0] - self.num_src)
                Rhat = np.dot(vl[:, sig], np.dot(np.diag(w[sig] - sigma), np.conj(vl[:, sig].T)))
            else:
                Rhat = R

            visi_noisy = extract_off_diag(Rhat)
            visi_noisy_all.append(visi_noisy)

        return visi_noisy_all

    def _gen_dirty_img(self):
        """
        Compute the dirty image associated with the given measurements. Here the Fourier transform
        that is not measured by the microphone array is taken as zero.
        :return:
        """

        sound_speed = self.c
        num_mic = self.M

        if self.dim == 2:

            x_plt, y_plt = polar2cart(1, self.grid.azimuth)
            img = np.zeros(self.grid.n_points, dtype=complex)

            pos_mic_x = self.L[0, :]
            pos_mic_y = self.L[1, :]
            for i in range(self.num_freq):

                visi = self.visi_noisy_all[:, i]
                omega_band = 2 * np.pi * self.freq_hz[i]

                pos_mic_x_normalised = pos_mic_x / (sound_speed / omega_band)
                pos_mic_y_normalised = pos_mic_y / (sound_speed / omega_band)

                count_visi = 0
                for q in range(num_mic):
                    p_x_outer = pos_mic_x_normalised[q]
                    p_y_outer = pos_mic_y_normalised[q]
                    for qp in range(num_mic):
                        if not q == qp:
                            p_x_qqp = p_x_outer - pos_mic_x_normalised[qp]  # a scalar
                            p_y_qqp = p_y_outer - pos_mic_y_normalised[qp]  # a scalar
                            # <= the negative sign converts DOA to propagation vector
                            img += visi[count_visi] * \
                                   np.exp(-1j * (p_x_qqp * x_plt + p_y_qqp * y_plt))
                            count_visi += 1

            return img / (num_mic * (num_mic - 1))

        elif self.dim == 3:

            raise ValueError('3D reconstruction is not yet available with FRIDA.')

# -------------MISC--------------#

# def polar2cart(rho, phi):
#     """
#     convert from polar to cartesian coordinates
#     :param rho: radius
#     :param phi: azimuth
#     :return:
#     """
#     x = rho * np.cos(phi)
#     y = rho * np.sin(phi)
#     return x, y
