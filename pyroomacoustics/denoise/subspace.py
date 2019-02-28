# Single Channel Noise Removal using the Subspace Approach
# Copyright (C) 2019  Eric Bezzam, Mathieu Lecoq, Gimena Segrelles
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# You should have received a copy of the MIT License along with this program. If
# not, see <https://opensource.org/licenses/MIT>.


import numpy as np


class Subspace(object):
    """
    A class for performing **single channel** noise reduction in the time domain
    via the subspace approach. This implementation is based off of the approach
    presented in:

        Y. Ephraim and H. L. Van Trees, *A signal subspace approach for speech enhancement,*
        IEEE Transactions on Speech and Audio Processing, vol. 3, no. 4, pp. 251-266, Jul 1995.

    Moreover, an adaptation of the subspace approach is implemented here, as
    presented in:

        Y. Hu and P. C. Loizou, *A subspace approach for enhancing speech corrupted by colored noise,*
        IEEE Signal Processing Letters, vol. 9, no. 7, pp. 204-206, Jul 2002.

    Namely, an eigendecomposition is performed on the matrix:

    .. math::

        \Sigma = R_n^{-1} R_y - I,

    where :math:`R_n` is the noise covariance matrix, :math:`R_y` is the
    covariance matrix of the input noisy signal, and :math:`I` is the identity
    matrix. The covariance matrices are estimated from past samples; the number
    of past samples/frames used for estimation can be set with the parameters
    `lookback` and `skip`. A simple energy threshold (`thresh` parameter) is
    used to identify noisy frames.

    The eigenvectors corresponding to the positive eigenvalues of
    :math:`\Sigma` are used to create a linear operator :math:`H_{opt}` in order
    to enhance the noisy input signal :math:`\mathbf{y}`:

    .. math::

        \mathbf{\hat{x}} = H_{opt} \cdot \mathbf{y}.

    The length of :math:`\mathbf{y}` is specified by the parameter `frame_len`;
    :math:`50\%` overlap and add with a Hanning window is used to reconstruct
    the output. A great summary of the approach can be found in the paper by
    Y. Hu and P. C. Loizou under Section III, B.

    Adjusting the factor :math:`\\mu` presents a trade-off between noise
    suppression and distortion.

    Below is an example of how to use this class to emulate a streaming/online
    input. A full example can be found `here <https://github.com/LCAV/pyroomacoustics/blob/master/examples/noise_reduction_subspace.py>`_.
    Depending on your choice for `frame_len`, `lookback`, and `skip`, the
    approach may not be suitable for real-time processing.

    ::

        # initialize Subspace object
        scnr = Subspace(frame_len=256, mu=10, lookback=10, skip=2, thresh=0.01)

        # apply block-by-block
        for n in range(num_blocks):
            denoised_signal = scnr.apply(noisy_input)


    There also exists a "one-shot" function.

    ::

        # import or create `noisy_signal`
        denoised_signal = apply_subspace(noisy_signal, frame_len=256, mu=10,
                                         lookback=10, skip=2, thresh=0.01)


    Parameters
    ----------
    frame_len : int
        Frame length in samples. Note that large values (above 256) will make
        the eigendecompositions very expensive.
    mu : float
        Enhancement factor, larger values suppress more noise but could lead
        to more distortion.
    lookback : int
        How many frames to look back for covariance matrix estimation.
    skip : int
        How many samples to skip when estimating the covariance matrices with
        past samples. `skip=1` will use all possible frames in the estimation.
    thresh : float
        Threshold to distinguish between (signal+noise) and (noise) frames. A
        high value will classify more frames as noise but might remove desired
        signal!
    data_type : 'float32' or 'float64'
        Data type to use in the enhancement procedure. Default is 'float32'.
    """


    def __init__(self, frame_len=256, mu=10, lookback=10, skip=2, thresh=0.01,
                 data_type='float32'):

        if frame_len % 2:
            raise ValueError("Frame length should be even as this method "
                             "performs 50% overlap.")

        if data_type is 'float64':
            data_type = np.float64
        else:
            data_type = np.float32

        self.frame_len = frame_len
        self.mu = mu
        self.dtype = data_type

        self.hop = frame_len // 2
        self.prev_samples = np.zeros(self.hop, dtype=data_type)
        self.prev_output = np.zeros(self.hop, dtype=data_type)
        self.current_out = np.zeros(self.hop, dtype=data_type)
        self.win = np.hanning(frame_len).astype(data_type)

        # enhancement filter parameter
        self.h_opt = np.zeros((frame_len, frame_len), dtype=data_type)

        # estimate (signal+noise) and noise covariance matrices
        self.thresh = thresh
        self.n_samples = self.hop * lookback + frame_len
        self.input_samples = np.zeros(self.n_samples, dtype=data_type)
        self.skip = skip
        self.n_frames = lookback * (self.hop//skip)
        self.n_noise_frames = np.ones(lookback) * (self.hop // skip)
        self.cov_sn = np.zeros((frame_len, frame_len), dtype=data_type)
        self._cov_sn = np.zeros((lookback, frame_len, frame_len),
                                dtype=data_type)
        self.cov_n = np.zeros((frame_len, frame_len), dtype=data_type)
        self._cov_n = np.zeros((lookback, frame_len, frame_len),
                               dtype=data_type)

    def apply(self, new_samples):
        """
        Parameters
        ----------
        new_samples: numpy array
            New array of samples of length `self.hop` in the time domain.

        Returns
        -------
        numpy array
            Denoised samples.
        """

        if len(new_samples) != self.hop:
            raise ValueError("Expected {} samples, got {}."
                             .format(self.hop, len(new_samples)))
        new_samples = new_samples.astype(self.dtype)

        # form input frame, 50% overlap
        input_frame = np.r_[self.prev_samples, new_samples]

        # update covariance matrix estimates
        self.update_cov_matrices(new_samples)

        # compute filter to project to signal subspace
        self.compute_signal_projection()

        # compute output
        denoised_out = self.win * np.dot(self.h_opt, input_frame)

        # update
        self.prev_samples[:] = new_samples
        self.current_out[:] = self.prev_output + denoised_out[:self.hop]
        self.prev_output[:] = denoised_out[self.hop:]

        return self.current_out

    def compute_signal_projection(self):

        sigma = np.linalg.lstsq(self.cov_n, self.cov_sn, rcond=None)[0] \
                - np.eye(self.frame_len)
        eigenvals, eigenvecs = np.linalg.eig(sigma)

        n_pos = sum(eigenvals > 0)
        order = np.argsort(eigenvals, axis=-1)[::-1]
        pos_eigenvals = np.real(eigenvals[order][:n_pos])
        q1 = np.zeros((self.frame_len, self.frame_len))
        for w in range(0, n_pos):
            q1[w, w] = pos_eigenvals[w] / (pos_eigenvals[w] + self.mu)

        v_t = np.transpose(-eigenvecs[:, order])
        self.h_opt[:] = np.real(np.dot(np.dot(np.linalg.pinv(v_t), q1), v_t))
        # self.h_opt = np.dot(np.linalg.lstsq(v_t, q1, rcond=None)[0], v_t)

    def update_cov_matrices(self, new_samples):

        # remove cov of old samples
        self.cov_sn *= self.n_frames
        self.cov_sn -= self._cov_sn[0]

        old_cov_n = self.cov_n.copy()
        self.cov_n *= sum(self.n_noise_frames)
        self.cov_n -= self._cov_n[0]

        # update samples
        self.input_samples = np.roll(self.input_samples, -self.hop)
        self.input_samples[-self.hop:] = new_samples

        # update cov matrices
        self._cov_sn = np.roll(self._cov_sn,  -1, axis=0)
        self._cov_sn[-1, :, :] = np.zeros((self.frame_len, self.frame_len))
        self._cov_n = np.roll(self._cov_n, -1, axis=0)
        self._cov_n[-1, :, :] = np.zeros((self.frame_len, self.frame_len))
        self.n_noise_frames = np.roll(self.n_noise_frames, -1)
        self.n_noise_frames[-1] = 0

        for i in range(0, self.hop, self.skip):
            a = self.n_samples - self.hop - self.frame_len + i
            b = a + self.frame_len
            _noisy_signal = self.input_samples[a:b]
            new_cov = np.outer(_noisy_signal, _noisy_signal).astype(self.dtype)

            # (signal+noise) cov
            self._cov_sn[-1] += new_cov

            # noise cov
            energy = np.std(_noisy_signal) ** 2
            if energy < self.thresh:
                self._cov_n[-1] += new_cov
                self.n_noise_frames[-1] += 1

        # if no new noise frames, use previous
        if self.n_noise_frames[-1] == 0:
            self._cov_n[-1] = old_cov_n
            self.n_noise_frames[-1] = 1

        # compute average for new estimate
        self.cov_sn = (self.cov_sn + self._cov_sn[-1]) / self.n_frames
        self.cov_n = (self.cov_n + self._cov_n[-1]) / sum(self.n_noise_frames)


def apply_subspace(noisy_signal, frame_len=256, mu=10, lookback=10, skip=2,
                   thresh=0.01, data_type=np.float32):
    """
    One-shot function to apply subspace denoising approach.

    Parameters
    ----------
    noisy_signal : numpy array
        Real signal in time domain.
    frame_len : int
        Frame length in samples. Note that large values (above 256) will make
        the eigendecompositions very expensive. 50% overlap is used with
        hanning window.
    mu : float
        Enhancement factor, larger values suppress more noise but could lead
        to more distortion.
    lookback : int
        How many frames to look back for covariance matrix estimation.
    skip : int
        How many samples to skip when estimating the covariance matrices with
        past samples. `skip=1` will use all possible frames in the estimation.
    thresh : float
        Threshold to distinguish between (signal+noise) and (noise) frames. A
        high value will classify more frames as noise and might even remove
        desired signal!
    data_type : 'float32' or 'float64'
        Data type to use in the enhancement procedure. Default is 'float32'.

    Returns
    -------
    numpy array
        Enhanced/denoised signal.
    """

    scnr = Subspace(frame_len, mu, lookback, skip, thresh, data_type)
    processed_audio = np.zeros(noisy_signal.shape)
    n = 0
    hop = frame_len // 2
    while noisy_signal.shape[0] - n >= hop:

        processed_audio[n:n + hop, ] = scnr.apply(noisy_signal[n:n + hop])

        # update step
        n += hop

    return processed_audio


