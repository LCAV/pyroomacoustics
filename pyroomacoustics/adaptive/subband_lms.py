# Subband Least Mean Squares
# Copyright (C) 2019  Eric Bezzam
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


class SubbandLMS:

    '''
    Frequency domain implementation of LMS. Adaptive filter for each
    subband.

    Parameters
    ----------
    num_taps: int 
        length of the filter
    num_bands: int 
        number of frequency bands, i.e. number of filters
    mu: float, optional
        step size for each subband (default 0.5)
    nlms: bool, optional
        whether or not to normalize as in NLMS (default is True)
    '''

    def __init__(self, num_taps, num_bands, mu=0.5, nlms=True):

        self.num_taps = num_taps
        self.num_bands = num_bands
        self.mu = mu
        self.nlms = nlms

        self.reset()

    def reset(self):

        # filter bank
        self.W = np.zeros((self.num_taps, self.num_bands), dtype=np.complex64)

        # input signal
        self.X = np.zeros((self.num_taps, self.num_bands), dtype=np.complex64)

        # reference signal
        self.D = np.zeros(self.num_bands, dtype=np.complex64)

        # error signal
        self.E = np.zeros(self.num_bands, dtype=np.complex64)

    def update(self, X_n, D_n):
        '''
        Updates the adaptive filters for each subband with the new
        block of input data.

        Parameters
        ----------
        X_n: numpy array, float
            new input signal (to unknown system) in frequency domain
        D_n: numpy array, float
            new noisy reference signal in frequency domain
        '''

        # update buffers
        self.X[1:, :] = self.X[0:-1, :]
        self.X[0, :] = X_n
        self.D = D_n

        # a priori error
        self.E = self.D - np.diag(np.dot(hermitian(self.W), self.X))

        # compute update
        update = self.mu * np.tile(self.E.conj(), (self.num_taps, 1)) * self.X
        if self.nlms:
            update /= np.tile(np.diag(np.dot(hermitian(self.X), self.X)),
                              (self.num_taps, 1)) + 1e-6

        # update filter coefficients
        self.W += update


def hermitian(X):
        '''
        Compute and return Hermitian transpose
        '''
        return X.conj().T


