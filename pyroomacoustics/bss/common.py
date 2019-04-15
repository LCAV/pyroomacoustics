# Common Functions used in BSS algorithms
# Copyright (C) 2019  Robin Scheibler, Yaron Dibner, Virgile Hernicot, Juan Azcarreta
#                     MIT License
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

def projection_back(Y, ref, clip_up=None, clip_down=None):
    '''
    This function computes the frequency-domain filter that minimizes
    the squared error to a reference signal. This is commonly used
    to solve the scale ambiguity in BSS.

    Derivation of the projection
    ----------------------------

    The optimal filter `z` minimizes the squared error.
    
    .. math::

        \min E[|z^* y - x|^2]

    It should thus satsify the orthogonality condition
    and can be derived as follows

    .. math::

        0 & = E[y^*\\, (z^* y - x)]

        0 & = z^*\\, E[|y|^2] - E[y^* x]

        z^* & = \\frac{E[y^* x]}{E[|y|^2]}

        z & = \\frac{E[y x^*]}{E[|y|^2]}

    In practice, the expectations are replaced by the sample
    mean.

    Parameters
    ----------
    Y: array_like (n_frames, n_bins, n_channels)
        The STFT data to project back on the reference signal
    ref: array_like (n_frames, n_bins)
        The reference signal
    clip_up: float, optional
        Limits the maximum value of the gain (default no limit)
    clip_down: float, optional
        Limits the minimum value of the gain (default no limit)
    '''

    num = np.sum(np.conj(ref[:,:,None]) * Y, axis=0)
    denom = np.sum(np.abs(Y)**2, axis=0)

    c = np.ones(num.shape, dtype=np.complex)
    I = denom > 0.
    c[I] = num[I] / denom[I]

    if clip_up is not None:
        I = np.logical_and(np.abs(c) > clip_up, np.abs(c) > 0)
        c[I] *= clip_up / np.abs(c[I])

    if clip_down is not None:
        I = np.logical_and(np.abs(c) < clip_down, np.abs(c) > 0)
        c[I] *= clip_down / np.abs(c[I])
    return c



def sparir(G, S, weights=np.array([]), gini=0, maxiter=50, tol=10, alpha=10, alphamax=1e5, alphamin=1e-7):
    '''

   Fast proximal algorithm implementation for sparse approximation of relative impulse
   responses from incomplete measurements of the corresponding relative transfer function
   based on

    Z. Koldovsky, J. Malek, and S. Gannot, "Spatial Source Subtraction based
    on Incomplete Measurements of Relative Transfer Function", IEEE/ACM
    Transactions on Audio, Speech, and Language Processing, TASLP 2015.

   The original Matlab implementation can be found at
    http://itakura.ite.tul.cz/zbynek/dwnld/SpaRIR.m

   and it is referred in

    Z. Koldovsky, F. Nesta, P. Tichavsky, and N. Ono, *Frequency-domain blind
    speech separation using incomplete de-mixing transform*, EUSIPCO 2016.

    Parameters
    ----------
    G: ndarray (nfrequencies, 1)
        Frequency representation of the (incomplete) transfer function
    S: ndarray (kfrequencies)
        Indexes  of active frequency bins for sparse AuxIVA
    weights: ndarray (kfrequencies) or int, optional
        The higher the value of weights(i), the higher the probability that g(i)
        is zero; if scalar, all weights are the same; if empty, default value is
        used
    gini: ndarray (nfrequencies)
        Initialization for the computation of g
    maxiter: int
        Maximum number of iterations before achieving convergence (default 50)
    tol: float
        Minimum convergence criteria based on the gradient difference between adjacent updates (default 10)
    alpha: float
        Inverse of the decreasing speed of the gradient at each iteration. This parameter
        is updated at every iteration (default 10)
    alphamax: float
        Upper bound for alpha (default 1e5)
    alphamin: float
        Lower bound for alpha (default 1e-7)

    Returns
    -------
    Returns the sparse approximation of the impulse response in the
    time-domain (real-valued) as an (nfrequencies) array.
    '''

    n_freq = G.shape[0]

    y = np.concatenate((np.real(G[S]), np.imag(G[S])), axis=0)
    M = y.shape[0]

    if gini == 0:  # if no initialization is given
        g = np.zeros((n_freq, 1))
        g[0] = 1
    else:
        g = gini

    if weights.size == 0:
        tau = np.sqrt(n_freq) / (y.conj().T.dot(y))
        tau = tau * np.exp(0.11 * np.abs((np.arange(1., n_freq + 1.).T)) ** 0.3)
        tau = tau.T
    elif weights.shape[0] == 1:
        tau = np.ones((n_freq, 1)) * weights
    else:
        tau = np.tile(weights.T, (1, 1)).reshape(n_freq)

    def soft(x, T):
        if np.sum(np.abs(T).flatten()) == 0:
            u = x
        else:
            u = np.max(np.abs(x) - T, 0)
            u = u / (u + T) * x
        return u

    aux = np.zeros((n_freq, 1), dtype=complex)
    G = np.fft.fft(g.flatten())
    Ag = np.concatenate((np.real(G[S]), np.imag(G[S])), axis=0)
    r = Ag - y.flatten()  # instead of r = A * g - y
    aux[S] = np.expand_dims(r[0:M // 2] + 1j * r[M // 2:], axis=1)
    gradq = n_freq * np.fft.irfft(aux.flatten(), n_freq)  # instead of gradq = A'*r
    gradq = np.expand_dims(gradq, axis=1)
    support = g != 0
    iter_ = 0 # initial iteration value

    # Define stopping criteria
    crit = np.zeros((maxiter, 1))
    criterion = -tau[support] * np.sign(g[support]) - gradq[support]
    crit[iter_] = np.sum(criterion ** 2)

    while (crit[iter_] > tol) and (iter_ < maxiter - 1):
        # Update gradient
        prev_r = r
        prev_g = g
        g = soft(prev_g - gradq * (1.0 / alpha), tau / alpha)
        dg = g - prev_g
        DG = np.fft.fft(dg.flatten())
        Adg = np.concatenate((np.real(DG[S]), np.imag(DG[S])), axis=0)
        r = prev_r + Adg.flatten()  # faster than A * g - y
        dd = np.dot(dg.flatten().conj().T, dg.flatten())
        dGd = np.dot(Adg.flatten().conj().T, Adg.flatten())
        alpha = min(alphamax, max(alphamin, dGd / (np.finfo(np.float32).eps + dd)))
        iter_ += 1
        support = g != 0
        aux[S] = np.expand_dims(r[0:M // 2] + 1j * r[M // 2:], axis=1)
        gradq = n_freq * np.fft.irfft(aux.flatten(), n_freq)
        gradq = np.expand_dims(gradq, axis=1)
        # Update stopping criteria
        criterion = -tau[support] * np.sign(g[support]) - gradq[support]
        crit[iter_] = sum(criterion ** 2) + sum(abs(gradq[~support]) - tau[~support] > tol)

    return g.flatten()
