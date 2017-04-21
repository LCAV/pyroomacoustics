# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

from __future__ import division

import numpy as np
import scipy.linalg as la

from .parameters import constants
from . import utilities as u
from .soundsource import build_rir_matrix
from . import windows
from . import stft


#=========================================================================
# Free (non-class-member) functions related to beamformer design
#=========================================================================


def H(A, **kwargs):
    '''Returns the conjugate (Hermitian) transpose of a matrix.'''

    return np.transpose(A, **kwargs).conj()


def sumcols(A):
    '''
    Sums the columns of a matrix (np.array).

    The output is a 2D np.array
    of dimensions M x 1.
    '''

    return np.sum(A, axis=1, keepdims=1)


def mdot(*args):
    '''Left-to-right associative matrix multiplication of multiple 2D ndarrays.'''

    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret, a)

    return ret


def distance(x, y):
    '''
    Computes the distance matrix E.

    E[i,j] = sqrt(sum((x[:,i]-y[:,j])**2)).
    x and y are DxN ndarray containing N D-dimensional vectors.
    '''

    # Assume x, y are arrays, *not* matrices
    x = np.array(x)
    y = np.array(y)

    # return np.sqrt((x[0,:,np.newaxis]-y[0,:])**2 + (x[1,:,np.newaxis]-y[1,:])**2)

    return np.sqrt(np.sum((x[:, :, np.newaxis] - y[:, np.newaxis, :])**2, axis=0))


def unit_vec2D(phi):
    return np.array([[np.cos(phi), np.sin(phi)]]).T


def linear_2D_array(center, M, phi, d):
    u = unit_vec2D(phi)
    return np.array(center)[:, np.newaxis] + d * \
        (np.arange(M)[np.newaxis, :] - (M - 1.) / 2.) * u


def circular_2D_array(center, M, phi0, radius):
    phi = np.arange(M) * 2. * np.pi / M
    return np.array(center)[:, np.newaxis] + radius * \
        np.vstack((np.cos(phi + phi0), np.sin(phi + phi0)))


def poisson_2D_array(center, M, d):
    ''' Create array of 2D positions drawn from Poisson process. '''

    from numpy.random import standard_exponential, randint

    R = d*standard_exponential((2, M))*(2*randint(0, 2, (2, M)) - 1)
    R = R.cumsum(axis=1)
    R -= R.mean(axis=1)[:, np.newaxis]
    R += np.array([center]).T

    return R


def square_2D_array(center, M, N, phi, d):

    c = linear_2D_array(center, M, phi+np.pi/2., d)
    R = np.zeros((2, M*N))
    for i in np.arange(M):
        R[:, i*N:(i+1)*N] = linear_2D_array(c[:, i], N, phi, d)

    return R

def spiral_2D_array(center, M, radius=1., divi=3, angle=None):
    '''
    generate microphone array layout randomly
    :param center: location of the center of the array
    :param radius: microphones are contained within a cirle of this radius (default 1)
    :param M: number of microphones
    :param divi: number of rotations of the spiral (default 3)
    :param angle: the angle offset of the spiral (default random)
    :return:
    '''
    num_seg = np.ceil(M / divi)

    pos_array_norm = np.linspace(0, radius, num=M, endpoint=False)

    pos_array_angle = np.reshape(np.tile(np.pi * 2 * np.arange(divi) / divi, num_seg),
                                 (divi, -1), order='F') + \
                      np.linspace(0, 2 * np.pi / divi,
                                  num=num_seg, endpoint=False)[np.newaxis, :]
    pos_array_angle = np.insert(pos_array_angle.flatten('F')[:M - 1], 0, 0)

    if angle is None:
        pos_array_angle += np.random.rand() * np.pi / divi
    else:
        pos_array_angle += angle

    pos_mic_x = pos_array_norm * np.cos(pos_array_angle)
    pos_mic_y = pos_array_norm * np.sin(pos_array_angle)

    return np.array([pos_mic_x, pos_mic_y])


def fir_approximation_ls(weights, T, n1, n2):

    freqs_plus = np.array(weights.keys())[:, np.newaxis]
    freqs = np.vstack([freqs_plus,
                       -freqs_plus])
    omega = 2 * np.pi * freqs
    omega_discrete = omega * T

    n = np.arange(n1, n2)

    # Create the DTFT transform matrix corresponding to a discrete set of
    # frequencies and the FIR filter indices
    F = np.exp(-1j * omega_discrete * n)

    w_plus = np.array(weights.values())[:, :, 0]
    w = np.vstack([w_plus,
                   w_plus.conj()])

    return np.linalg.pinv(F).dot(w)


#=========================================================================
# Classes (microphone array and beamformer related)
#=========================================================================


class MicrophoneArray(object):

    '''Microphone array class.'''

    def __init__(self, R, Fs):
        self.dim = R.shape[0]   # are we in 2D or in 3D
        self.M = R.shape[1]     # number of microphones
        self.R = R              # array geometry

        self.Fs = Fs            # sampling frequency of microphones

        self.signals = None

        self.center = np.mean(R, axis=1, keepdims=True)


    def record(self, signals, Fs):
        '''
        This simulates the recording of the signals by the microphones.
        In particular, if the microphones and the room simulation
        do not use the same sampling frequency, down/up-sampling
        is done here.

        Parameters:
        -----------

        signals: An ndarray with as many lines as there are microphones.
        Fs: the sampling frequency of the signals.
        '''

        if signals.shape[0] != self.M:
            raise NameError('The signals array should have as many lines as there are microphones.')

        if signals.ndim != 2:
            raise NameError('The signals should be a 2D array.')

        if Fs != self.Fs:
            from scikits.samplerate import resample
            Fs_ratio = self.Fs/float(Fs)
            newL = int(Fs_ratio*signals.shape[1])-1
            self.signals = np.zeros((self.M, newL))
            # scikits.samplerate resample function considers columns as channels (hence the transpose)
            for m in range(self.M):
                self.signals[m] = resample(signals[m], Fs_ratio, 'sinc_best')
        else:
            self.signals = signals


    def to_wav(self, filename, mono=False, norm=False, type=float):
        '''Save all the signals to wav files.'''
        from scipy.io import wavfile

        if mono is True:
            signal = self.signals[self.M/2]
        else:
            signal = self.signals.T  # each column is a channel

        if type is float:
            bits = None
        elif type is np.int8:
            bits = 8
        elif type is np.int16:
            bits = 16
        elif type is np.int32:
            bits = 32
        elif type is np.int64:
            bits = 64
        else:
            raise NameError('No such type.')

        if norm is True:
            from utilities import normalize
            signal = normalize(signal, bits=bits)

        signal = np.array(signal, dtype=type)

        wavfile.write(filename, self.Fs, signal)


class Beamformer(MicrophoneArray):

    '''
    Beamformer class.

    At some point, in some nice way, the design methods
    should also go here. Probably with generic arguments.
    '''

    def __init__(self, R, Fs, N=1024, Lg=None, hop=None, zpf=0, zpb=0):
        '''
        Parameters:
        -----------
        R        Mics positions
        Fs       Sampling frequency
        N=1024   Length of FFT, i.e. number of FD beamforming weights, equally spaced. Defaults to 1024.
        Lg=N     Length of time-domain filters. Default to N.
        hop=N/2  Hop length for frequency domain processing. Default to N/2.
        zpf=0    Front zero padding length for frequency domain processing. Default is 0.
        zpb=0    Zero padding length for frequency domain processing. Default is 0.

        '''
        MicrophoneArray.__init__(self, R, Fs)

        # only support even length (in freq)
        if N % 2 is 1:
            N += 1

        self.N = int(N)    # FFT length

        if Lg is None:
            self.Lg = N    # TD filters length
        else:
            self.Lg = int(Lg)

        # setup lengths for FD processing
        self.zpf = int(zpf)
        self.zpb = int(zpb)
        self.L = self.N - self.zpf - self.zpb
        if hop is None:
            self.hop = self.L/2
        else:
            self.hop = hop

        # for now only support equally spaced frequencies
        self.frequencies = np.arange(0, self.N/2+1)/float(self.N)*float(self.Fs)

        # weights will be computed later, the array is of shape (M, N/2+1)
        self.weights = None

        # the TD beamforming filters (M, Lg)
        self.filters = None

    def __add__(self, y):
        ''' Concatenates two beamformers together.'''

        newR = np.concatenate((self.R, y.R), axis=1)
        return Beamformer(newR, self.Fs, self.Lg, self.N, hop=self.hop, zpf=self.zpf, zpb=self.zpb)

    def filters_from_weights(self, non_causal=0.):
        ''' Compute time-domain filters from frequency domain weights.'''

        if self.weights is None:
            raise NameError('Weights must be defined.')

        self.filters = np.zeros((self.M, self.Lg))

        if self.N <= self.Lg:

            # go back to time domain and shift DC to center
            tw = np.fft.irfft(np.conj(self.weights), axis=1, n=self.N)
            self.filters[:, :self.N] = np.concatenate((tw[:, -self.N/2:], tw[:, :self.N/2]), axis=1)

        elif self.N > self.Lg:

            # Least-square projection
            for i in np.arange(self.M):
                Lgp = np.floor((1 - non_causal)*self.Lg)
                Lgm = self.Lg - Lgp
                # the beamforming weights in frequency are the complex conjugates of the FT of the filter
                w = np.concatenate((np.conj(self.weights[i]), self.weights[i, -2:0:-1]))

                # create partial Fourier matrix
                k = np.arange(self.N)[:, np.newaxis]
                l = np.concatenate((np.arange(self.N-Lgm, self.N), np.arange(Lgp)))
                F = np.exp(-2j*np.pi*k*l/float(self.N))

                self.filters[i] = np.real(np.linalg.lstsq(F, w)[0])

    def weights_from_filters(self):

        if self.filters is None:
            raise NameError('Filters must be defined.')

        # this is what we want to use, really.
        #self.weights = np.conj(np.fft.rfft(self.filters, n=self.N, axis=1))

        # quick hack to be able to use MKL acceleration package from anaconda
        self.weights = np.zeros((self.M, self.N/2+1), dtype=np.complex128)
        for m in range(self.M):
            self.weights[m] = np.conj(np.fft.rfft(self.filters[m], n=self.N))

    def steering_vector_2D(self, frequency, phi, dist, attn=False):

        phi = np.array([phi]).reshape(phi.size)

        # Assume phi and dist are measured from the array's center
        X = dist * np.array([np.cos(phi), np.sin(phi)]) + self.center

        D = distance(self.R, X)
        omega = 2 * np.pi * frequency

        if attn:
            # TO DO 1: This will mean slightly different absolute value for
            # every entry, even within the same steering vector. Perhaps a
            # better paradigm is far-field with phase carrier.
            return 1. / (4 * np.pi) / D * np.exp(-1j * omega * D / constants.get('c'))
        else:
            return np.exp(-1j * omega * D / constants.get('c'))

    def steering_vector_2D_from_point(self, frequency, source, attn=True, ff=False):
        ''' Creates a steering vector for a particular frequency and source

        Args:
            frequency
            source: location in cartesian coordinates
            attn: include attenuation factor if True
            ff:   uses far-field distance if true

        Return:
            A 2x1 ndarray containing the steering vector.
        '''

        X = np.array(source)
        if X.ndim == 1:
            X = source[:, np.newaxis]

        omega = 2 * np.pi * frequency

        # normalize for far-field if requested
        if (ff):
            # unit vectors pointing towards sources
            p = (X - self.center)
            p /= np.linalg.norm(p)

            # The projected microphone distances on the unit vectors
            D = np.dot(self.R.T, p)

            # subtract minimum in each column
            D -= np.min(D)

        else:

            D = distance(self.R, X)

        phase = np.exp(-1j * omega * D / constants.get('c'))

        if attn:
            # TO DO 1: This will mean slightly different absolute value for
            # every entry, even within the same steering vector. Perhaps a
            # better paradigm is far-field with phase carrier.
            return 1. / (4 * np.pi) / D * phase
        else:
            return phase

    def response(self, phi_list, frequency):

        i_freq = np.argmin(np.abs(self.frequencies - frequency))

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError('Beamforming weights or filters need to be computed first.')

        # For the moment assume that we are in 2D
        bfresp = np.dot(H(self.weights[:,i_freq]), self.steering_vector_2D(
            self.frequencies[i_freq], phi_list, constants.get('ffdist')))

        return self.frequencies[i_freq], bfresp

    def response_from_point(self, x, frequency):

        i_freq = np.argmin(np.abs(self.frequencies - frequency))

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError('Beamforming weights or filters need to be computed first.')

        # For the moment assume that we are in 2D
        bfresp = np.dot(H(self.weights[:, i_freq]), self.steering_vector_2D_from_point(
            self.frequencies[i_freq], x, attn=True, ff=False))

        return self.frequencies[i_freq], bfresp

    def plot_response_from_point(self, x, legend=None):

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError('Beamforming weights or filters need to be computed first.')

        if x.ndim == 0:
            x = np.array([x])

        import matplotlib.pyplot as plt

        HF = np.zeros((x.shape[1], self.frequencies.shape[0]), dtype=complex)
        for k, p in enumerate(x.T):
            for i, f in enumerate(self.frequencies):
                r = np.dot(H(self.weights[:, i]),
                           self.steering_vector_2D_from_point(f, p, attn=True, ff=False))
                HF[k, i] = r[0]

        plt.subplot(2, 1, 1)
        plt.title('Beamformer response')
        for hf in HF:
            plt.plot(self.frequencies, np.abs(hf))
        plt.ylabel('Modulus')
        plt.axis('tight')
        plt.legend(legend)

        plt.subplot(2, 1, 2)
        for hf in HF:
            plt.plot(self.frequencies, np.unwrap(np.angle(hf)))
        plt.ylabel('Phase')
        plt.xlabel('Frequency [Hz]')
        plt.axis('tight')
        plt.legend(legend)

    def plot_beam_response(self):

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError('Beamforming weights or filters need to be computed first.')

        phi = np.linspace(-np.pi, np.pi-np.pi/180, 360)
        freq = self.frequencies

        resp = np.zeros((freq.shape[0], phi.shape[0]), dtype=complex)

        for i, f in enumerate(freq):
            # For the moment assume that we are in 2D
            resp[i,:] = np.dot(H(self.weights[:,i]), self.steering_vector_2D(
                f, phi, constants.get('ffdist')))

        H_abs = np.abs(resp)**2
        H_abs /= H_abs.max()
        H_abs = 10*np.log10(H_abs)

        p_min = 0
        p_max = 100
        vmin, vmax = np.percentile(H_abs.flatten(), [p_min, p_max])

        import matplotlib.pyplot as plt

        plt.imshow(H_abs,
                   aspect='auto',
                   origin='lower',
                   interpolation='sinc',
                   vmax=vmax, vmin=vmin)

        plt.xlabel('Angle [rad]')
        xticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        for i, p in enumerate(xticks):
            xticks[i] = np.argmin(np.abs(p - phi))
        xticklabels = ['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$']
        plt.setp(plt.gca(), 'xticks', xticks)
        plt.setp(plt.gca(), 'xticklabels', xticklabels)

        plt.ylabel('Freq [kHz]')
        yticks = np.zeros(4)
        f_0 = np.floor(self.Fs/8000.)
        for i in np.arange(1, 5):
            yticks[i-1] = np.argmin(np.abs(freq - 1000.*i*f_0))
        #yticks = np.array(plt.getp(plt.gca(), 'yticks'), dtype=np.int)
        plt.setp(plt.gca(), 'yticks', yticks)
        plt.setp(plt.gca(), 'yticklabels', np.arange(1, 5)*f_0)

    def snr(self, source, interferer, f, R_n=None, dB=False):

        i_f = np.argmin(np.abs(self.frequencies - f))

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError('Beamforming weights or filters need to be computed first.')

        # This works at a single frequency because otherwise we need to pass
        # many many covariance matrices. Easy to change though (you can also
        # have frequency independent R_n).

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        # To compute the SNR, we /must/ use the real steering vectors, so no
        # far field, and attn=True
        A_good = self.steering_vector_2D_from_point(self.frequencies[i_f], source.images, attn=True, ff=False)

        if interferer is not None:
            A_bad  = self.steering_vector_2D_from_point(self.frequencies[i_f], interferer.images, attn=True, ff=False)
            R_nq = R_n + sumcols(A_bad) * H(sumcols(A_bad))
        else:
            R_nq = R_n

        w = self.weights[:, i_f]
        a_1 = sumcols(A_good)

        SNR = np.real(mdot(H(w), a_1, H(a_1), w) / mdot(H(w), R_nq, w))

        if dB is True:
            SNR = 10 * np.log10(SNR)

        return SNR

    def udr(self, source, interferer, f, R_n=None, dB=False):

        i_f = np.argmin(np.abs(self.frequencies - f))

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is None and self.filters is None:
            raise NameError('Beamforming weights or filters need to be computed first.')

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        A_good = self.steering_vector_2D_from_point(self.frequencies[i_f], source.images, attn=True, ff=False)

        if interferer is not None:
            A_bad  = self.steering_vector_2D_from_point(self.frequencies[i_f], interferer.images, attn=True, ff=False)
            R_nq = R_n + sumcols(A_bad).dot(H(sumcols(A_bad)))
        else:
            R_nq = R_n

        w = self.weights[:, i_f]

        UDR = np.real(mdot(H(w), A_good, H(A_good), w) / mdot(H(w), R_nq, w))
        if dB is True:
            UDR = 10 * np.log10(UDR)

        return UDR

    def process(self, FD=False):

        if self.signals is None or len(self.signals) == 0:
            raise NameError('No signal to beamform')

        if FD is True:

            # STFT processing

            if self.weights is None and self.filters is not None:
                self.weights_from_filters()
            elif self.weights is None and self.filters is None:
                raise NameError('Beamforming weights or filters need to be computed first.')

            # create window function
            win = np.concatenate((np.zeros(self.zpf),
                                  windows.hann(self.L),
                                  np.zeros(self.zpb)))

            # do real STFT of first signal
            tfd_sig = stft.stft(self.signals[0],
                                self.L,
                                self.hop,
                                zp_back=self.zpb,
                                zp_front=self.zpf,
                                transform=np.fft.rfft,
                                win=win) * np.conj(self.weights[0])
            for i in range(1, self.M):
                tfd_sig += stft.stft(self.signals[i],
                                     self.L,
                                     self.hop,
                                     zp_back=self.zpb,
                                     zp_front=self.zpf,
                                     transform=np.fft.rfft,
                                     win=win) * np.conj(self.weights[i])

            #  now reconstruct the signal
            output = stft.istft(
                tfd_sig,
                self.L,
                self.hop,
                zp_back=self.zpb,
                zp_front=self.zpf,
                transform=np.fft.irfft)

            # remove the zero padding from output signal
            if self.zpb is 0:
                output = output[self.zpf:]
            else:
                output = output[self.zpf:-self.zpb]

        else:

            # TD processing

            if self.weights is not None and self.filters is None:
                self.filters_from_weights()
            elif self.weights is None and self.filters is None:
                raise NameError('Beamforming weights or filters need to be computed first.')

            from scipy.signal import fftconvolve

            # do real STFT of first signal
            output = fftconvolve(self.filters[0], self.signals[0])
            for i in range(1, len(self.signals)):
                output += fftconvolve(self.filters[i], self.signals[i])

        return output

    def plot(self, sum_ir=False, FD=True):

        if self.weights is None and self.filters is not None:
            self.weights_from_filters()
        elif self.weights is not None and self.filters is None:
            self.filters_from_weights()
        elif self.weights is None and self.filters is None:
            raise NameError('Beamforming weights or filters need to be computed first.')

        import matplotlib.pyplot as plt

        if FD is True:
            plt.subplot(2, 2, 1)
            plt.plot(self.frequencies, np.abs(self.weights.T))
            plt.title('Beamforming weights [modulus]')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Weight modulus')

            plt.subplot(2, 2, 2)
            plt.plot(self.frequencies, np.unwrap(np.angle(self.weights.T), axis=0))
            plt.title('Beamforming weights [phase]')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Unwrapped phase')

            plt.subplot(2, 1, 2)

        plt.plot(np.arange(self.Lg)/float(self.Fs), self.filters.T)

        plt.title('Beamforming filters')
        plt.xlabel('Time [s]')
        plt.ylabel('Filter amplitude')
        plt.axis('tight')

    def far_field_weights(self, phi):
        '''
        This method computes weight for a far field at infinity

        phi: direction of beam
        '''

        u = unit_vec2D(phi)
        proj = np.dot(u.T, self.R - self.center)[0]

        # normalize the first arriving signal to ensure a causal filter
        proj -= proj.max()

        self.weights = np.exp(2j * np.pi * 
        self.frequencies[:, np.newaxis] * proj / constants.get('c')).T


    def rake_delay_and_sum_weights(self, source, interferer=None, R_n=None, attn=True, ff=False):

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        K = source.images.shape[1] - 1

        for i, f in enumerate(self.frequencies):
            W = self.steering_vector_2D_from_point(f, source.images, attn=attn, ff=ff)
            self.weights[:,i] = 1.0/self.M/(K+1) * np.sum(W, axis=1)


    def rake_one_forcing_weights(self, source, interferer, R_n=None, ff=False, attn=True):

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        for i, f in enumerate(self.frequencies):
            if interferer is None:
                A_bad = np.array([[]])
            else:
                A_bad = self.steering_vector_2D_from_point(f, interferer.images, attn=attn, ff=ff)

            R_nq = R_n + sumcols(A_bad).dot(H(sumcols(A_bad)))

            A_s      = self.steering_vector_2D_from_point(f, source.images, attn=attn, ff=ff)
            R_nq_inv = np.linalg.pinv(R_nq)
            D = np.linalg.pinv(mdot(H(A_s), R_nq_inv, A_s))

            self.weights[:, i] = sumcols(mdot(R_nq_inv, A_s, D))[:, 0]

    def rake_max_sinr_weights(self, source, interferer, R_n=None,
                           rcond=0., ff=False, attn=True):
        '''
        This method computes a beamformer focusing on a number of specific sources
        and ignoring a number of interferers.

        INPUTS
          * source     : source locations
          * interferer : interferer locations
        '''

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        for i, f in enumerate(self.frequencies):

            A_good = self.steering_vector_2D_from_point(f, source.images, attn=attn, ff=ff)

            if interferer is None:
                A_bad = np.array([[]])
            else:
                A_bad = self.steering_vector_2D_from_point(f, interferer.images, attn=attn, ff=ff)

            a_good = sumcols(A_good)
            a_bad = sumcols(A_bad)

            # TO DO: Fix this (check for numerical rank, use the low rank approximation)
            K_inv = np.linalg.pinv(a_bad.dot(H(a_bad)) + R_n + rcond * np.eye(A_bad.shape[0]))
            self.weights[:, i] = (K_inv.dot(a_good) / mdot(H(a_good), K_inv, a_good))[:, 0]

    def rake_max_udr_weights(self, source, interferer, R_n=None, ff=False, attn=True):
        
        if source.images.shape[1] == 1:
            self.rake_max_sinr_weights(source.images, interferer.images, R_n=R_n, ff=ff, attn=attn)
            return

        if R_n is None:
            R_n = np.zeros((self.M, self.M))

        self.weights = np.zeros((self.M, self.frequencies.shape[0]), dtype=complex)

        for i, f in enumerate(self.frequencies):
            A_good = self.steering_vector_2D_from_point(f, source.images, attn=attn, ff=ff)

            if interferer is None:
                A_bad = np.array([[]])
            else:
                A_bad = self.steering_vector_2D_from_point(f, interferer.images, attn=attn, ff=ff)

            R_nq = R_n + sumcols(A_bad).dot(H(sumcols(A_bad)))

            C = np.linalg.cholesky(R_nq)
            l, v = np.linalg.eig(mdot(np.linalg.inv(C), A_good, H(A_good), H(np.linalg.inv(C))))

            self.weights[:, i] = np.linalg.inv(H(C)).dot(v[:, 0])

    def rake_max_udr_filters(self, source, interferer, R_n, delay=0.03, epsilon=5e-3):
        '''
        Compute directly the time-domain filters for a UDR maximizing beamformer.
        '''

        H = build_rir_matrix(self.R, (source, interferer), self.Lg, self.Fs, epsilon=epsilon, unit_damping=True)
        L = H.shape[1]/2

        # Delay of the system in samples
        kappa = int(delay*self.Fs)
        precedence = int(0.030*self.Fs)

        # the constraint
        n = np.minimum(L, kappa+precedence)
        Hnc = H[:, :kappa]
        Hpr = H[:, kappa:n]
        Hc = H[:, n:L]
        A = np.dot(Hpr, Hpr.T)
        B = np.dot(Hnc, Hnc.T) + np.dot(Hc, Hc.T) + np.dot(H[:, L:], H[:, L:].T) + R_n

        # solve the problem
        SINR, v = la.eigh(A, b=B, eigvals=(self.M*self.Lg-1, self.M*self.Lg-1), overwrite_a=True, overwrite_b=True, check_finite=False)
        g_val = np.real(v[:, 0])

        # reshape and store
        self.filters = g_val.reshape((self.M, self.Lg))

        # compute and return SNR
        return SINR[0]


    def rake_perceptual_filters(self, source, interferer, R_n, delay=0.03, d_relax=0.035, epsilon=5e-3):
        '''
        Compute directly the time-domain filters for a perceptually motivated beamformer.
        The beamformer minimizes noise and interference, but relaxes the response of the
        filter within the 30 ms following the delay.
        '''

        # build the channel matrix
        H = build_rir_matrix(self.R, (source, interferer), self.Lg, self.Fs, epsilon=epsilon, unit_damping=True)
        L = H.shape[1] // 2

        # Delay of the system in samples
        tau = int(delay*self.Fs)
        kappa = int(d_relax*self.Fs)

        # the constraint
        A = np.concatenate((H[:,:tau+1], H[:,tau+kappa:]), axis=1)
        b = np.zeros((A.shape[1],1))
        b[tau,0] = 1

        # We first assume the sample are uncorrelated
        K_nq = np.dot(H[:, L:], H[:, L:].T) + R_n

        # causal response construction
        C = la.cho_factor(K_nq, overwrite_a=True, check_finite=False)
        B = la.cho_solve(C, A)
        D = np.dot(A.T, B)
        C = la.cho_factor(D, overwrite_a=True, check_finite=False)
        x = la.cho_solve(C, b)
        g_val = np.dot(B, x)

        # reshape and store
        self.filters = g_val.reshape((self.M, self.Lg))

        # compute and return SNR
        A = np.dot(g_val.T, H[:, :L])
        num = np.dot(A, A.T)
        denom =  np.dot(np.dot(g_val.T, K_nq), g_val)

        return num/denom


    def rake_max_sinr_filters(self, source, interferer, R_n, epsilon=5e-3, delay=0.):
        '''
        Compute the time-domain filters of SINR maximizing beamformer.
        '''

        H = build_rir_matrix(self.R, (source, interferer), self.Lg, self.Fs, epsilon=epsilon, unit_damping=True)
        L = H.shape[1]/2

        # We first assume the sample are uncorrelated
        K_s = np.dot(H[:, :L], H[:, :L].T)
        K_nq = np.dot(H[:, L:], H[:, L:].T) + R_n

        # Compute TD filters using generalized Rayleigh coefficient maximization
        SINR, v = la.eigh(K_s, b=K_nq, eigvals=(self.M*self.Lg-1, self.M*self.Lg-1), overwrite_a=True, overwrite_b=True, check_finite=False)
        g_val = np.real(v[:, 0])

        self.filters = g_val.reshape((self.M, self.Lg))

        # compute and return SNR
        return SINR[0]

    def rake_distortionless_filters(self, source, interferer, R_n, delay=0.03, epsilon=5e-3):
        '''
        Compute time-domain filters of a beamformer minimizing noise and interference
        while forcing a distortionless response towards the source.
        '''

        H = build_rir_matrix(self.R, (source, interferer), self.Lg, self.Fs, epsilon=epsilon, unit_damping=True)
        L = H.shape[1]/2

        # We first assume the sample are uncorrelated
        K_nq = np.dot(H[:, L:], H[:, L:].T) + R_n

        # constraint
        kappa = int(delay*self.Fs)
        A = H[:, :L]
        b = np.zeros((L, 1))
        b[kappa, 0] = 1

        # filter computation
        C = la.cho_factor(K_nq, overwrite_a=True, check_finite=False)
        B = la.cho_solve(C, A)
        D = np.dot(A.T, B)
        C = la.cho_factor(D, overwrite_a=True, check_finite=False)
        x = la.cho_solve(C, b)
        g_val = np.dot(B, x)

        # reshape and store
        self.filters = g_val.reshape((self.M, self.Lg))

        # compute and return SNR
        A = np.dot(g_val.T, H[:, :L])
        num = np.dot(A, A.T)
        denom = np.dot(np.dot(g_val.T, K_nq), g_val)

        return num/denom

    def rake_mvdr_filters(self, source, interferer, R_n, delay=0.03, epsilon=5e-3):
        '''
        Compute the time-domain filters of the minimum variance distortionless
        response beamformer.
        '''

        H = build_rir_matrix(self.R, (source, interferer), self.Lg, self.Fs, epsilon=epsilon, unit_damping=True)
        L = H.shape[1] // 2

        # the constraint vector
        kappa = int(delay*self.Fs)
        h = H[:, kappa]

        # We first assume the sample are uncorrelated
        R_xx = np.dot(H[:, :L], H[:, :L].T)
        K_nq = np.dot(H[:, L:], H[:, L:].T) + R_n

        # Compute the TD filters
        C = la.cho_factor(R_xx + K_nq, check_finite=False)
        g_val = la.cho_solve(C, h)

        g_val /= np.inner(h, g_val)
        self.filters = g_val.reshape((self.M, self.Lg))

        # compute and return SNR
        num = np.inner(g_val.T, np.dot(R_xx, g_val))
        denom = np.inner(np.dot(g_val.T, K_nq), g_val)

        return num/denom

    def rake_one_forcing_filters(self, sources, interferers, R_n, epsilon=5e-3):
        '''
        Compute the time-domain filters of a beamformer with unit response
        towards multiple sources.
        '''

        dist_mat = distance(self.R, sources.images)
        s_time = dist_mat / constants.get('c')
        s_dmp = 1./(4*np.pi*dist_mat)

        dist_mat = distance(self.R, interferers.images)
        i_time = dist_mat / constants.get('c')
        i_dmp = 1./(4*np.pi*dist_mat)

        # compute offset needed for decay of sinc by epsilon
        offset = np.maximum(s_dmp.max(), i_dmp.max())/(np.pi*self.Fs*epsilon)
        t_min = np.minimum(s_time.min(), i_time.min())
        t_max = np.maximum(s_time.max(), i_time.max())

        # adjust timing
        s_time -= t_min - offset
        i_time -= t_min - offset
        Lh = np.ceil((t_max - t_min + 2*offset)*float(self.Fs))

        # the channel matrix
        K = sources.images.shape[1]
        Lg = self.Lg
        off = (Lg - Lh)/2
        L = self.Lg + Lh - 1

        H = np.zeros((Lg*self.M, 2*L))
        As = np.zeros((Lg*self.M, K))

        for r in np.arange(self.M):

            # build constraint matrix
            hs = u.low_pass_dirac(s_time[r, :, np.newaxis], s_dmp[r, :, np.newaxis], self.Fs, Lh)[:, ::-1]
            As[r*Lg+off:r*Lg+Lh+off, :] = hs.T

            # build interferer RIR matrix
            hx = u.low_pass_dirac(s_time[r, :, np.newaxis], s_dmp[r, :, np.newaxis], self.Fs, Lh).sum(axis=0)
            H[r*Lg:(r+1)*Lg, :L] = u.convmtx(hx, Lg).T

            # build interferer RIR matrix
            hq = u.low_pass_dirac(i_time[r, :, np.newaxis], i_dmp[r, :, np.newaxis], self.Fs, Lh).sum(axis=0)
            H[r*Lg:(r+1)*Lg, L:] = u.convmtx(hq, Lg).T

        ones = np.ones((K, 1))

        # We first assume the sample are uncorrelated
        K_x = np.dot(H[:, :L], H[:, :L].T)
        K_nq = np.dot(H[:, L:], H[:, L:].T) + R_n

        # Compute the TD filters
        K_nq_inv = np.linalg.inv(K_x+K_nq)
        C = np.dot(K_nq_inv, As)
        B = np.linalg.inv(np.dot(As.T, C))
        g_val = np.dot(C, np.dot(B, ones))
        self.filters = g_val.reshape((self.M, Lg))

        # compute and return SNR
        A = np.dot(g_val.T, H[:, :L])
        num = np.dot(A, A.T)
        denom = np.dot(np.dot(g_val.T, K_nq), g_val)

        return num/denom
