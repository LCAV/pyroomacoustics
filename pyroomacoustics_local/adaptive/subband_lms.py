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
        self.mu = 0.5
        self.nlms = nlms

        self.reset()


    def reset(self):

        # filter bank
        self.W = np.zeros((self.num_taps,self.num_bands),dtype=np.complex64)

        # input signal
        self.X = np.zeros((self.num_taps,self.num_bands),dtype=np.complex64)

        # reference signal
        self.D = np.zeros((self.num_bands),dtype=np.complex64)

        # error signal
        self.E = np.zeros((self.num_bands),dtype=np.complex64)


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
        self.X[1:,:] = self.X[0:-1,:]
        self.X[0,:] = X_n
        self.D = D_n

        # a priori error
        self.E = self.D - np.diag(np.dot(Hermitian(self.W),self.X))

        # compute update
        update = self.mu * np.tile(self.E.conj(),(self.num_taps,1)) * self.X
        if self.nlms:
            update /= np.tile(np.diag(np.dot(Hermitian(self.X),self.X)),
                (self.num_taps,1)) + 1e-6

        # update filter coefficients
        self.W += update



def Hermitian(X):
        '''
        Compute and return Hermitian transpose
        '''
        return X.conj().T


