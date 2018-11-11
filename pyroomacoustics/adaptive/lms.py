'''
Least Mean Squares Family
=========================

Implementations of adaptive filters from the LMS class. These algorithms have a
low complexity and reliable behavior with a somewhat slower convergence.
'''
from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.linalg as la

from .adaptive_filter import AdaptiveFilter

class NLMS(AdaptiveFilter):
    '''
    Implementation of the normalized least mean squares algorithm (NLMS)


    Parameters
    ----------
    length: int
        the length of the filter
    mu: float, optional
        the step size (default 0.5)
    '''
    
    def __init__(self, length, mu=0.5):
        
        self.mu = mu
        AdaptiveFilter.__init__(self, length)
        
    def update(self, x_n, d_n):
        '''
        Updates the adaptive filter with a new sample

        Parameters
        ----------
        x_n: float
            the new input sample
        d_n: float
            the new noisy reference signal
        '''
        
        AdaptiveFilter.update(self, x_n, d_n)
        
        e = self.d - np.inner(self.x, self.w)
        self.w += self.mu*e*self.x/np.inner(self.x,self.x)
        

class BlockLMS(NLMS):
    '''
    Implementation of the least mean squares algorithm (NLMS) in its block form

    Parameters
    ----------
    length: int
        the length of the filter
    mu: float, optional
        the step size (default 0.01)
    L: int, optional
        block size (default is 1)
    nlms: bool, optional
        whether or not to normalize as in NLMS (default is False)
    '''
    
    def __init__(self, length, mu=0.01, L=1, nlms=False):
        
        self.nlms = nlms
        
        # sketching parameters
        self.L = L # block size
        
        NLMS.__init__(self, length, mu=mu)
        self.reset()
        
    def reset(self):
        '''
        Reset the state of the adaptive filter
        '''
        NLMS.reset(self)
        # We need to redefine these two guys
        self.d = np.zeros((self.L))
        self.x = np.zeros((self.L+self.length-1))
        
    def update(self, x_n, d_n):
        '''
        Updates the adaptive filter with a new sample

        Parameters
        ----------
        x_n: float
            the new input sample
        d_n: float
            the new noisy reference signal
        '''
        
        # Update the internal buffers
        self.n += 1
        slot = self.L - ((self.n-1) % self.L) - 1
        self.x[slot] = x_n
        self.d[slot] = d_n
        
        # Block update
        if self.n % self.L == 0:
            
            # block-update parameters
            X = la.hankel(self.x[:self.L],r=self.x[self.L-1:])
            
            e = self.d - np.dot(X, self.w)
            
            if self.nlms:
                norm = np.linalg.norm(X, axis=1)**2
                if self.L == 1:
                    X = X/norm[0]
                else:
                    X = (X.T/norm).T
                
            self.w += self.mu*np.dot(X.T, e)
            
            # Remember a few values
            self.x[-self.length+1:] = self.x[0:self.length-1]
        
