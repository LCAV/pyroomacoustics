from __future__ import division, print_function
import scipy.linalg as la
import numpy as np


class AdaptiveFilter:
    '''
    The dummy base class of an adaptive filter. This class doesn't compute
    anything. It merely stores values in a buffer. It is used as a template
    for all other algorithms.
    '''
    
    def __init__(self, length):
        
        # filter length
        self.length = length
        
        self.reset()
    
    def reset(self):
        '''
        Reset the state of the adaptive filter
        '''
        
        # index
        self.n = 0
        
        # filter
        self.w = np.zeros((self.length))
        
        # system input signal
        self.x = np.zeros((self.length))
        
        # reference signal
        self.d = np.zeros((1))
        
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
        
        self.n += 1
        
        # update buffers
        self.x[1:] = self.x[0:-1]
        self.x[0] = x_n
        self.d = d_n
        
    def name(self):
        
        return self.__class__.__name__

