'''
Recursive Least Squares Family
==============================

Implementations of adaptive filters from the RLS class. These algorithms
typically have a higher computational complexity, but a faster convergence.
'''

from __future__ import division, print_function, absolute_import
import numpy as np
from .data_structures import Buffer, Powers
from .adaptive_filter import AdaptiveFilter
from .util import hankel_stride_trick

# First the classic RLS (for real numbered signals)
class RLS(AdaptiveFilter):
    '''
    Implementation of the exponentially weighted Recursive Least Squares (RLS)
    adaptive filter algorithm.

    Parameters
    ----------
    length: int
        the length of the filter
    lmbd: float, optional
        the exponential forgetting factor (default 0.999)
    delta: float, optional
        the regularization term (default 10)
    dtype: numpy type
        the bit depth of the numpy arrays to use (default np.float32)

    '''
    
    def __init__(self, length, lmbd=0.999, delta=10, dtype=np.float32):
        
        self.lmbd = lmbd
        self.lmbd_inv = 1/lmbd
        self.delta = delta
        self.x_buf_size = 10*length

        self.dtype = dtype
        
        AdaptiveFilter.__init__(self, length)   
    
        self.reset()

    def reset(self):
        '''
        Reset the state of the adaptive filter
        '''

        AdaptiveFilter.reset(self)
        
        if self.delta <= 0:
            raise ValueError('Delta should be a positive constant.')
        else:
            self.P = np.eye(self.length, dtype=self.dtype)/self.delta

        self.x_buf_len = 10 * self.length
        self.x = Buffer(length=self.x_buf_len, dtype=self.dtype)
        for i in range(self.length-1):
            self.x.push(0)
        self.d = 0.

        self.outer_buf = np.zeros((self.length, self.length), dtype=self.dtype)
        self.g = np.zeros(self.length, dtype=self.dtype)

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
        
        # update buffers
        #AdaptiveFilter.update(self, x_n, d_n)
        self.n += 1

        self.x.push(x_n)
        self.d = d_n

        x_vec = self.x.top(self.length)

        # a priori estimation error
        alpha = (self.d - np.inner(x_vec, self.w))
        
        # update the gain vector
        np.dot(self.P, x_vec*self.lmbd_inv, out=self.g)
        denom = 1 + np.inner(x_vec, self.g)
        self.g /= denom
        
        # update the filter
        self.w += alpha*self.g
        
        # update P matrix
        np.outer(self.g, np.inner(x_vec, self.P), out=self.outer_buf)
        self.P -= self.outer_buf
        self.P *= self.lmbd_inv

        # flush at regular intervals old values
        # (but not too often for efficiency)
        if self.n % self.x_buf_len == 0:
            self.x.flush(self.x.size() - self.length + 1)


class BlockRLS(RLS):
    '''
    Block implementation of the recursive least-squares (RLS) algorithm.
    The difference with the vanilla implementation is that chunks of the input
    signals are processed in batch and some savings can be made there.

    Parameters
    ----------
    length: int
        the length of the filter
    lmbd: float, optional
        the exponential forgetting factor (default 0.999)
    delta: float, optional
        the regularization term (default 10)
    dtype: numpy type
        the bit depth of the numpy arrays to use (default np.float32)
    L: int, optional
        the block size (default to length)
    '''
    
    def __init__(self, length, lmbd=0.999, delta=10, dtype=np.float32, L=None):
        
        # block size
        if L is None:
            self.block = length
        else:
            self.block = int(L) 
        
        RLS.__init__(self, length, lmbd=lmbd, delta=delta, dtype=dtype) 
        self.reset()
        
    def reset(self):
        '''
        Reset the state of the adaptive filter
        '''
        RLS.reset(self)

        # We need to redefine these two guys
        self.d = Buffer(length=self.block)
        self.x = Buffer(length=self.length + self.block - 1)
        for i in range(self.length-1):
            self.x.push(0)

        # Precompute the powers of lambda
        self.lmbd_pwr = Powers(self.lmbd, length=2*self.block, dtype=self.dtype)
        self.lmbd_inv_pwr = Powers(1. / self.lmbd, length=2*self.block, dtype=self.dtype)
        
        
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

        # Push new data in buffers
        self.x.push(x_n)
        self.d.push(d_n)
        
        # Block update
        if self.n % self.block == 0:

            x_vec = self.x.top(self.block + self.length - 1)
            d_vec = self.d.top(self.block)
            
            # The Hankel data matrix
            X = hankel_stride_trick(x_vec, (self.block, self.length))

            # Compute the error term
            alpha = d_vec - np.dot(X, self.w)
            
            # Compute gain vector
            pi = np.dot(self.P, X.T)
            g = np.linalg.solve((np.diag(self.lmbd_pwr[self.block:0:-1]) + np.dot(X,pi)).T, pi.T).T
            
            # Update filter
            self.w += np.dot(g, alpha)
            
            # Update inverse matrix
            self.P = self.P - np.dot(g, pi.T)
            self.P *= self.lmbd_inv_pwr[self.block]  # * 1 / lmbd**self.block
            
            # Remember a few values
            self.x.flush(self.block)

