from __future__ import division, print_function

import numpy as np

class Buffer:
    '''
    A simple buffer class with amortized cost

    Parameters
    ----------
    length: int
        buffer length
    dtype: numpy.type
        data type
    '''

    def __init__(self, length=20, dtype=np.float64):

        self.buf = np.zeros(length, dtype=dtype)
        self.len = length
        self.head = self.len

    def push(self, val):
        ''' Add one element at the front of the buffer '''

        # Increase size if the buffer is too small
        if self.head == 0:
            self.buf = np.concatenate((np.zeros(self.len, dtype=self.buf.dtype), self.buf))
            self.head += self.len
            self.len *= 2

        # store value at head
        self.buf[self.head-1] = val

        # move head to next free spot
        self.head -= 1

    def top(self, n):
        ''' Returns the n elements at the front of the buffer from newest to oldest '''

        return self.buf[self.head:self.head+n]

    def flush(self, n):
        ''' Removes the n oldest elements in the buffer '''

        if n > self.len - self.head:
            n = self.len - self.head

        new_head = self.head + n

        # copy the remaining items to the right
        self.buf[new_head:] = self.buf[self.head:-n]

        # move head
        self.head = new_head

    def size(self):
        ''' Returns the number of elements in the buffer '''
        return self.len - self.head

    def __getitem__(self, r):
        ''' Allows to retrieve element at a specific position '''

        # create a view that starts at head
        ptr = self.buf[self.head:]

        # returned desired range
        return ptr[r]

    def __repr__(self):

        if self.head == self.len:
            return '[]'
        else:
            return str(self.buf[self.head:])

class Powers:
    '''
    This class allows to store all powers of a small number
    and get them 'a la numpy' with the bracket operator.
    There is automatic increase when new values are requested

    Parameters
    ----------
    a: float
        the number
    length: int
        the number of integer powers
    dtype: numpy.type, optional
        the data type (typically np.float32 or np.float64)

    Example
    -------

        >>> an = Powers(0.5)
        >>> print(an[4])
        0.0625

    '''

    def __init__(self, a, length=20, dtype=np.float64):

        self.a = dtype(a)
        self.pwr = self.a ** np.arange(length)

    def __getitem__(self, r):

        # find maximum power requested
        if isinstance(r, int):
            high = r + 1
        elif isinstance(r, slice):
            high = r.stop
        elif isinstance(r, list):
            high = max(r) + 1
        else:
            high = int(r + 1)

        # Compute it if needed
        if high > self.pwr.shape[0]:
            self.pwr = np.concatenate((self.pwr, self.a**np.arange(self.pwr.shape[0], high)))

        return self.pwr[r]

    def __repr__(self):
        return str(self.pwr)


class CoinFlipper:
    '''
    This class efficiently generates large number of coin flips.
    Because each call to ``numpy.random.rand`` is a little bit costly,
    it is more efficient to generate many values at once.
    This class does this and stores them in advance. It generates
    new fresh numbers when needed.

    Parameters
    ----------
    p: float, 0 < p < 1
        probability to output a 1
    length: int
        the number of flips to precompute
    '''

    def __init__(self, p, length=10000):

        self.p = p
        self.length = length
        self.buffer = np.random.random(length) < p
        self.dirty_coins = 0

    def fresh_flips(self, n):
        ''' Generates n binary random values now '''

        return np.random.random(n) < self.p

    def flip_all(self):
        ''' Regenerates all the used up values '''

        remaining = self.length - self.dirty_coins
        self.buffer[:self.dirty_coins] = self.fresh_flips(self.dirty_coins)
        self.dirty_coins = 0


    def flip(self, n):
        ''' Get n random binary values from the buffer '''


        # If more flips than computed are requested
        # increase buffer size and flip again
        if n > self.length:
            self.buffer = np.pad(self.buffer, (0, 2 * n - self.length), mode='constant')
            self.buffer[self.length:] = self.fresh_flips(2 * n - self.length)
            self.length = 2 * n

        remaining = self.length - self.dirty_coins
        if remaining < n:
            self.flip_all()

        flips = self.buffer[self.dirty_coins:self.dirty_coins+n]
        self.dirty_coins += n

        return flips

