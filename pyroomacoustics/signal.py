# Signal object for simulating source signals. 
# Copyright (C) 2019  Robin Scheibler, Ivan Dokmanic, Sidney Barthe
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

r'''
Signal
======

Class to generate some common source signals, or read source signals from file.
'''

from __future__ import print_function

import numpy as np

available_types = {
    'mono': {'frequency': 'Frequency of mono sound (Hz)', 't_max':'Maximum time (s)'},
    'random': {'N': 'Number of random samples'}
}

class Signal(object):

    def __init__(self, signal_type='undefined', fs=8000, **kwargs):
        self.type = signal_type
        self.params = kwargs
        self.fs = fs
        for param, definition in available_types[self.type].items():
            assert self.params.get(param), f'Need to provide {param}:{definition} for {self.type}.'

        self.samples = self.generate_samples()

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return f'Signal instance of type {self.type}, with parameters {self.params}'

    def generate_samples(self):
        if self.type == 'undefined':
            samples = []
        elif self.type == 'mono':
            t_max = self.params.get('t_max')
            N = int(t_max * self.fs)
            f = self.params.get('frequency')
            times = np.linspace(0, t_max, N)
            samples = np.sin(2 * np.pi * f * times)
        elif self.type == 'random':
            samples = np.random.randn(self.params.get('N'))
        return samples

    def convolve(self, h):
        from scipy.signal import fftconvolve
        return fftconvolve(h, self.samples)
