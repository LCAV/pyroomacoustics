# Classes for creating distributions to randomly sample.
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
from abc import ABCMeta, abstractmethod


class Distribution(object):
    """
    Abstract class for distributions.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def sample(self):
        pass


class UniformDistribution(Distribution):
    """

    Create a uniform distribution between two values.

    Parameters
    -------------
    vals_range : tuple / list
        Tuple or list of two values, (lower bound, upper bound).

    """
    def __init__(self, vals_range):
        super(UniformDistribution, self).__init__()
        assert len(vals_range) == 2, 'Length of `vals_range` must be 2.'
        assert vals_range[0] <= vals_range[1], '`vals_range[0]` must be ' \
                                               'less than or equal to ' \
                                               '`vals_range[1]`.'
        self.vals_range = vals_range

    def sample(self):
        return np.random.uniform(self.vals_range[0], self.vals_range[1])


class MultiUniformDistribution(Distribution):
    """

    Sample from multiple uniform distributions.

    Parameters
    ------------
    ranges : list of tuples / lists
        List of tuples / lists, each with two values.

    """
    def __init__(self, ranges):
        super(MultiUniformDistribution, self).__init__()
        self.distributions = [UniformDistribution(r) for r in ranges]

    def sample(self):
        return [d.sample() for d in self.distributions]


class DiscreteDistribution(Distribution):
    """

    Create a discrete distribution which samples from a given set of values
    and (optionally) a given set of probabilities.

    Parameters
    ------------
    values : list
        List of values to sample from.
    prob : list
        Corresponding list of probabilities. Default to equal probability for
        all values.

    """
    def __init__(self, values, prob=None):
        super(DiscreteDistribution, self).__init__()
        if prob is None:
            prob = np.ones_like(values)
        assert len(values) == len(prob), \
            'len(values)={}, len(prob)={}'.format(len(values), len(prob))
        self.values = values
        self.prob = np.array(prob) / float(sum(prob))

    def sample(self):
        return np.random.choice(self.values, p=self.prob)


class MultiDiscreteDistribution(Distribution):
    """

    Sample from multiple discrete distributions.

    Parameters
    ------------
    ranges : list of tuples / lists
        List of tuples / lists, each with two values.

    """
    def __init__(self, values_list, prob_list=None):
        super(MultiDiscreteDistribution, self).__init__()
        if prob_list is not None:
            assert len(values_list) == len(prob_list), \
                'Lengths of `values_list` and `prob_list` must match.'
        else:
            prob_list = [None] * len(values_list)
        self.distributions = [
            DiscreteDistribution(values=tup[0], prob=tup[1])
            for tup in zip(values_list, prob_list)
        ]

    def sample(self):
        return [d.sample() for d in self.distributions]

