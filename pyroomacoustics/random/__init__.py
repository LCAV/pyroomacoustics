"""
Random
======

This sub-package provides classes and methods in order to randomly generate
objects rooms of e.g. various shapes, microphone and source placement, and
reverberation properties.

"""

from pyroomacoustics.random.microphone import *
from pyroomacoustics.random.room import ShoeBoxRoomGenerator
from pyroomacoustics.random.distribution import UniformDistribution, \
    MultiUniformDistribution, DiscreteDistribution, MultiDiscreteDistribution