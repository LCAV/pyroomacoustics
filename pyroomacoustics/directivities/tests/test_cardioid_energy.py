"""
This test compares the theoretical value of the energy integral of the cardioid function
with the numerical value obtained by integrating the squared cardioid function over the
unit sphere. The test is performed for different values of the parameter p and the gain.
"""

import numpy as np
import pytest

from pyroomacoustics.directivities.analytic import (
    CardioidFamily,
    cardioid_energy,
    cardioid_func,
)
from pyroomacoustics.directivities.direction import DirectionVector
from pyroomacoustics.directivities.integration import spherical_integral
from pyroomacoustics.doa import cart2spher

PARAMETERS = [(p, G) for p in [0.0, 0.25, 0.5, 0.75, 1.0] for G in [1.0, 0.5, 2.0]]


@pytest.mark.parametrize("p,gain", PARAMETERS)
def test_cardioid_func_energy(p, gain):
    n_points = 10000
    direction = np.array([0.0, 0.0, 1.0])

    def func(points):
        return cardioid_func(points, direction, p, gain=gain) ** 2

    num = spherical_integral(func, n_points)
    thy = cardioid_energy(p, gain=gain)
    assert abs(num - thy) < 1e-4


@pytest.mark.parametrize("p,gain", PARAMETERS)
def test_cardioid_family_energy(p, gain):
    n_points = 10000
    direction = np.array([0.0, 0.0, 1.0])

    e3 = DirectionVector(0.0, 0.0)  # vector pointing up

    card_obj = CardioidFamily(e3, p, gain=gain)

    def func(points):
        az, co, _ = cart2spher(points)
        return card_obj.get_response(az, co, degrees=False) ** 2

    num = spherical_integral(func, n_points)
    thy = cardioid_energy(p, gain=gain)
    assert abs(num - thy) < 1e-4
