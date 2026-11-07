import numpy as np
import pytest

from pyroomacoustics.directivities import (
    Cardioid,
    CardioidFamily,
    DirectionVector,
    FigureEight,
    HyperCardioid,
    Omnidirectional,
    SubCardioid,
)


@pytest.mark.parametrize(
    "azimuth_deg, colatitude_deg", [(0.0, 90.0), (30.0, 60.0), (-120.0, 135.0)]
)
def test_cardioid_get_orientation(azimuth_deg, colatitude_deg):
    # get_azimuth / get_colatitude used to raise TypeError because they passed
    # the `degrees` flag twice to DirectionVector
    orientation = DirectionVector(azimuth_deg, colatitude_deg, degrees=True)
    d = CardioidFamily(orientation, p=0.5)

    assert d.get_azimuth() == pytest.approx(azimuth_deg)
    assert d.get_colatitude() == pytest.approx(colatitude_deg)
    assert d.get_azimuth(degrees=True) == pytest.approx(azimuth_deg)
    assert d.get_colatitude(degrees=True) == pytest.approx(colatitude_deg)
    assert d.get_azimuth(degrees=False) == pytest.approx(np.radians(azimuth_deg))
    assert d.get_colatitude(degrees=False) == pytest.approx(np.radians(colatitude_deg))


def test_cardioid_get_orientation_after_set():
    d = CardioidFamily(DirectionVector(0.0, 90.0, degrees=True), p=0.5)
    d.set_orientation(DirectionVector(45.0, 30.0, degrees=True))

    assert d.get_azimuth() == pytest.approx(45.0)
    assert d.get_colatitude() == pytest.approx(30.0)


@pytest.mark.parametrize(
    "cls", [SubCardioid, Cardioid, HyperCardioid, FigureEight, Omnidirectional]
)
def test_cardioid_subclasses_get_orientation(cls):
    kwargs = {} if cls is Omnidirectional else dict(orientation=[0.0, 1.0, 0.0])
    d = cls(**kwargs)
    # must not raise
    az = d.get_azimuth()
    col = d.get_colatitude()
    assert np.isfinite(az) and np.isfinite(col)
    if cls is not Omnidirectional:
        assert az == pytest.approx(90.0)
        assert col == pytest.approx(90.0)
