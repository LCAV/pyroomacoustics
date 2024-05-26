import numpy as np


class DirectionVector(object):
    """
    Object for representing direction vectors in 3D, parameterized by an azimuth and colatitude
    angle.

    Parameters
    ----------
    azimuth : float
    colatitude : float, optional
        Default to PI / 2, only XY plane.
    degrees : bool
        Whether provided values are in degrees (True) or radians (False).
    """

    def __init__(self, azimuth, colatitude=None, degrees=True):
        if degrees is True:
            azimuth = np.radians(azimuth)
            if colatitude is not None:
                colatitude = np.radians(colatitude)
        self._azimuth = azimuth
        if colatitude is None:
            colatitude = np.pi / 2
        assert colatitude <= np.pi and colatitude >= 0

        self._colatitude = colatitude

        self._unit_v = np.array(
            [
                np.cos(self._azimuth) * np.sin(self._colatitude),
                np.sin(self._azimuth) * np.sin(self._colatitude),
                np.cos(self._colatitude),
            ]
        )

    def get_azimuth(self, degrees=False):
        if degrees:
            return np.degrees(self._azimuth)
        else:
            return self._azimuth

    def get_colatitude(self, degrees=False):
        if degrees:
            return np.degrees(self._colatitude)
        else:
            return self._colatitude

    @property
    def unit_vector(self):
        """Direction vector in cartesian coordinates."""
        return self._unit_v
