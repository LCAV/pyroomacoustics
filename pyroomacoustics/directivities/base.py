import abc

from .direction import DirectionVector


class Directivity(abc.ABC):
    """
    Abstract class for directivity patterns.

    """

    def __init__(self, orientation):
        assert isinstance(orientation, DirectionVector)
        self._orientation = orientation

    @property
    def is_impulse_response(self):
        """
        Indicates whether the array returned has coefficients
        for octave bands or is a full-size impulse response
        """
        return False

    def get_azimuth(self, degrees=True):
        return self._orientation.get_azimuth(degrees, degrees=degrees)

    def get_colatitude(self, degrees=True):
        return self._orientation.get_colatitude(degrees, degrees=degrees)

    def set_orientation(self, orientation):
        """
        Set orientation of directivity pattern.

        Parameters
        ----------
        orientation : DirectionVector
            New direction for the directivity pattern.
        """
        assert isinstance(orientation, DirectionVector)
        self._orientation = orientation

    @abc.abstractmethod
    def get_response(
        self, azimuth, colatitude=None, magnitude=False, frequency=None, degrees=True
    ):
        """
        Get response for provided angles and frequency.
        """
        return
