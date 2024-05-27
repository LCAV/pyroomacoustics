import abc

from .direction import DirectionVector, Rotation3D


class Directivity(abc.ABC):
    """
    Abstract class for directivity patterns.
    """

    @property
    @abc.abstractmethod
    def is_impulse_response(self):
        """
        Indicates whether the array contains coefficients for octave bands
        (returns ``False``) or is a full-size impulse response (returns
        ``True``).
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def filter_len_ir(self):
        """
        When ``is_impulse_response`` returns ``True``, this property returns the
        lengths of the impulse responses returned.
        All impulse responses are assumed to have the same length.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_response(
        self, azimuth, colatitude=None, magnitude=False, frequency=None, degrees=True
    ):
        """
        Get response for provided angles and frequency.
        """
        raise NotImplementedError
