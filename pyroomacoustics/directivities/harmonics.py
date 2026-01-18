import numpy as np
import scipy.special

from .base import Directivity


def get_mn_in_acn_order(order):
    """Calculates the (m,n) pairs in ACN order up to a given order.

    Parameters:
    order : int
        Maximum degree of the spherical harmonics.

    Returns:
    all_m : ndarray
        Array of orders m in ACN order.
    all_n : ndarray
        Array of degrees n in ACN order.
    """
    all_m = np.array([j - i for i in range(0, order + 1) for j in range(0, 2 * i + 1)])
    all_n = np.array([i for i in range(0, order + 1) for _ in range(0, 2 * i + 1)])
    return all_m, all_n


def real_sph_harm(n, m, theta, phi, condon_shortley_phase=False):
    """Calculates the real spherical harmonics.

    Parameters:
    n : int
        Degree of the spherical harmonic.
    m : int
        Order of the spherical harmonic.
    theta : array_like
        Polar (colatitudinal) coordinate in radians.
    phi : array_like
        Azimuthal coordinate in radians.
    condon_shortley_phase : bool, optional
        If True, includes the Condon-Shortley phase factor (-1)^m. Default is False.

    Returns:
    y_real : ndarray
        Real spherical harmonics evaluated at the given angles.
    """
    m = np.atleast_1d(m)

    try:
        ysh_complx = scipy.special.sph_harm_y(n, m, theta, phi)
    except AttributeError:
        # Deprecated since scipy v1.15.0.
        ysh_complx = scipy.special.sph_harm(m, n, phi, theta)

    y_real = np.empty_like(ysh_complx, dtype=np.float64)
    y_real[(m >= 0)] = np.real(ysh_complx[(m >= 0)])
    y_real[(m < 0)] = np.imag(ysh_complx[(m < 0)])

    if not condon_shortley_phase:
        # Cancel Condon-Shortley Phase (term (-1) ** m) by multiplying with (-1) ** m
        # In Rafaely's book, this step is not done
        y_real[(m != 0)] *= np.sqrt(2) * np.array([-1.0]) ** m[(m != 0)]
        # In the formular, for m<0, |m| is used. We consider this by the use of the constraint.
        y_real[np.logical_and(m < 0, (m % 2) == 0)] = -y_real[
            np.logical_and(m < 0, (m % 2) == 0)
        ]

    return y_real


class RealSHDirectivity(Directivity):
    """
    A class for real spherical harmonic directivity patterns.

    Parameters
    ----------
    m: int
        Order of the spherical harmonic.
    n: int
        Degree of the spherical harmonic.
    condon_shortley_phase: bool, optional
        If True, includes the Condon-Shortley phase factor (-1)^m. Default is False.
    """

    def __init__(self, m, n, condon_shortley_phase: bool = False):
        self.m = m
        self.n = n

        self.condon_shortley_phase = condon_shortley_phase

    @property
    def is_impulse_response(self):
        return True

    @property
    def filter_len_ir(self):
        return 1

    def get_response(
        self, azimuth, colatitude=None, magnitude=False, frequency=None, degrees=True
    ):

        if degrees:
            azimuth = np.radians(azimuth)
            colatitude = np.radians(colatitude)
        return real_sph_harm(
            self.m,
            self.n,
            colatitude,
            azimuth,
            condon_shortley_phase=self.condon_shortley_phase,
        )[:, np.newaxis]
