import numpy as np
from numpy.random import default_rng

_eps = 1e-7


def uniform_spherical(dim=3, size=None):
    """
    Generates uniform samples on the n-sphere

    Parameters
    ----------
    size: int or tuple of ints, optional
        The number of samples to generate
    dim: int, optional
        The number of dimensions of the sphere, the default is dim=3

    Returns
    -------
    out: ndarray, shape (*size, dim)
        The samples draw from the uniform distribution on the n-sphere
    """
    if size is None:
        size = [1, dim]
    elif isinstance(size, int):
        size = [size, dim]
    else:
        size = list(size) + [dim]

    rng = default_rng()

    out = rng.standard_normal(size=size)
    out /= np.linalg.norm(out, axis=-1, keepdims=True)

    return out


def power_spherical(loc=None, scale=None, dim=3, size=None):
    """
    Generates power spherical samples on the (n-1)-sphere according to

    Nicola De Cao, Wilker Aziz, "The Power Spherical distribution", arXiv, 2020.
    http://arxiv.org/abs/2006.04437v1

    Parameters
    ----------
    loc: float or array_like of floats, optional
        The location (i.e., direction) unit vector
    scale: float or array_like of floats
        The scale parameter descibing the spread of the distribution
    dim: int, optional
        The number of dimensions of the sphere, the default is dim=3
    size: int or tuple of ints, optional
        The number of samples to generate

    Returns
    -------
    out: ndarray, shape (*size, dim)
        The samples draw from the uniform distribution on the n-sphere
    """

    if size is None:
        size = [1, dim]
    elif isinstance(size, int):
        size = [size, dim]
    else:
        size = list(size) + [dim]

    e1 = np.zeros(dim)
    e1[0] = 1.0

    if loc is None:
        loc = e1.copy()

    if scale is None:
        scale = 1.0

    rng = default_rng()

    z = rng.beta((dim - 1.0) / 2.0 + scale, (dim - 1) / 2.0, size=size[:-1])
    v = uniform_spherical(size=size[:-1], dim=dim - 1)

    t = 2 * z[..., None] - 1
    y = np.concatenate((t, np.sqrt(1 - t**2) * v), axis=-1)  # shape (*size, dim)

    u_hat = e1 - loc
    # here the _eps is to avoid division by zero so that it is fine that when
    # e1 == loc the vector u is zero
    # this was verified in the code from the original paper
    # https://github.com/nicola-decao/power_spherical/blob/master/power_spherical/distributions.py
    u = u_hat / (np.linalg.norm(u_hat, axis=-1, keepdims=True) + _eps)

    # out = y - u * (u[..., None, :] @ y[..., :, None])[..., 0]
    out = y - 2 * u * (u * y).sum(axis=-1, keepdims=True)

    return out
