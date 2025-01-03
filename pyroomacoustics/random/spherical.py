import numpy as np

_eps = 1e-7


def uniform_spherical(dim=3, size=None, rng=None):
    """
    Generates uniform samples on the n-sphere

    Parameters
    ----------
    size: int or tuple of ints, optional
        The number of samples to generate
    dim: int, optional
        The number of dimensions of the sphere, the default is dim=3
    rng: numpy.random.Generator or None
        A numpy.random.Generator object or None. If None, numpy.random.default_rng
        is used to obtain a Generator object.

    Returns
    -------
    out: ndarray, shape (*size, dim)
        The samples draw from the uniform distribution on the n-sphere
    """
    if size is None:
        size = [dim]
    elif isinstance(size, int):
        size = [size, dim]
    else:
        size = list(size) + [dim]

    if rng is None:
        rng = np.random.default_rng()

    out = rng.standard_normal(size=size)
    out /= np.linalg.norm(out, axis=-1, keepdims=True)

    return out


def power_spherical(loc=None, scale=None, size=None, rng=None):
    """
    Generates power spherical samples on the (n-1)-sphere according to

    Nicola De Cao, Wilker Aziz, "The Power Spherical distribution", arXiv, 2020.
    http://arxiv.org/abs/2006.04437v1

    Parameters
    ----------
    loc: float or array_like of floats, optional
        The location (i.e., direction) unit vector. If None, then
        ``loc = np.array([1.0, 0.0, 0.0])`` is used.
    scale: float or array_like of floats
        The scale parameter descibing the spread of the distribution
    size: int or tuple of ints, optional
        The number of samples to generate
    rng: numpy.random.Generator or None
        A numpy.random.Generator object or None. If None, numpy.random.default_rng
        is used to obtain a Generator object.

    Returns
    -------
    out: ndarray, shape (*size, dim)
        The samples draw from the uniform distribution on the n-sphere
    """
    if loc is None:
        loc = np.array([1.0, 0.0, 0.0])
    else:
        loc = np.array(loc)

    if loc.ndim != 1:
        raise ValueError(f"The location should be a 1d array (got {loc.shape=})")

    e1 = np.zeros_like(loc)
    e1[0] = 1.0

    dim = len(loc)
    if size is None:
        size = [dim]
    elif isinstance(size, int):
        size = [size, dim]
    else:
        size = list(size) + [dim]

    if scale is None:
        scale = 1.0

    if rng is None:
        rng = np.random.default_rng()

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
