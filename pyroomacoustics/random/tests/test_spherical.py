import numpy as np
from numpy.random import default_rng

import pyroomacoustics as pra


def test_stat_prop_power_spherical():

    rng = default_rng()

    for d in range(2, 4):

        pass


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as axes3d
    import numpy as np

    # let's visualize samples in the sphere

    theta, phi = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
    THETA, PHI = np.meshgrid(theta, phi)
    X, Y, Z = np.sin(PHI) * np.cos(THETA), np.sin(PHI) * np.sin(THETA), np.cos(PHI)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection="3d")
    ax.plot_wireframe(X, Y, Z, linewidth=1, alpha=0.25, color="gray")

    for loc, scale in zip(np.eye(3), [10, 1, 0.1]):
        print(loc, scale)
        X, Y, Z = pra.random.power_spherical(loc=loc, scale=scale, size=1000).T

        ax.scatter(X, Y, Z, s=50)
        r"""
        ax.plot(
            *np.stack((torch.zeros_like(loc), loc)).T,
            linewidth=4,
            label="$\kappa={}$".format(scale)
        )
        """

    ax.view_init(30, 45)
    ax.tick_params(axis="both")
    plt.legend()

    plt.show()
