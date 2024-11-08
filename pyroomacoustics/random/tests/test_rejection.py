import pyroomacoustics as pra
from pyroomacoustics.directivities import CardioidFamilySampler

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as axes3d
    import numpy as np

    # let's visualize samples in the sphere

    theta, phi = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
    THETA, PHI = np.meshgrid(theta, phi)
    X, Y, Z = np.sin(PHI) * np.cos(THETA), np.sin(PHI) * np.sin(THETA), np.cos(PHI)

    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection="3d")
    ax.plot_wireframe(X, Y, Z, linewidth=1, alpha=0.25, color="gray")
    """

    for loc, scale in zip([[1, 1, 1]], [10, 1, 0.1]):
        print(loc, scale)
        # Figure-of-eight
        sampler = CardioidFamilySampler(loc=loc, p=0)

        points = sampler(size=10000).T  # shape (n_dim, n_points)

        # Create a spherical histogram
        hist = pra.doa.SphericalHistogram(n_bins=500)
        hist.push(points)
        hist.plot()

        """
        ax.scatter(X, Y, Z, s=50)
        ax.plot(
            *np.stack((torch.zeros_like(loc), loc)).T,
            linewidth=4,
            label="$\kappa={}$".format(scale)
        )
        """

        print("Sampler's efficiency:", sampler.efficiency)

    """
    ax.view_init(30, 45)
    ax.tick_params(axis="both")
    plt.legend()
    """

    plt.show()
