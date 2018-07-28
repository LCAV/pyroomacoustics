# @version: 1.0  2018, Juan Azcarreta
from unittest import TestCase

import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

class TestWhitening(TestCase):

    def test_whitening(self):
        # Create multivariate distribution
        mean = [0, 0]               # zero mean
        covx = [[100, 5], [5, 2]]   # diagonal covariance positive-semidefinite, e.g., covx = [[100, 5], [5, 2]]
        samples = 5000

        # Test the input
        assert np.all(np.linalg.eigvalsh(covx) >= 0), "Covariance matrix is not positive-semidefinite"
        dimensions = len(covx)      # should be equal to two
        assert dimensions == 2, "The number of dimensions must be equal to two"
        X = np.zeros([samples,1,dimensions])
        # Create multivariate Gaussian distribution
        x0, x1 = np.random.multivariate_normal(mean, covx, samples).T
        X[:,:,0] = x0[:,None]
        X[:,:,1] = x1[:,None]

        # Apply whitening
        Y = pra.whitening(X)

        # Test the output
        covy = np.dot(Y[:,0,:].T,np.conj(Y[:,0,:])).T/samples
        # Verify that the new correlation matrix is orthonormal
        test = ortho_group.rvs(dimensions)
        assert np.all(abs(np.dot(test,np.conj(test).T) - covy) < 1E-10), "Whitening unsuccessful"
        y0 = Y[:,:,0]
        y1 = Y[:,:,1]

        # Plot the input
        plt.subplot(1, 2, 1)
        plt.plot(x0, x1, 'x')
        plt.title('Correlated multivariate distribution')

        # Plot the output
        plt.subplot(1, 2, 2)
        plt.plot(y0, y1, 'rx')
        plt.title('Whitened multivariate distribution')
        plt.show()

if __name__ == '__main__':
    TestCase()
