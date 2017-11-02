
import numpy as np
from scipy.stats import norm as _norm
import pyroomacoustics as pra

def test_median():

    # simple tests
    x = np.arange(1, 11)
    m = pra.median(x)
    assert m == 5.5

    x = np.arange(1, 12)
    m = pra.median(x)
    assert m == 6

    # test dimensions
    x = np.random.rand(10,9,8)

    m, ci = pra.median(x, alpha=0.05)
    assert m.shape == (10,9)
    assert ci.shape == (2,10,9)

    m, ci = pra.median(x, alpha=0.05, axis=1)
    assert m.shape == (10,8)
    assert ci.shape == (2,10,8)

    m, ci = pra.median(x, alpha=0.05, axis=1, keepdims=True)
    assert m.shape == (10,1,8)
    assert ci.shape == (2,10,1,8)


    # Now test statistical property of the confidence interval
    np.random.seed(0)  # fix this for repeatable outcome
    R = 200
    N = [10, 100, 1000]
    alpha = [0.05, 0.01]

    from scipy.stats import uniform
    dist = uniform()
    true_median = dist.median()

    for n in N:
        for a in alpha:

            failures = np.zeros(R, dtype=bool)

            for r in range(R):
                x = dist.rvs(size=n)
                m,ci = pra.median(x, alpha=a)

                if true_median < m+ci[0] or m+ci[1] < true_median:
                    failures[r] = 1

            assert sum(failures) / R <= 1.2 * a  # allow some slack for realizations


if __name__ == '__main__':

    test_median()


