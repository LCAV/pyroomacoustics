import numpy as np
import pyroomacoustics as pra


def test_cart_spher_convertion(repetitions=10, vectorized=True):
    """Tests that the convertion is invertible"""

    np.random.seed(124584)

    for epoch in range(repetitions):
        if vectorized:
            cart = np.random.randn(3, 10)
        else:
            cart = np.random.randn(3)

        sph = np.array(pra.doa.cart2spher(cart))
        cart_back = pra.doa.spher2cart(*sph)
        sph_back = np.array(pra.doa.cart2spher(cart_back))

        err1 = np.max(np.abs(sph_back - sph))
        err2 = np.max(np.abs(cart_back - cart))

        assert err1 < 1e-13
        assert err2 < 1e-13


if __name__ == "__main__":
    test_cart_spher_convertion(vectorized=True)
    test_cart_spher_convertion(vectorized=False)
