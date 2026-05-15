import numpy as np

import pyroomacoustics as pra


def test_spiral_2d_array_center():
    """Test that spiral_2D_array correctly uses the center parameter."""

    # Test with fixed angle to avoid random variance
    angle = 0.3

    # Array at origin
    arr0 = pra.spiral_2D_array([0, 0], 10, radius=1.0, divi=3, angle=angle)

    # Same array shifted
    center = np.array([2.5, -1.5])
    arr_shifted = pra.spiral_2D_array(center, 10, radius=1.0, divi=3, angle=angle)

    # Every point should be shifted by exactly the center offset
    np.testing.assert_allclose(arr_shifted[0] - arr0[0], center[0])
    np.testing.assert_allclose(arr_shifted[1] - arr0[1], center[1])

    # Test single point (M=1) — should be at center
    arr_single = pra.spiral_2D_array([5.0, -3.0], 1, radius=2.0, angle=0.0)
    np.testing.assert_allclose(arr_single.ravel(), [5.0, -3.0])

    # Test that center as list also works
    arr_list = pra.spiral_2D_array([1.0, 2.0], 4, radius=0.5, divi=2, angle=0.0)
    assert arr_list.shape == (2, 4)
