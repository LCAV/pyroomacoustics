import numpy as np
import pytest
from scipy.signal import lfilter

import pyroomacoustics as pra


@pytest.fixture(params=[None, 1, 4])
def adaptive_filter(request):
    if request.param is None:
        return pra.adaptive.NLMS(8, mu=0.5)
    return pra.adaptive.BlockLMS(8, mu=0.5, L=request.param, nlms=True)


def test_silent_input_preserves_weights(adaptive_filter):
    weights = np.linspace(-0.5, 0.5, 8)
    adaptive_filter.w[:] = weights
    with np.errstate(divide="raise", invalid="raise"):
        for _ in range(16):
            adaptive_filter.update(0.0, 1.0)
    np.testing.assert_array_equal(adaptive_filter.w, weights)


def test_adaptation_recovers_after_silence(adaptive_filter):
    rng = np.random.default_rng(0)
    target = np.array([0.5, -0.2, 0.1, 0, 0, 0, 0, 0])
    signal = np.concatenate(
        [np.zeros(32), rng.normal(size=1000), np.zeros(32), rng.normal(size=1000)]
    )
    desired = lfilter(target, [1.0], signal)
    with np.errstate(divide="raise", invalid="raise"):
        for sample, reference in zip(signal, desired):
            adaptive_filter.update(sample, reference)
    np.testing.assert_allclose(adaptive_filter.w, target, atol=1e-8)


def test_normalized_block_uses_nonzero_rows():
    adaptive_filter = pra.adaptive.BlockLMS(2, mu=0.5, L=4, nlms=True)
    # Only two of the four input vectors carry energy: [1, 0] and [2, 1].
    with np.errstate(divide="raise", invalid="raise"):
        for sample in [0.0, 0.0, 1.0, 2.0]:
            adaptive_filter.update(sample, sample)
    np.testing.assert_allclose(adaptive_filter.w, [0.9, 0.2])
