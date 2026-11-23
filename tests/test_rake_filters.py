import numpy as np
import pytest

import pyroomacoustics as pra

room = pra.ShoeBox([4, 6], fs=16000, max_order=1)

# add sources in the room
room.add_source([2, 1.5])  # nice source
room.add_source([2, 4.5])  # interferer

# add a circular beamforming array
shape = pra.circular_2D_array([2.5, 3], 8, 0.0, 0.15)
bf = pra.Beamformer(shape, room.fs, Lg=500)
room.add_microphone_array(bf)

# run the ISM
room.image_source_model()

# the noise matrix, note that the size is the number of
# sensors multiplied by the filter size
Rn = np.eye(bf.M * bf.Lg) * 1e-5


@pytest.mark.parametrize("attn", [False, True])
def test_far_field_steering(attn):
    array = pra.Beamformer(np.array([[2.0, 2.3, 2.1], [3.0, 3.1, 3.4]]), 16000)
    directions = np.array([[0.6, -0.8], [0.8, 0.6]])
    ranges = np.array([8.0, 13.0])
    sources = array.center + directions * ranges
    frequency = 750
    steering = array.steering_vector_2D_from_point
    result = steering(frequency, sources, ff=True, attn=attn)
    for i in range(sources.shape[1]):
        projection = array.R.T @ directions[:, i]
        delay = (projection.max() - projection) / pra.constants.get("c")
        expected = np.exp(-2j * np.pi * frequency * delay)
        if attn:
            expected /= 4 * np.pi * ranges[i]
        np.testing.assert_allclose(result[:, i], expected)
        single = steering(frequency, sources[:, i], ff=True, attn=attn)
        np.testing.assert_allclose(result[:, i], single[:, 0])


def test_far_field_source_at_center():
    with pytest.raises(ValueError, match="array center"):
        bf.steering_vector_2D_from_point(750, bf.center, ff=True)


def test_rake_max_udr_filters():
    # no interferer
    bf.rake_max_udr_filters(room.sources[0][:4], R_n=Rn, delay=0.015, epsilon=1e-2)
    # with interferer
    bf.rake_max_udr_filters(
        room.sources[0][:4],
        interferer=room.sources[1][:4],
        R_n=Rn,
        delay=0.015,
        epsilon=1e-2,
    )


def test_perceptual_filters():
    # no interferer
    bf.rake_perceptual_filters(room.sources[0][:4], R_n=Rn)
    # with interferer
    bf.rake_perceptual_filters(
        room.sources[0][:4], interferer=room.sources[1][:4], R_n=Rn
    )
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    bf.rake_perceptual_filters(
        room.sources[0][:4], interferer=room.sources[1][:4], R_n=Rn, epsilon=0.1
    )
    bf.plot()

    room.plot(img_order=1, freq=[700.0, 1000.0, 2000.0])

    plt.show()
