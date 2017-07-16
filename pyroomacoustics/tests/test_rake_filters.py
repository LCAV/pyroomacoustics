
import numpy as np
import pyroomacoustics as pra

room = pra.ShoeBox([4,6], fs=16000, max_order=1)

# add sources in the room
room.add_source([2, 1.5])  # nice source
room.add_source([2,4.5])   # interferer

# add a circular beamforming array
shape = pra.circular_2D_array([2.5,3], 8, 0., 0.15)
bf = pra.Beamformer(shape, room.fs, Lg=500)
room.add_microphone_array(bf)

# run the ISM
room.image_source_model()

# the noise matrix, note that the size is the number of
# sensors multiplied by the filter size
Rn = np.eye(bf.M * bf.Lg) * 1e-5

def test_rake_max_udr_filters():
    # no interferer
    bf.rake_max_udr_filters(room.sources[0][:4], R_n=Rn, delay=0.015, epsilon=1e-2)
    # with interferer
    bf.rake_max_udr_filters(room.sources[0][:4], interferer=room.sources[1][:4], R_n=Rn, delay=0.015, epsilon=1e-2)

def test_perceptual_filters():
    # no interferer
    bf.rake_perceptual_filters(room.sources[0][:4], R_n=Rn)
    # with interferer
    bf.rake_perceptual_filters(room.sources[0][:4], interferer=room.sources[1][:4], R_n=Rn)
    pass

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    bf.rake_perceptual_filters(room.sources[0][:4], interferer=room.sources[1][:4], R_n=Rn, epsilon=0.1)
    bf.plot()

    room.plot(img_order=1, freq=[700., 1000., 2000.])

    plt.show()
