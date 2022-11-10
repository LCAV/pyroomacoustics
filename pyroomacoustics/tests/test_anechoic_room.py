import numpy as np
import pyroomacoustics as pra


def test_anechoic_room(debug=False):

    # sound signal
    fs = 40000
    freq = 440
    times = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * freq * times)

    for dim in [2, 3]:

        # create Anechoic room using the correct class
        room_infinite = pra.AnechoicRoom(fs=fs, dim=dim)

        # create same room using Shoebox
        room_shoebox = pra.ShoeBox([10] * dim, fs=fs, max_order=0)

        # Add a source somewhere in the room
        soundsource = pra.SoundSource([5] * dim, signal=signal)
        room_infinite.add_soundsource(soundsource)
        room_shoebox.add_soundsource(soundsource)

        # Create a linear array beamformer with 4 microphones
        # with angle 0 degrees and inter mic distance 10 cm
        R = pra.linear_2D_array([2, 1.5], 4, 0, 0.04)
        if dim == 3:
            R = np.r_[R, np.zeros((1, R.shape[1]))]

        beamformer = pra.Beamformer(R, fs)

        room_infinite.add_microphone_array(beamformer)
        room_shoebox.add_microphone_array(beamformer)

        # Now compute the delay and sum weights for the beamformer
        for room in [room_infinite, room_shoebox]:
            room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1])
            if debug:
                room.plot(freq=[1000, 2000, 4000, 8000], img_order=0)

        # create signal
        room_infinite.sources[0].add_signal(signal)

        # make sure that the premix signals for both are the same.
        premix_infinite = room_infinite.simulate(return_premix=True)
        premix_shoebox = room_shoebox.simulate(return_premix=True)

        if debug:
            plt.figure()
            for m in range(premix_infinite.shape[1]):
                plt.plot(
                    premix_infinite[0][m],
                    label="mic {} infinite".format(m),
                    color="C{}".format(m),
                )
                plt.plot(
                    premix_shoebox[0][m],
                    label="mic {} shoebox".format(m),
                    ls=":",
                    color="C{}".format(m),
                )
            plt.legend()
        else:
            np.testing.assert_allclose(premix_infinite, premix_shoebox)
    if debug:
        plt.show()


if __name__ == "__main__":
    import matplotlib.pylab as plt

    test_anechoic_room(debug=True)
