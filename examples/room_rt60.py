'''
Room RT60
---------

In this example, we create a room with a pre-set reverberation time
(according to Sabine's formula), and then check how the simulated RIR
verifies the prediction.

### Example

Simulate a ``10 x 7.5 x 3.2`` room with ``RT60 = 0.5 s``

    python examples/room_rt60.py 10 7.5 3.2 0.5

'''
import argparse
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

def rt60_analysis(room, mic, src, plot=True, rt60_tgt=None):
    '''
    Plot and analyze the RT60 of one of the RIR in the room.
    '''

    # Remove coefficients that might cause problems with the log
    h = room.rir[mic][src]
    T = np.arange(len(h)) / room.fs
    I = h > 0

    # The power of the impulse response in dB
    power = 20 * np.log10(np.abs(h[I]))
    T = T[I]

    # Adjust to location and amplitude of peak power
    i_max = np.argmax(power)
    power -= power[i_max]
    power = power[i_max:]
    T = T[i_max:]

    # linear fit
    a, b = np.polyfit(T, power, 1)
    est_rt60 = - 60 / a

    if plot:
        # show result
        plt.plot(T, power)
        plt.plot(T, a * T + b, '--')
        plt.plot(T, np.ones_like(T) * -60, '--')
        plt.vlines(est_rt60, np.min(power), 0, linestyles='dashed')

        if rt60_tgt is not None:
            plt.vlines(rt60_tgt, np.min(power), 0)

        plt.legend(['RIR', 'Lin Fit', '-60 dB', 'Estimated T60', 'Target T60',])

        plt.show()

    return est_rt60

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Explore the parameters and RT60 of rooms')
    parser.add_argument('dim', type=float, nargs='+',
            help='The room dimensions')
    parser.add_argument('rt60', type=float,
            help='The desired reverberation time')
    args = parser.parse_args()

    # Define room dimensions and RT60
    room_dim = args.dim
    rt60 = args.rt60

    if len(room_dim) not in [2,3]:
        raise ValueError('The room dimension must be a pair or triplet of numbers')

    mic_loc = 0.33 * np.array(room_dim)
    src_loc = 0.66 * np.array(room_dim)

    # we just do this to avoid some probability zero
    # placements with special artefacts
    mic_loc += 0.0005 * np.random.randn(len(room_dim))
    src_loc += 0.0005 * np.random.randn(len(room_dim))

    # Create the room and place equipment in it
    room = pra.ShoeBox(
            room_dim,
            fs=16000,
            rt60=rt60,
            )
    room.add_source(src_loc)
    room.add_microphone_array(
            pra.MicrophoneArray(
                np.c_[ mic_loc, ],
                room.fs,
                )
            )

    # Simulate and analyze
    room.compute_rir()
    rt60_analysis(room, 0, 0, rt60_tgt=rt60)
