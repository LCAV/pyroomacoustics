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
import math, itertools, argparse
import numpy as np
from scipy.signal import fftconvolve
import pyroomacoustics as pra
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Explore the parameters and RT60 of rooms')
    parser.add_argument('dim', type=float, nargs='+',
            help='The room dimensions')
    parser.add_argument('rt60', type=float,
            help='The desired reverberation time')
    parser.add_argument('-r', '--repeat', metavar='N', type=int, default=1,
            help='Repeat the measurement for N source locations and show a histogram')
    args = parser.parse_args()

    # Define room dimensions and RT60
    room_dim = np.array(args.dim)
    rt60 = args.rt60
    repeat = args.repeat

    if len(room_dim) not in [2,3]:
        raise ValueError('The room dimension must be a pair or triplet of numbers')


    rt60_lst = []

    for n in range(repeat):

        mic_loc = np.random.rand(len(room_dim)) * room_dim
        src_loc = np.random.rand(len(room_dim)) * room_dim

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
        rt60_lst.append(
                pra.experimental.measure_rt60(
                    room.rir[0][0],
                    fs=room.fs,
                    plot=(repeat == 1),
                    rt60_tgt=rt60,
                    )
                )

    if repeat > 1:
        plt.hist(rt60_lst)
        plt.show()
