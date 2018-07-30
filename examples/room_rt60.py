'''
Room RT60
---------

In this example, we create a room with a pre-set reverberation time
(according to Sabine's formula), and then check how the simulated RIR
verifies the prediction.
'''
import math, itertools
import numpy as np
from scipy.signal import fftconvolve
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


# Define room dimensions and RT60
room_dim = [10, 7.5, 3.6]
rt60 = 0.3
c = 343

# Create the room and place equipment in it
room = pra.ShoeBox(
        room_dim,
        fs=16000,
        rt60=rt60,
        )
room.add_source([2.2, 3.3, 1.75])
room.add_microphone_array(
        pra.MicrophoneArray(
            np.c_[
                [4.5, 6.2, 1.6],
                [4.5, 6.25, 1.6],
                ],
            room.fs,
            )
        )

# Simulate and analyze
room.compute_rir()
rt60_analysis(room, 1, 0, rt60_tgt=rt60)
