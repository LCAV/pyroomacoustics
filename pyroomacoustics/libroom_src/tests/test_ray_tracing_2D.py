import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.utilities import fractional_delay
import matplotlib.pyplot as plt
import time
import scipy
from scipy.io import wavfile
from scipy import signal
from pyroomacoustics import build_rir

wall_corners_L_shape = [
    np.array([  # left
        [0, 0],
        [0,5],
    ]),
    np.array([  # back
        [0, 5],
        [5, 5],
    ]),
    np.array([  # up right
        [5, 5],
        [5, 3],
    ]),
    np.array([  # horizontal right
        [5, 3],
        [3, 3],
    ]),
    np.array([  # down right
        [3, 3],
        [3, 0],
    ]),
    np.array([  # front
        [3, 0],
        [0, 0],
    ]),
]

wall_corners_square = [
    np.array([  # left
        [0, 0],
        [0, 5],
    ]),
    np.array([  # back
        [0, 5],
        [5, 5],
    ]),
    np.array([  # right
        [5, 5],
        [5, 0],
    ]),
    np.array([  # front
        [5, 0],
        [0, 0],
    ]),
]

absorptions = [0.1]*len(wall_corners_L_shape)

room_shape = 0
def test_room_construct():

    global room_shape
    # Choose which room to use
    if room_shape == 0 :

        absorptions = [0.1] * len(wall_corners_square)
        walls = [pra.libroom.Wall2D(c, a) for c, a in zip(wall_corners_square, absorptions)]
        obstructing_walls = []
        microphones = np.array([
            [3.2, ],
            [0.7, ],
        ])

        room = pra.libroom.Room2D(walls, obstructing_walls, microphones)

    else :
        absorptions = [0.1] * len(wall_corners_L_shape)
        walls = [pra.libroom.Wall2D(c, a) for c, a in zip(wall_corners_L_shape, absorptions)]
        obstructing_walls = []
        microphones = np.array([
            [3.9, ],
            [4.1, ],
        ])

        room = pra.libroom.Room2D(walls, obstructing_walls, microphones)

    return room


def compute_rir(mic_log, time_thres, fs, mic_id=1, plot=True):

    TIME = 0
    ENERGY = 1

    # ======= WITH FRACTIONAL PART =======

    # the python utilities to compute the rir
    fdl = pra.constants.get('frac_delay_length')
    fdl2 = (fdl - 1) // 2  # Integer division

    ir = np.zeros(int(time_thres * fs) + fdl)

    for entry in mic_log:
        time_ip = int(np.floor(entry[TIME] * fs))

        if time_ip > len(ir) - fdl2 or time_ip < fdl2:
            continue

        time_fp = (entry[TIME] * fs) - time_ip

        ir[time_ip - fdl2:time_ip + fdl2 + 1] += (entry[ENERGY] * fractional_delay(time_fp))


    if plot :
        x = np.arange(len(ir)) / fs
        plt.figure()
        plt.plot(x, ir)
        plt.title("RIR for microphone " + str(mic_id))
        plt.show()

    return ir

def apply_rir(rir, wav_file_name, fs, cutoff=200, result_name="aaa.wav"):


    fs0, audio_anechoic = wavfile.read(wav_file_name)


    if len(audio_anechoic.shape) > 1 :
        audio_anechoic = audio_anechoic[:,0]

    audio_anechoic = audio_anechoic - np.mean(audio_anechoic)

    # Compute the convolution and set all coefficients between -1 and 1 (range for float32 .wav files)
    result = scipy.signal.fftconvolve(rir, audio_anechoic)

    if cutoff > 0:
        result = highpass(result, fs, cutoff)

    result /= np.abs(result).max()
    result -= np.mean(result)
    wavfile.write(result_name, rate=fs, data=result.astype('float32'))

def highpass(audio, fs, cutoff=200, butter_order=5):
    nyq = 0.5 * fs
    fc_norm = cutoff / nyq
    b, a = signal.butter(butter_order, fc_norm, btype="high", analog=False)
    return signal.lfilter(b, a, audio)


if __name__ == '__main__':

    room = test_room_construct()

    # parameters
    nb_phis = 50
    nb_thetas = 50
    source_pos = [2.,2.2]
    mic_radius = 0.5


    scatter_coef = 0.1


    time_thres = 0.8 #s
    energy_thres = 0.001
    sound_speed = 340

    fs = 16000

    # compute the log with the C++ code
    chrono = time.time()
    log = room.get_rir_entries(nb_phis, nb_thetas, source_pos, mic_radius, scatter_coef, time_thres, energy_thres, sound_speed)


    print("\nNumber of phi : ", nb_phis)
    print("Number of theta :", nb_thetas)
    print("\nThere are", len(log), " microphone(s).")
    print("(", nb_phis, "*", nb_thetas,") =", nb_phis * nb_thetas, " rays traced in ", time.time() - chrono, " seconds")

    print("------------------\n")

    # For all microphones
    for k in range(len(log)):
        mic_log = log[k]
        print("For microphone n° ", k+1, ": ", len(mic_log), " entries to build the rir")

        rir = compute_rir(mic_log, time_thres, fs, mic_id = k+1, plot=True)
        apply_rir(rir, "0riginal.wav", fs, cutoff=0, result_name="aaa_mic"+str(k+1))



