from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
import IPython

fs, audio_anechoic = wavfile.read('notebooks/arctic_a0010.wav')
IPython.display.display(IPython.display.Audio(audio_anechoic, rate=fs))

pol = np.array([[0,0], [0,4], [3,2], [3,0]]).T
room = pra.Room.from_corners(pol, fs=16000, max_order=10, absorption=0.1)

# Adding the source
room.add_source(np.array([1.8, 0.4]), signal=audio_anechoic)

# Adding the microphone
R = np.array([[0.5],[1.2],[0.5]])
room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

room.plot()
plt.title("2D shape of the room (the height of the room is 2 meters)")

def get_rir(size='medium', absorption='medium'):

    
    if absorption=='high':
        absor = 0.7
    elif absorption=='medium':
        absor = 0.3
    elif absorption=='low':
        absor = 0.1
    else:
        raise ValueError("The absorption parameter can only take values ['low', 'medium', 'high']")
    
    if size=='large':
        size_coef = 5.
    elif size=='medium':
        size_coef = 2.5
    elif size=='small':
        size_coef = 1.
    else:
        raise ValueError("The size parameter can only take values ['small', 'medium', 'large']")
        
        
    pol = size_coef * np.array([[0,0], [0,4], [3,2], [3,0]]).T
    room = pra.Room.from_corners(pol, fs=16000, max_order=1, absorption=absor)

    # Create the 3D room by extruding the 2D by a specific height
    room.extrude(size_coef * 2.5, absorption=absor)

    # Adding the source
    room.add_source(size_coef * np.array([1.8, 0.4, 1.6]), signal=audio_anechoic)

    # Adding the microphone
    R = size_coef * np.array([[0.5],[1.2],[0.5]])
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    # Compute the RIR using the hybrid method
    room.compute_rir(mode='hybrid', nb_thetas=500, nb_phis=500, mic_radius=0.15, scatter_coef=0.)

    # Plot and apply the RIR on the audio file
    room.plot_rir()
    plt.show()

    return room.rir[0][0]

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: {} [small/medium/large] [low/medium/high]'.format(sys.argv[0]))

    else:

        size = sys.argv[1]
        absorption = sys.argv[2]

        new_rir = get_rir(size=size, absorption=absorption)

        fs, old_rir = wavfile.read('notebooks/rir_{}_{}.wav'.format(size, absorption))

        plt.figure()
        plt.plot(old_rir, label='old')
        plt.plot(new_rir, label='new')
        plt.legend()
        plt.show()

        print('Max error (rel):', np.max(np.abs(new_rir - old_rir))/np.max(np.abs(new_rir)))
        print('Mean error (rel):', np.mean(np.abs(new_rir - old_rir))/np.max(np.abs(new_rir)))


