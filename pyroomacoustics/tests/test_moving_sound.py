import numpy as np

import pyroomacoustics as pra


def test_simulation_output_shape():
    # Set up input parameters for the test
    position = np.array([0, 0])
    signal = np.random.rand(44100)  # Replace with an actual signal
    fs = 44100
    stept = 200
    speed = 5
    x_direction = True
    y_direction = True

    # Create an instance of the Room class
    room = pra.ShoeBox([100, 100])  # Adjust the room dimensions as needed

    # Microphone parameters
    # Two microphones at different locations
    mic_locations = np.array([[1, 1], [3, 1]])
    room.add_microphone_array(pra.MicrophoneArray(mic_locations, room.fs))

    # Set the source location as a 1D or 2D array with the correct shape
    position = np.array([[2, 3]])  # 2D array with shape (1, 2)

    # Call the simulate_moving_sound function
    movemix, filter_kernels = room.simulate_moving_sound(
        position=position,
        signal=signal,
        fs=fs,
        stept=stept,
        speed=speed,
        x_direction=x_direction,
        y_direction=y_direction,
    )

    print(movemix, filter_kernels)


if __name__ == "__main__":
    test_simulation_output_shape()
