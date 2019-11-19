import numpy as np
import json
import os
from glob import glob
from pprint import pprint
import random
import soundfile as sf
import argparse

from pyroomacoustics.utilities import rms, sample_audio
from pyroomacoustics.random.room import ShoeBoxRoomGenerator


"""

Example script for:

1) Generating a dataset of random room configuration and saving their 
corresponding room impulse responses.
```
python examples/generate_room_dataset.py make_dataset
```

2) Randomly selecting a room from the dataset and applying its room impulse
responses to a randomly selected speech file and (depending on the selected 
room) some noise sources.
```
python examples/generate_room_dataset.py apply_rir \
    --room_dataset <ROOM_DATASET_PATH>
```

"""

example_noise_files = [
    'examples/input_samples/doing_the_dishes.wav',
    'examples/input_samples/exercise_bike.wav',
    'examples/input_samples/running_tap.wav',
]


def make_dataset(n_rooms, source_min_dist_mic):
    """

    Generate a dataset of room impulse responses. A new folder will be created
    with the name `pra_room_dataset_<TIMESTAMP>` with the following structure:

    ```
    pra_room_dataset_<TIMESTAMP>/
        room_metadata.json
        data/
            room_<uuid>.npz
            room_<uuid>.npz
            ...
    ```

    where `room_metadata.json` contains metadata about each room configuration
    in the `data` folder.

    The `apply_rir` functions shows a room can be selected at random in order
    to simulate a measurement in one of the randomly generated configurations.


    Parameters
    -----------
    n_rooms : int
        Number of room configurations to generate.
    source_min_dist_mic : float
        Minimum distance between each source and the microphone(s).
    """

    room_generator = ShoeBoxRoomGenerator(
        source_min_dist_mic=source_min_dist_mic)
    room_generator.create_dataset(n_rooms)


def apply_rir(room_dataset, target_speech, noise_dir, snr_db, output_file):
    """

    Randomly selecting a room from the dataset and applying its room impulse
    responses to a randomly selected speech file and (depending on the selected
    room) some noise sources.

    Parameters
    -----------
    room_dataset : str
        Path to room dataset from calling `make_dataset`.
    target_speech : str
        Path to a target speech WAV file.
    noise_dir : str
        Path to a directory with noise WAV files. Default is to apply the room
        impulse response to WAV file(s) from `examples/input_samples`.
    snr_db : float
        Desired signal-to-noise ratio resulting from simulation.
    output_file : str
        Path of output WAV file from simulation.

    """
    if room_dataset is None:
        raise ValueError('Provide a path to a room dataset. You can compute '
                         'one with the `make_dataset` command.')

    with open(os.path.join(room_dataset, 'room_metadata.json')) as json_file:
        room_metadata = json.load(json_file)

    # pick a room at random
    random_room_key = random.choice(list(room_metadata.keys()))
    _room_metadata = room_metadata[random_room_key]
    print('Room metadata')
    pprint(_room_metadata)

    # load target audio
    target_data, fs_target = sf.read(target_speech)

    # load impulse responses
    ir_file = os.path.join(room_dataset, 'data', _room_metadata['file'])
    ir_data = np.load(ir_file)
    n_noises = ir_data['n_noise']
    sample_rate = ir_data['sample_rate']
    assert sample_rate == fs_target, 'Target sampling rate does not match IR' \
                                     'sampling rate.'

    # apply target IR
    target_ir = ir_data['target_ir']
    n_mics, ir_len = target_ir.shape
    output_len = ir_len + len(target_data) - 1
    room_output = np.zeros((n_mics, output_len))
    for n in range(n_mics):
        room_output[n] = np.convolve(target_data, target_ir[n])

    # apply noise IR(s) if applicable
    if n_noises:

        if noise_dir is None:
            noise_files = example_noise_files
        else:
            noise_files = glob(os.path.join(noise_dir, '*.wav'))
        print('\nNumber of noise files : {}'.format(len(noise_files)))

        _noise_files = np.random.choice(noise_files, size=n_noises,
                                        replace=False)
        print('Selected noise file(s) : {}'.format(_noise_files))
        noise_output = np.zeros_like(room_output)
        for k, _file in enumerate(_noise_files):

            # load audio
            noise_data, fs_noise = sf.read(_file)
            assert fs_noise == sample_rate, 'Noise sampling rate {} does ' \
                                            'not match IR sampling rate.' \
                                            ''.format(_file)

            # load impulse response
            noise_ir = ir_data['noise_ir_{}'.format(k)]

            # sample segment of noise and normalize so each source has
            # roughly similar amplitude
            # take a bit more audio than target audio so we are sure to fill
            # up the end with noise (end of IR is sparse)
            _noise = sample_audio(noise_data, int(1.1*output_len))
            _noise /= _noise.max()

            # apply impulse response
            for n in range(n_mics):
                noise_output[n] = np.convolve(_noise, noise_ir[n])[:output_len]

        # rescale noise according to specified SNR, add to target signal
        noise_rms = rms(noise_output[0])
        signal_rms = rms(room_output[0])
        noise_fact = signal_rms / noise_rms * 10 ** (-snr_db / 20.)
        room_output += (noise_output * noise_fact)

    else:
        print('\nNo noise source in selected room!')

    # write output to file
    sf.write(output_file, np.squeeze(room_output), sample_rate)
    print('\nOutput written to : {}'.format(output_file))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Show how to generate a dataset of random room '
                    'configurations and how to apply the impulse responses '
                    'of one of the rooms.')
    parser.add_argument('-n', '--n_rooms', type=int, default=50,
                        help='Number of rooms to generate')
    parser.add_argument('-d', '--source_min_dist_mic', type=float,
                        default=0.5,
                        help='Minimum distance between each source and the'
                             'microphone(s)')
    parser.add_argument('-a', '--apply_room', type=str,
                        help='Path to room dataset. If provided, a room will'
                             'be randomly selected.')
    parser.add_argument('-t', '--target_speech', type=str,
                        default='examples/input_samples/cmu_arctic_us_aew_a0001.wav',
                        help='Path to a target speech WAV file.')
    parser.add_argument('-v', '--noise_dir', type=str,
                        help='Path to a directory with noise WAV files. '
                             'Default is to apply the room impulse response '
                             'to WAV file(s) from `examples/input_samples`.')
    parser.add_argument('-s', '--snr_db', type=float,
                        default=5.,
                        help='Desired signal-to-noise ratio resulting from '
                             'simulation.')
    parser.add_argument('-o', '--output_file', type=str,
                        default='simulated_output.wav',
                        help='Path of output WAV file from simulation.')
    args = parser.parse_args()

    if args.apply_room is not None:
        apply_rir(room_dataset=args.apply_room,
                  target_speech=args.target_speech,
                  noise_dir=args.noise_dir,
                  snr_db=args.snr_db,
                  output_file=args.output_file)
    else:
        make_dataset(args.n_rooms, args.source_min_dist_mic)
