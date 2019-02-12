import unittest
import numpy as np
import pyroomacoustics as pra

room_dim = [15, 14, 16]
absorption = 0.2
source_position = [2.0, 3.1, 2.0]
mic_position = [2.0, 1.5, 2.0]
fs = 16000
max_order = 5

# scenario A
def get_room_constructor_args():
    '''
    When provided with sources and microphones, the constructor
    should try to compute the RIR immediately
    '''
    source = pra.SoundSource(position=source_position)
    mics = pra.MicrophoneArray(np.array([mic_position]).T, fs)
    shoebox = pra.ShoeBox(
            room_dim,
            absorption=absorption,
            fs=fs,
            max_order=max_order,
            sources=[source],
            mics=mics,
            )

    shoebox.image_source_model()
    shoebox.compute_rir()
    return shoebox

#scenario B
def get_room_add_method():
    shoebox = pra.ShoeBox(room_dim, absorption=absorption, fs=fs, max_order=max_order)
    shoebox.add_source(source_position)
    mics = pra.MicrophoneArray(np.array([mic_position]).T, fs)
    shoebox.add_microphone_array(mics)

    shoebox.image_source_model()
    shoebox.compute_rir()
    return shoebox

class RoomConstructorSources(unittest.TestCase):

    def test_room_constructor(self):

        room_1 = get_room_constructor_args()
        self.assertTrue(isinstance(room_1.sources[0], pra.SoundSource))

    def test_room_add_method(self):
        room_2 = get_room_add_method()
        self.assertTrue(isinstance(room_2.sources[0], pra.SoundSource))

    def test_rir_equal(self):
        room_1 = get_room_constructor_args()
        room_2 = get_room_add_method()
        self.assertTrue(np.allclose(room_1.rir[0][0], room_2.rir[0][0]))


if __name__ == '__main__':

    unittest.main()
