import unittest
import numpy as np
import pyroomacoustics as pra

class TestRayEnergy(unittest.TestCase):

    def test_sqare_room(self):

        '''
        This is a cubic room of 2x2x2 meters. The source is placed at [0.5,0.5, 1]
        and the receiver at [1.5, 1.5, 1]. A ray is launched towards [1, 0, 1] so that
        the first receiver hit is after travel distance of 2*sqrt(2) and each subsequent
        hit travels a further 4*sqrt(2) until the threshold energy is reached.
        '''

        absorption = 0.1
        round_trip = 4 * np.sqrt(2)
        energy_thresh = 1e-7
        sound_speed = pra.constants.get('c')

        # Create the groundtruth list of energy and travel time
        log_gt = []
        energy = 1. * (1. - absorption) ** 2
        distance = round_trip / 2.
        while energy / (4 * np.pi * distance) > energy_thresh:
            log_gt.append([distance / sound_speed, energy / (4 * np.pi * distance)])
            energy *= (1. - absorption) ** 4
            distance += round_trip

        print('Creating the python room')
        room = pra.ShoeBox([2, 2, 2], fs=16000, absorption=absorption)
        #room = pra.Room(walls, fs=16000)
        room.add_source([0.5, 0.5, 1])
        room.add_microphone_array(pra.MicrophoneArray(np.c_[[1.5, 1.5, 1.]], room.fs))

        ray_0 = np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0.])

        print('Creating the cpp room')
        c_room = pra.room_factory(room.walls, room.obstructing_walls, room.mic_array.R)

        print('Running ray tracing')
        log = c_room.get_rir_entries(
                np.c_[[-np.pi / 4., np.pi / 2.]],
                room.sources[0].position,  # source loc
                0.15,  # detector radius
                0.,  # scat. coeff
                5.,  # time thresh
                energy_thresh,  # energy thresh
                sound_speed,  # speed sound
                False,  # is it hybrid model ?
                2,  # order of ism
                )

        self.assertTrue(np.allclose(log[0], log_gt))

if __name__ == '__main__':
    unittest.main()
