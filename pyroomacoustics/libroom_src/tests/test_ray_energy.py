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

        absorption = 0.07
        round_trip = 4 * np.sqrt(2)
        energy_thresh = 1e-7
        transmission = np.sqrt(1. - absorption)

        # Create the groundtruth list of energy and travel time
        log_gt = []
        transmitted = 1. * transmission ** 2
        distance = round_trip / 2.
        while transmitted / distance > np.sqrt(energy_thresh):
            log_gt.append([distance, transmitted / distance])
            transmitted *= transmission ** 4
            distance += round_trip

        print('Creating the python room')
        room = pra.ShoeBox([2, 2, 2], fs=16000, absorption=absorption)
        # room = pra.Room(walls, fs=16000)
        room.add_source([0.5, 0.5, 1])
        room.add_microphone_array(pra.MicrophoneArray(np.c_[[1.5, 1.5, 1.]], room.fs))

        ray_0 = np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0.])

        print('Creating the cpp room')
        room.room_engine.set_params(
                room.c,
                0,
                energy_thresh,  # energy threshold for rays
                5.,  # time threshold for rays
                0.15,  # detector radius
                0.004, # resolution of histogram [s]
                False,  # is it hybrid model ?
                )

        print('Running ray tracing')
        log = room.room_engine.get_rir_entries(
                np.c_[[-np.pi / 4., np.pi / 2.]],
                room.sources[0].position,  # source loc
                )

        log_tr = []
        for hit in log[0]:
            log_tr.append([hit.distance, hit.transmitted[0]])

        import pdb; pdb.set_trace()

        self.assertTrue(np.allclose(log_tr, log_gt))

if __name__ == '__main__':
    unittest.main()
