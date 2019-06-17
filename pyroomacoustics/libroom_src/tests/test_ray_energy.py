import unittest
import numpy as np
import pyroomacoustics as pra


class SimpleHistogram(list):
    """ A Histogram class based on list """

    def __init__(self, bin_size):
        self.bin_size = bin_size
        list.__init__([])

    def add(self, delay, val):
        """
        Adds val to the bin corresponding to delay.
        Pad the histogram to correct size if too short.
        """
        pos = int(delay / self.bin_size)

        if pos >= len(self):
            while len(self) < pos:
                self.append(0)
            self.append(val)
        else:
            self[pos] += val


class TestRayEnergy(unittest.TestCase):
    def test_sqare_room(self):

        """
        This is a cubic room of 2x2x2 meters. The source is placed at [0.5,0.5, 1]
        and the receiver at [1.5, 1.5, 1]. A ray is launched towards [1, 0, 1] so that
        the first receiver hit is after travel distance of 2*sqrt(2) and each subsequent
        hit travels a further 4*sqrt(2) until the threshold energy is reached.
        """

        absorption = 0.07
        round_trip = 4 * np.sqrt(2)
        energy_thresh = 1e-7
        detector_radius = 0.15
        hist_bin_size = 0.004  # resolution of histogram [s]

        histogram_gt = SimpleHistogram(hist_bin_size * pra.constants.get("c"))

        # Create the groundtruth list of energy and travel time
        initial_energy = 2. / detector_radius ** 2  # defined in libroom.Room.get_rir_entries
        transmitted = 1.0 * (1. - absorption) ** 2 * initial_energy
        distance = round_trip / 2.0

        while transmitted / distance > energy_thresh:
            histogram_gt.add(distance, transmitted)
            transmitted *= (1. - absorption) ** 4
            distance += round_trip

        print("Creating the python room")
        room = pra.ShoeBox([2, 2, 2], fs=16000, materials=pra.Material.make_freq_flat(absorption))
        # room = pra.Room(walls, fs=16000)
        room.add_source([0.5, 0.5, 1])
        room.add_microphone_array(pra.MicrophoneArray(np.c_[[1.5, 1.5, 1.0]], room.fs))

        print("Creating the cpp room")
        room.room_engine.set_params(
            room.c,
            0,
            energy_thresh,  # energy threshold for rays
            5.0,  # time threshold for rays
            detector_radius,  # detector radius
            hist_bin_size,  # resolution of histogram [s]
            False,  # is it hybrid model ?
        )

        print("Running ray tracing")
        room.room_engine.get_rir_entries(
            np.c_[[-np.pi / 4.0, np.pi / 2.0]], room.sources[0].position  # source loc
        )

        h = room.room_engine.microphones[0].histograms[0].get_hist()
        histogram_rt = np.array(h[0])[:len(histogram_gt)]

        self.assertTrue(np.allclose(histogram_rt, histogram_gt))


if __name__ == "__main__":
    unittest.main()
