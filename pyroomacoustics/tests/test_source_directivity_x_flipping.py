import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pyroomacoustics.directivities import DirectivityPattern, \
    DirectionVector, CardioidFamily
from unittest import TestCase 

# create room
room  = pra.ShoeBox(
            p = [10,14,10],
            materials=pra.Material(0.07),
            fs=16000,
            max_order=1,
        )

# add source 
source=[5,2,5]
room.add_source(source)

# add microphone 
mic=[5,12,5]
room.add_microphone(mic)

# compute image sources
room.image_source_model()

# compute azimuth_s and colatitude_s pair for images along x-axis
source_angle_array = pra.room.source_angle_function(image_source_array=room.sources[0].images, n_array=abs(room.sources[0].orders_xyz), mic=mic)

image_1_angles = source_angle_array[:,2]
image_2_angles = source_angle_array[:,4]

azimuth_s_1 = image_1_angles[0]
colatitude_s_1 = image_1_angles[1]
azimuth_s_2 = image_2_angles[0]
colatitude_s_2 = image_2_angles[1]


class Test_Source_Directivity_X(TestCase):

    def test_image_1(self):
        self.assertTrue(azimuth_s_1 == 3*np.pi/4)
        self.assertTrue(colatitude_s_1 == np.pi/2)

    def test_image_2(self):
        self.assertTrue(azimuth_s_2 == np.pi/4)
        self.assertTrue(colatitude_s_2 == np.pi/2)
    


def get_error_image_1():

    print("Error in image 1 along x-axis")
    error_azimuth = abs(azimuth_s_1 - (3*np.pi/4))
    error_colatitude = abs(colatitude_s_1 - (np.pi/2))
    print("-" * 40)
    print("The error in azimuth_s calculation is: {}".format(error_azimuth))
    print("The error in colatitude_s calculation is: {}".format(error_colatitude))
    print("-" * 40)
    print()
    print()


def get_error_image_2():

    print("Error in image 2 along x-axis")
    error_azimuth = abs(azimuth_s_2 - (np.pi/4))
    error_colatitude = abs(colatitude_s_2 - (np.pi/2))
    print("-" * 40)
    print("The error in azimuth_s calculation is: {}".format(error_azimuth))
    print("The error in colatitude_s calculation is: {}".format(error_colatitude))
    print("-" * 40)
    print()
    print()


if __name__ == '__main__':
    get_error_image_1()
    get_error_image_2()