
import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.c_package import libroom, c_float_p
import matplotlib.pyplot as plt
import ctypes

fs = 8000
t0 = 1./(fs*np.pi*1e-2)
absorption = 0.90
max_order_sim = 2
sigma2_n = 5e-7

corners = np.array([[0,5,5,3,3,2,2,0], [0,0,5,5,2,2,5,5]])
room1 = pra.Room.fromCorners(
    corners,
    absorption,
    fs,
    t0,
    max_order_sim,
    sigma2_n)
    
room1.addSource([1, 4], signal=None, delay=0, compute_images=True)
mics = pra.MicrophoneArray(np.array([[4], [4]]),fs)
room1.addMicrophoneArray(mics)


computed = room1.checkVisibilityForAllImages(room1.sources[0], mics.R[:,0])
expected = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

print expected
print computed
print room1.sources[0].damping

room1.plot(img_order=2)


room1.image_source_model()
print room1.sources[0].images
print room1.sources[0].damping

plt.show()
