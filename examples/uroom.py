from __future__ import print_function

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
room = pra.Room.fromCorners(
    corners,
    absorption,
    fs,
    t0,
    max_order_sim,
    sigma2_n)
    
room.addSource([1, 4], signal=None, delay=0)
mics = pra.MicrophoneArray(np.array([[4, 4], [2.5, 1], [2.5, 4]]).T,fs)
room.addMicrophoneArray(mics)

room.image_source_model(use_libroom=False)

ordering = np.lexsort(room.sources[0].images)

print("Image sources:", room.sources[0].images[:,ordering])
print("Visibles:", room.visibility[0][:,ordering])

room.plot(img_order=2)

plt.show()
