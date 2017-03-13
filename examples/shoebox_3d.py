from __future__ import print_function
import time
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

fs = 16000
t0 = 1./(fs*np.pi*1e-2)
absorption = 0.80
max_order_sim = 10
sigma2_n = 5e-7

room_dim = [5, 4, 6]
shoebox = pra.ShoeBox(
    room_dim,
    absorption=absorption,
    fs=fs,
    t0=t0,
    max_order=max_order_sim,
    sigma2_awgn=sigma2_n
    )

room = pra.Room(
        shoebox.walls,
        fs=fs,
        t0=t0,
        max_order=max_order_sim,
        sigma2_awgn=sigma2_n
        )

source_loc = [2, 3.5, 2]
mic_loc = np.array([[2, 1.5, 2]]).T
shoebox.addSource(source_loc)
room.addSource(source_loc)

room.addMicrophoneArray(pra.MicrophoneArray(mic_loc, fs))
shoebox.addMicrophoneArray(pra.MicrophoneArray(mic_loc, fs))

then = time.time()
shoebox.image_source_model(use_libroom=False)
shoebox.compute_RIR()
shoebox_exec_time = time.time() - then

then = time.time()
room.image_source_model(use_libroom=True)
room.compute_RIR()
room_exec_time = time.time() - then

print("Time spent (room):", room_exec_time)
print("Time spent (shoebox):", shoebox_exec_time)
