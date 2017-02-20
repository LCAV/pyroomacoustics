
import numpy as np
import pyroomacoustics as pra

fs = 8000
t0 = 1./(fs*np.pi*1e-2)
absorption = 0.90
max_order_sim = 2
sigma2_n = 5e-7

room_dim = [6, 6, 6]
room1 = pra.Room.shoeBox3D(
    [0,0,0],
    room_dim,
    absorption,
    fs,
    t0,
    max_order_sim,
    sigma2_n)
    
room1.addSource([3, 3, 3], None, 0)


computed = room1.checkVisibilityForAllImages(room1.sources[0], np.array([5, 3, 3]))
expected = [1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,1,1,1,1,1,0,0,1,1,1,1]

print expected[4]
print computed[4]
print room1.walls[3].plane_basis
