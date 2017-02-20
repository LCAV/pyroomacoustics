
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

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
    
room1.addSource([1, 4], None, 0)


computed = room1.checkVisibilityForAllImages(room1.sources[0], np.array([4, 4]))
expected = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

print expected
print computed
room1.plot(img_order=2)

plt.show()
