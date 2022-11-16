import numpy as np
import matplotlib.pyplot as plt

from pyroomacoustics import dB, all_combinations
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
    cardioid_func,
)
from pyroomacoustics.doa import spher2cart

ORIENTATION = DirectionVector(azimuth=0, colatitude=90, degrees=True)
azimuth = np.radians(np.linspace(start=0, stop=360, num=361, endpoint=True))
colatitude = np.radians(np.linspace(start=0, stop=180, num=180, endpoint=True))
LOWER_GAIN = -40

""" 2D """
# get cartesian coordinates
cart = spher2cart(azimuth=azimuth)
direction = spher2cart(azimuth=225, degrees=True)

# compute response
resp = cardioid_func(x=cart, direction=direction, coef=0.5, magnitude=True)
resp_db = dB(np.array(resp))

# plot
plt.figure()
plt.polar(azimuth, resp_db)
plt.ylim([LOWER_GAIN, 0])
ax = plt.gca()
ax.yaxis.set_ticks(np.arange(start=LOWER_GAIN, stop=5, step=10))
plt.tight_layout()

""" 3D """
# get cartesian coordinates
spher_coord = all_combinations(azimuth, colatitude)
cart = spher2cart(azimuth=spher_coord[:, 0], colatitude=spher_coord[:, 1])
direction = spher2cart(azimuth=0, colatitude=45, degrees=True)

# compute response
resp = cardioid_func(x=cart, direction=direction, coef=0.25, magnitude=True)

# plot (surface plot)
fig = plt.figure()
RESP_2D = resp.reshape(len(azimuth), len(colatitude))
AZI, COL = np.meshgrid(azimuth, colatitude)
X = RESP_2D.T * np.sin(COL) * np.cos(AZI)
Y = RESP_2D.T * np.sin(COL) * np.sin(AZI)
Z = RESP_2D.T * np.cos(COL)
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.plot_surface(X, Y, Z)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

plt.show()
