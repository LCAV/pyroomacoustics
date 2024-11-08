import matplotlib.pyplot as plt
import numpy as np

from pyroomacoustics import dB
from pyroomacoustics.directivities import (
    Cardioid,
    DirectionVector,
    FigureEight,
    HyperCardioid,
    SubCardioid,
)

orientation = DirectionVector(azimuth=0, colatitude=90, degrees=True)
lower_gain = -20

# plot each directivity
angles = np.linspace(start=0, stop=360, num=361, endpoint=True)
angles = np.radians(angles)

# plot each pattern
fig = plt.figure()
ax = plt.subplot(111, projection="polar")
for obj in [SubCardioid, Cardioid, HyperCardioid, FigureEight]:
    dir_obj = obj(orientation=orientation)
    resp = dir_obj.get_response(azimuth=angles, magnitude=True, degrees=False)
    resp_db = dB(np.array(resp))
    ax.plot(angles, resp_db, label=dir_obj.directivity_pattern)

plt.legend(bbox_to_anchor=(1, 1))
plt.ylim([lower_gain, 0])
ax.yaxis.set_ticks(np.arange(start=lower_gain, stop=5, step=5))
plt.tight_layout()
plt.show()
