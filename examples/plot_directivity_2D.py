import numpy as np
import matplotlib.pyplot as plt

from pyroomacoustics import dB
from pyroomacoustics.directivities import DirectivityPattern, \
    DirectionVector, CardioidFamily

ORIENTATION = DirectionVector(azimuth=0, colatitude=90, degrees=True)
LOWER_GAIN = -20

# plot each directivity
angles = np.linspace(start=0, stop=360, num=361, endpoint=True)
angles = np.radians(angles)

# plot each pattern
fig = plt.figure()
ax = plt.subplot(111, projection="polar")
for _dir in DirectivityPattern.values():

    dir_obj = CardioidFamily(orientation=ORIENTATION, pattern_name=_dir)
    gains = []
    for a in angles:
        direction = DirectionVector(azimuth=a, degrees=False)
        gains.append(dir_obj.get_response(direction))
    gains_db = dB(np.array(gains))
    ax.plot(angles, gains_db, label=_dir)

plt.legend(bbox_to_anchor=(1, 1))
plt.ylim([LOWER_GAIN, 0])
ax.yaxis.set_ticks(np.arange(start=LOWER_GAIN, stop=5, step=5))
plt.tight_layout()
plt.show()
