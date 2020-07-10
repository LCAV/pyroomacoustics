import numpy as np
import matplotlib.pyplot as plt

from pyroomacoustics.directivities import DirectivityPattern, \
    DirectionVector, Directivity


PATTERN = DirectivityPattern.HYPERCARDIOID
ORIENTATION = DirectionVector(azimuth=0, colatitude=45, degrees=True)

# create cardioid object
dir_obj = Directivity(orientation=ORIENTATION, pattern=PATTERN)

# plot
azimuth = np.linspace(start=0, stop=360, num=361, endpoint=True)
colatitude = np.linspace(start=0, stop=180, num=180, endpoint=True)
# colatitude = None   # for 2D plot
dir_obj.plot_response(azimuth=azimuth, colatitude=colatitude, degrees=True)
plt.show()
