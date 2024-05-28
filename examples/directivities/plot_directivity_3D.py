import matplotlib.pyplot as plt
import numpy as np

from pyroomacoustics.directivities import (
    HyperCardioid,
    DirectionVector,
)

orientation = DirectionVector(azimuth=0, colatitude=45, degrees=True)

# create cardioid object
dir_obj = HyperCardioid(orientation=orientation)

# plot
azimuth = np.linspace(start=0, stop=360, num=361, endpoint=True)
colatitude = np.linspace(start=0, stop=180, num=180, endpoint=True)
# colatitude = None   # for 2D plot
dir_obj.plot_response(azimuth=azimuth, colatitude=colatitude, degrees=True)
plt.show()
