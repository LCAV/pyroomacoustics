import numpy as np
import matplotlib.pyplot as plt
from pyroomacoustics.beamforming import circular_2D_array
from pyroomacoustics.directivities import DirectivityPattern, DirectionVector, CardioidFamily


def circular_microphone_array_helper_xyplane_plot(center, M, phi0, radius, directivity_pattern=None): 
    """
        Plots a circular microphone array with directivities pointing outwards
        Parameters
        ----------
        center: array_like
            The center of the microphone array
        M: int
            The number of microphones
        phi0: float
            The counterclockwise rotation (in degrees) of the first element in the microphone array (from
            the x-axis)
        radius: float
            The radius of the microphone array
        directivity_pattern: string
            The directivity pattern (FIGURE_EIGHT/HYPERCARDIOID/CARDIOID/SUBCARDIOID/OMNI)
        """
    
    azimuth_list = np.arange(M)*(360/M)
    phi_array = np.ones(M)*phi0
    azimuth_list = np.add(azimuth_list, phi_array)

    R = circular_2D_array(center=center, M=M, phi0=phi0, radius=radius)

    if directivity_pattern == "FIGURE_EIGHT":
        PATTERN = DirectivityPattern.FIGURE_EIGHT
    elif directivity_pattern == "HYPERCARDIOID":
        PATTERN = DirectivityPattern.HYPERCARDIOID
    elif directivity_pattern == "CARDIOID":
        PATTERN = DirectivityPattern.CARDIOID
    elif directivity_pattern == "SUBCARDIOID":
        PATTERN = DirectivityPattern.SUBCARDIOID
    else:
        PATTERN = DirectivityPattern.OMNI

    ax = None

    for i in range(M):
        # make directivity object
        ORIENTATION = DirectionVector (azimuth=azimuth_list[i], colatitude=90, degrees=True)
        dir_obj = CardioidFamily (orientation=ORIENTATION, pattern_enum=PATTERN)
        # plot
        azimuth = np.linspace(start=0, stop=360, num=361, endpoint=True)
        colatitude = np.linspace(start=0, stop=180, num=180, endpoint=True)
        ax = dir_obj.plot_response(azimuth=azimuth, colatitude=colatitude, degrees=True, ax=ax, offset=[R[0][i],R[1][i],0])
        
    plt.show()



circular_microphone_array_helper_xyplane_plot(center=[2.,2.], M=6, phi0=0, radius=10, directivity_pattern="HYPERCARDIOID")
