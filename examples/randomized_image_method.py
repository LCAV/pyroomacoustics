"""
Randomized image method example
===============================

In this example, we will show the benefits of using the randomized
image method to remove sweeping echoes in RIRs simulated with ISM.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pyroomacoustics as pra
from pyroomacoustics import metrics as met
from pyroomacoustics.transform import stft

# create an example with sweeping echo - from Enzo's paper
room_size = [4, 4, 4]
source_loc = [1, 2, 2]
center_x = room_size[0] / 2
center_y = room_size[1] / 2
center_z = room_size[2] / 2

# number of mics
M = 15
mic_loc = np.zeros((3, M), dtype=float)
gridSpacing = 0.25


# =============================================================================
# create uniform grid of mics
# =============================================================================
count = 0
Nmic_each_axis = int((M / 3 - 1) / 2)
for n in range(-Nmic_each_axis, Nmic_each_axis + 1):
    mic_loc[:, count] = [center_x + n * gridSpacing, center_y, center_z]
    mic_loc[:, count + 1] = [center_x, center_y + n * gridSpacing, center_z]
    mic_loc[:, count + 2] = [center_x, center_y, center_z + n * gridSpacing]
    count += 3


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(source_loc[0], source_loc[1], source_loc[2], marker="+", c="k", s=40)
ax.scatter(mic_loc[0, :], mic_loc[1, :], mic_loc[2, :])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Speaker and mic array configuration")


plt.xlim([0, 4])
plt.ylim([0, 4])


###############################################################
# normal ISM with sweeping echoes
###############################################################

# sampling frequency
fs = 44100

room = pra.ShoeBox(room_size, fs, materials=pra.Material(0.1), max_order=50)

room.add_source(source_loc)

room.add_microphone_array(pra.MicrophoneArray(mic_loc, fs))

room.compute_rir()

# plot spectrograms to check for sweeping echoes

fft_size = 512  # fft size for analysis
fft_hop = 128  # hop between analysis frame
fft_zp = 512  # zero padding
analysis_window = pra.hann(fft_size)

print("Sweeping echo measure for ISM is :")
for n in range(M):
    if n == 0:
        S = stft.analysis(
            room.rir[n][0], fft_size, fft_hop, win=analysis_window, zp_back=fft_zp
        )

        f, (ax1, ax2) = plt.subplots(2, 1)

        ax1.imshow(
            pra.dB(S.T),
            extent=[0, len(room.rir[n][0]), 0, fs / 2],
            vmin=-100,
            vmax=0,
            origin="lower",
            cmap="jet",
        )
        ax1.set_title("RIR for Mic location " + str(n) + " without random ISM")
        ax1.set_ylabel("Frequency")
        ax1.set_aspect("auto")

        # plot RIR
        ax2.plot(room.rir[n][0])
        ax2.set_xlabel("Num samples")
        ax2.set_ylabel("Amplitude")

    # measure of spectral flatness of sweeping echos
    # higher value is desired
    ssf = met.sweeping_echo_measure(room.rir[n][0], fs)
    print(ssf)


##########################################################
# Randomized ISM should reduce sweeping echoes
# choose a maximum displacement of 5cm
##########################################################

room = pra.ShoeBox(
    room_size,
    fs,
    materials=pra.Material(0.1),
    max_order=50,
    use_rand_ism=True,
    max_rand_disp=0.05,
)

room.add_source(source_loc)

room.add_microphone_array(pra.MicrophoneArray(mic_loc, fs))

room.compute_rir()


print("Sweeping echo measure for randomized ISM is:")
for n in range(M):
    if n == 0:
        S = stft.analysis(
            room.rir[n][0], fft_size, fft_hop, win=analysis_window, zp_back=fft_zp
        )

        f, (ax1, ax2) = plt.subplots(2, 1)

        ax1.imshow(
            pra.dB(S.T),
            extent=[0, len(room.rir[n][0]), 0, fs / 2],
            vmin=-100,
            vmax=0,
            origin="lower",
            cmap="jet",
        )
        ax1.set_title("RIR for Mic location " + str(n) + " with random ISM")
        ax1.set_ylabel("Frequency")
        ax1.set_aspect("auto")

        # plot RIR
        ax2.plot(room.rir[n][0])
        ax2.set_xlabel("Num samples")
        ax2.set_ylabel("Amplitude")

    # measure of spectral flatness of sweeping echos
    # higher value is desired
    ssf = met.sweeping_echo_measure(room.rir[n][0], fs)
    print(ssf)

    # show plots
    plt.show()
