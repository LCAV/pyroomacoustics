
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

room_ll = [-1,-1]
room_ur = [1,1]
src_pos = [0,0]
mic_pos = [0.5, 0.1]

# Create a 4 by 6 metres shoe box room
room = pra.Room.shoeBox2D(room_ll, room_ur, max_order=10)

# Add a source somewhere in the room
room.addSource(src_pos)

print(room.sources[0].images)
print(room.checkVisibilityForAllImages(room.sources[0], mic_pos))

# Create a linear array beamformer with 4 microphones
# with angle 0 degrees and inter mic distance 10 cm
R = pra.linear2DArray(mic_pos, 1, np.pi/4, 0.15) 
room.addMicrophoneArray(pra.Beamformer(R, room.fs))

room.compute_RIR()

# plot the room and resulting beamformer
room.plot(img_order=10)

plt.figure()
room.plotRIR()

plt.show()
