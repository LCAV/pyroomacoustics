import numpy as np
import matplotlib.pyplot as plt

import pyroomacoustics as pra

"""
Here is an example of creating a room, adding a source and plotting the result.
Some information about the images is returned (including visibility at a certain point)
"""


# Simulation parameters
fs = 8000
t0 = 1./(fs*np.pi*1e-2)
absorption = 0.90
max_order_sim = 3
sigma2_n = 5e-7
pVis = [1,1] # this point is where we check visibility (warning : put 3 coordinates for 3D)


# Create the rooms
rooms = []
rooms.append(pra.Room.fromCorners(np.array([[0,5,5,3,3,2,2,0], [0,0,5,5,2,2,5,5]]),absorption,fs,t0,max_order_sim,sigma2_n))
rooms.append(pra.Room.shoeBox2D([0,0],[4, 4],absorption,fs,t0,max_order_sim,sigma2_n))
rooms.append(pra.Room.shoeBox3D([0,0,0],[6, 6, 6],absorption,fs,t0,max_order_sim,sigma2_n))


# Select an id corresponding to the room we want to show
# 0 = 2D "U room"
# 1 = 2D shoe box
# 2 = 3D shoe box
roomId = 1 # change this id to select a different room
room = rooms[roomId]


# Add a source to the room
sourcePos = [2,1] # position of the sound source (warning : put 3 coordinates for 3D)
room.addSource(sourcePos, None, 0)


# Plotting the result
print("POSITIONS :")
print(room.sources[0].images)
print("PARENTS :")
print(room.sources[0].generators)
print("GENERATING WALLS")
print(room.sources[0].walls)
print("ORDERS :")
print(room.sources[0].orders)
print("IS VISIBLE FROM "+str(pVis)+" :")
print(room.checkVisibilityForAllImages(room.sources[0], np.array(pVis)))

if(roomId != 2):
    room.plot()
    plt.show()
