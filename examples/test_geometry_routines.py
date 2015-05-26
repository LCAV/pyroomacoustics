from unittest import TestCase

import numpy as np

import pyroomacoustics as pra

room = pra.Room.shoeBox2D([0,0],[4,4],0)
s1 = [[0, 4], [2, 2]]
s2 = [[2, 2], [0, 4]]
s3 = [[4, 4], [0, 4]]
s4 = [[4, 4], [0, 4]]
s5 = [[0, 2], [2, 2]]

print('===ORIENTATION===')
print('orientation : ',room.ccw3p(np.array([[6, 4, 2], [0, 2, 0]])))

print('===ORIENTATION===')
print('orientation : ',room.ccw3p(np.array([[2, 4, 6], [0, 2, 0]])))

print('===INTERSECTION (0)===')
print('intersects : ',room.intersects(np.array(s4), np.array(s5)))

print('===INTERSECTION (1)===')
print('intersects : ',room.intersects(np.array(s1), np.array(s2)))

print('===INTERSECTION (2)===')
print('intersects : ', room.intersects(np.array(s1), np.array(s3)))

print('===INSIDEOUTSIDE (true)===')
print('inside : ', room.isInside(np.array([2,2]), room.corners, True))

print('===INSIDEOUTSIDE (false)===')
print('outside : ', room.isInside(np.array([5,0]), room.corners, True))
   
print('===INSIDEOUTSIDE (true)===')
print('on left border inclusive : ', room.isInside(np.array([0,2]), room.corners, True))

print('===INSIDEOUTSIDE (true)===')
print('on right border inclusive : ', room.isInside(np.array([4,2]), room.corners, True))

print('===INSIDEOUTSIDE (false)===')
print('on left border exclusive : ', room.isInside(np.array([0,2]), room.corners, False))

print('===INSIDEOUTSIDE (false)===')
print('on right border exclusive : ', room.isInside(np.array([4,2]), room.corners, False))

print('===INTERSECTION===')
print(room.intersection(np.array([0,0]), np.array([4,0]), np.array([2, -2]), np.array([2, 2])))

print('===OBSTRUCTION===')
print(room.isObstructed(room.corners, np.array([2,2]), np.array([5,2]), np.array([[4, 0],[4, 4]])))

print(room.walls)