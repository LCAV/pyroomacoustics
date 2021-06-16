import numpy as np

A=np.array([[0,0,0],[0,0,1],[1,1,1]])       # set of multiple points in 3-D
X=np.array([[0,0,0]])                       # set of 1 point in 3-D

B=np.array([0,0,1])                         # point in 3-D
C=np.array([0,0,-1])                        # point in 3-D

A1=np.array([[0,0],[0,1]])              # set of multiple points in 2-D
X1=np.array([[1,1]])                    # set of only 1 point in 2-D

B1=np.array([0,0])                      # point in 2-D
C1=np.array([0,-1])                     # point in 2-D


print("angle function for points A and B returns: ")
print(angle_function(A,B))

print("angle function for points A and C returns: ")
print(angle_function(A,C))

print("angle function for points X and B returns: ")
print(angle_function(X,B))

print("angle function for points X and C returns: ")
print(angle_function(X,C))

print("angle function for points A1 and B1 returns: ")
print(angle_function(A1,B1))

print("angle function for points A1 and C1 returns: ")
print(angle_function(A1,C1))

print("angle function for points X1 and B1 returns: ")
print(angle_function(X1,B1))

print("angle function for points X1 and C1 returns: ")
print(angle_function(X1,C1))
