import numpy as np
from pyroomacoustics import angle_function
import unittest 


pi=np.pi

a1=np.array([[0,0,0],[0,0,1],[1,1,1]]) 
a2=np.array([[0,0,1],[0,0,1]])
a3=np.array([[0,0,1]])
    
b1=np.array([0,0,0])
b2=np.array([1,-1,1])
b3=np.array([1,0,1])                        

c1=np.array([[0,0],[0,0]])
c2=np.array([[1,0],[0,-1],[0,0]])
c3=np.array([[1,1]])

d1=np.array([0,0])
d2=np.array([0,1])
d3=np.array([1,-1])


class TestAngleFunction(unittest.TestCase):
    def test_angle_function(self):

        self.assertTrue(angle_function(a1,b1).all()==np.array([[0,0],[0,pi/2],[pi/4,pi/4]]).all())
        self.assertTrue(angle_function(a1,b2).all()==np.array([[3*pi/4,-pi/4],[3*pi/4,0],[pi/2,0]]).all())
        self.assertTrue(angle_function(a1,b3).all()==np.array([[pi,-1.10714872],[pi,0],[pi/2,0]]).all())
        self.assertTrue(angle_function(a2,b1).all()==np.array([[0,pi/2],[0,pi/2]]).all())
        self.assertTrue(angle_function(a2,b2).all()==np.array([[3*pi/4,0],[3*pi/4,0]]).all())
        self.assertTrue(angle_function(a2,b3).all()==np.array([[pi,0],[pi,0]]).all())
        self.assertTrue(angle_function(a3,b1).all()==np.array([[0,pi/2]]).all())
        self.assertTrue(angle_function(a3,b2).all()==np.array([[3*pi/4,0]]).all())
        self.assertTrue(angle_function(a3,b3).all()==np.array([[pi,0]]).all())

        self.assertTrue(angle_function(c1,d1).all()==np.array([[0,0],[0,0]]).all())
        self.assertTrue(angle_function(c1,d2).all()==np.array([[-pi/2,0],[-pi/2,0]]).all())
        self.assertTrue(angle_function(c1,d3).all()==np.array([[3*pi/4,0],[3*pi/4,0]]).all())
        self.assertTrue(angle_function(c2,d1).all()==np.array([[0,0],[-pi/2,0],[pi/4,0]]).all())
        self.assertTrue(angle_function(c2,d2).all()==np.array([[-pi/4,0],[-pi/2,0],[-pi/2,0]]).all())
        self.assertTrue(angle_function(c2,d3).all()==np.array([[pi/2,0],[pi,0],[3*pi/4,0]]).all())
        self.assertTrue(angle_function(c3,d1).all()==np.array([[pi/4,0]]).all())
        self.assertTrue(angle_function(c3,d2).all()==np.array([[0,0]]).all())
        self.assertTrue(angle_function(c3,d3).all()==np.array([[pi/2,0]]).all())


if __name__=='__main__':
    unittest.main()
