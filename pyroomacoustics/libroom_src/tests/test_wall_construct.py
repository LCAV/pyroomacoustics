import numpy as np
import pyroomacoustics as pra

eps = 1e-6

# The vertices of the wall are assumed to turn counter-clockwise around the
# normal of the wall

# A very simple wall
walls = [
        {
            'corners' : np.array([
                [0., 1., 1., 0.],
                [0., 0., 1., 1.],
                [0., 0., 0., 0.],
                ]),
            'area' : 1,
            'absorption' : 0.2,
            },
        {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'area' : 3.4641016151377557,  # this is an equilateral triangle with side sqrt(8)
            'absorption' : 0.2,
            },
        ]



def test_wall_3d_construct_0():
    ''' Tests construction of a wall '''
    w_info = walls[0]
    wall = pra.libroom.Wall(w_info['corners'], 0.2)

    return wall

def test_wall_3d_construct_1():
    ''' Tests construction of a wall '''
    w_info = walls[1]
    wall = pra.libroom.Wall(w_info['corners'], 0.2)

    return wall

def test_wall_3d_area_0():
    ''' Tests the area computation '''
    w_info = walls[0]
    wall = pra.libroom.Wall(w_info['corners'], w_info['absorption'])
    err = abs(wall.area() - w_info['area'])
    assert err < 1, 'The error is {}'.format(err)

def test_wall_3d_area_1():
    ''' Tests the area computation '''
    w_info = walls[1]
    wall = pra.libroom.Wall(w_info['corners'], w_info['absorption'])
    err = abs(wall.area() - w_info['area'])
    assert err < 1, 'The error is {}'.format(err)

def test_wall_3d_normal_0():
    ''' Tests direction of normal wrt to point arrangement '''
    w_info = walls[0]
    wall1 = pra.libroom.Wall(w_info['corners'], 0.2)
    # the same wall with normal pointing the other way
    wall2 = pra.libroom.Wall(w_info['corners'][:,::-1], 0.2)

    err = np.linalg.norm(wall1.normal + wall2.normal)
    assert err < eps, 'The error is {}'.format(err)

def test_wall_3d_normal_1():
    ''' Tests direction of normal wrt to point arrangement '''
    w_info = walls[1]
    wall1 = pra.libroom.Wall(w_info['corners'], 0.2)
    # the same wall with normal pointing the other way
    wall2 = pra.libroom.Wall(w_info['corners'][:,::-1], 0.2)

    err = np.linalg.norm(wall1.normal + wall2.normal)
    assert err < eps, 'The error is {}'.format(err)

if __name__ == '__main__':

        wall0 = test_wall_3d_construct_0()
        test_wall_3d_normal_0()
        test_wall_3d_area_0()
        wall1 = test_wall_3d_construct_1()
        test_wall_3d_normal_1()
        test_wall_3d_area_1()

