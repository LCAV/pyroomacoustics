import numpy as np
import pyroomacoustics as pra

eps = 1e-6

corners = {
        '3d' : np.array([
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
            ]),
        '2d' : np.array([
            [ -1, 1 ],
            [ -2, 2 ],
            ]),
        }

points = {
        '3d_up' : {
            'd' : '3d',
            'p' : [-1,-1,-1],
            'expect' : -1,
            'reflect' : [5/3, 5/3, 5/3],
            },
        '3d_down' : {
            'd' : '3d',
            'p' : [1,1,1],
            'expect' : 1,
            'reflect' : [-1/3, -1/3, -1/3],
            },
        '3d_on' : {
            'd' : '3d',
            'p' : [1/3, 1/3, 1/3],
            'expect' : 0,
            'reflect' : [1/3, 1/3, 1/3],
            },

        '2d_down' : {
            'd' : '2d',
            'p' : [-2, 1],
            'expect' : -1,
            'reflect' : [2, -1],
            },
        '2d_up' : {
            'd' : '2d',
            'p' : [2, -1],
            'expect' : 1,
            'reflect' : [-2, 1],
            },
        '2d_on' : {
            'd' : '2d',
            'p' : [0.5, 1],
            'expect' : 0,
            'reflect' : [0.5, 1],
            },
        }

def run_side(label):
    p = points[label]['p']
    r_exp = points[label]['expect']
    d = points[label]['d']

    wall = pra.wall_factory(corners[d], 0.1)
    r = wall.side(p)

    assert r == r_exp, 'failed side - {} : returned={} expected={}'.format(label, r, r_exp)

def run_reflect(label):

    p = points[label]['p']
    p_refl = np.array(points[label]['reflect'])
    r_exp = points[label]['expect']
    d = points[label]['d']

    wall = pra.wall_factory(corners[d], 0.1)

    x = np.zeros(wall.dim, dtype=np.float32)
    wall.reflect(p, x)

    err = np.linalg.norm(x - p_refl) < eps

    assert err, 'failed reflect - {} : error={}'.format(label, err)

def test_side_3d_up():
    run_side('3d_up')

def test_side_3d_down():
    run_side('3d_down')

def test_side_3d_on():
    run_side('3d_on')

def test_side_2d_up():
    run_side('2d_up')

def test_side_2d_down():
    run_side('2d_down')

def test_side_2d_on():
    run_side('2d_on')

def test_reflect_3d_up():
    run_reflect('3d_up')

def test_reflect_3d_down():
    run_reflect('3d_down')

def test_reflect_3d_on():
    run_reflect('3d_on')

def test_reflect_2d_up():
    run_reflect('2d_up')

def test_reflect_2d_down():
    run_reflect('2d_down')

def test_reflect_2d_on():
    run_reflect('2d_on')

if __name__ == '__main__':

    test_side_3d_up()
    test_side_3d_down()
    test_side_3d_on()
    test_reflect_3d_up()
    test_reflect_3d_down()
    test_reflect_3d_on()

    test_side_2d_up()
    test_side_2d_down()
    test_side_2d_on()
    test_reflect_2d_up()
    test_reflect_2d_down()
    test_reflect_2d_on()
