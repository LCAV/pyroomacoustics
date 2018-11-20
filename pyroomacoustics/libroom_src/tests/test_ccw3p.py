import numpy as np
import pyroomacoustics as pra

cases = {
        'anti-clockwise' : {
            'points' : np.array([
                [1,-1], [2,-1], [1,0]
                ]),
            'expected' : 1,  # anti-clockwise
            'label' : 'Test: CCW3P anti-clockwise',
            },
        'clockwise' : {
            'points' : np.array([
                [1,-1], [1,0], [2,-1]
                ]),
            'expected' : -1,  # clockwise
            'label' : 'Test: CCW3P clockwise',
            },
        'co-linear' : {
            'points' : np.array([
                [0,0], [0.5,0.5], [1,1]
                ]),
            'expected' : 0,  # co-linear
            'label' : 'Test: CCW3P co-linear',
            },
        }

def ccw3p(case):

    p1, p2, p3 = case['points']

    r = pra.libroom.ccw3p(p1, p2, p3)

    assert r == case['expected'], (case['label']
            + ' returned: {}, expected {}'.format(r, case['expected']))

def test_ccw3p_anticlockwise():
    ccw3p(cases['anti-clockwise'])

def test_ccw3p_clockwise():
    ccw3p(cases['clockwise'])

def test_ccw3p_colinear():
    ccw3p(cases['co-linear'])

if __name__ == '__main__':

    for lbl, case in cases.items():

        try:
            ccw3p(case)
        except:
            print('{} failed'.format(lbl))
