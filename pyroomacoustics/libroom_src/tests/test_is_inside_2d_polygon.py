import numpy as np
import pyroomacoustics as pra

polygons = [
        np.array([  # this one is clockwise
            [0, 4, 4, 0,],
            [0, 0, 4, 4,],
            ]),

        np.array([  # this one is clockwise!
            [0, 0, 1, 1, 3, 3],
            [0, 1, 1, 2, 2, 0],
            ]),

        np.array([  # this one is clockwise!
            [0, 1, 1, 3, 3],
            [0, 1, 2, 2, 0],
            ]),
        ]

cases = {
        'inside' : {
            'pol' : 0,
            'p' : [2,2],
            'ret' : 0,
            },

        'on_border' : {
            'pol' : 0,
            'p' : [0,2],
            'ret' : 1,
            },

        'on_corner' : {
            'pol' : 0,
            'p' : [4,4],
            'ret' : 1,
            },

        'outside' : {
            'pol' : 0,
            'p' : [5,5],
            'ret' : -1,
            },

        # horizontal wall aligned with point
        'horiz_wall_align' : {
            'pol' : 1,
            'p' : [2,1],
            'ret' : 0,
            },
        # ray is going through vertex
        'ray_through_vertex' : {
            'pol' : 2,
            'p' : [2,1],
            'ret' : 0,
            },

        # point is at the same height as top of polygon, but outside
        'top_outside' : {
            'pol' : 2,
            'p' : [4,2],
            'ret' : -1,
            },
        }

def run_inside_pol(lbl):
    pol = polygons[cases[lbl]['pol']]
    p = cases[lbl]['p']
    r_exp = cases[lbl]['ret']

    ret = pra.libroom.is_inside_2d_polygon(p, pol)

    assert ret == r_exp, '{} : returned={} expected={}'.format(lbl, ret, r_exp)

def test_inside():
    run_inside_pol('inside')

def test_on_border():
    run_inside_pol('on_border')

def test_on_corner():
    run_inside_pol('on_corner')

def test_outside():
    run_inside_pol('outside')

def test_horiz_wall_align():
    run_inside_pol('horiz_wall_align')

def test_ray_through_vertex():
    run_inside_pol('ray_through_vertex')

def test_top_outside():
    run_inside_pol('top_outside')

if __name__ == '__main__':

    test_inside()
    test_on_border()
    test_on_corner()
    test_outside()
    test_top_outside()
    test_horiz_wall_align()
    test_ray_through_vertex()
