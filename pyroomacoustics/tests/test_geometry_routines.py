# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

from unittest import TestCase

import pyroomacoustics as pra

class TestGeometryRoutines(TestCase):

    def test_area_square(self):
        self.assertEquals(pra.geometry.area([[0, 4, 4, 0], [0, 0, 4, 4]]), 4*4)

    def test_area_triangle(self):
        self.assertEquals(pra.geometry.area([[0, 4, 2], [0, 0, 2]]), (4*2)/2)

    def test_side_left(self):
        self.assertEquals(pra.geometry.side([-1, 0], [0, 0], [1, 0]), -1)

    def test_side_right(self):
        self.assertEquals(pra.geometry.side([1, 0], [0, 0], [1, 0]), 1)

    def test_side_middle(self):
        self.assertEquals(pra.geometry.side([0, 1], [0, 0], [1, 0]), 0)

    def test_ccw3p_counterclockwise(self):
        self.assertEquals(pra.geometry.ccw3p([0, 0], [2, 0], [1, 1]), 1)

    def test_ccw3p_clockwise(self):
        self.assertEquals(pra.geometry.ccw3p([1, 1], [2, 0], [0, 0]), -1)

    def test_ccw3p_collinear(self):
        self.assertEquals(pra.geometry.ccw3p([0, 0], [1, 0], [2, 0]), 0)

    def test_intersection2DSegments_cross(self):
        p, endOfA, endOfB = pra.geometry.intersection_2D_segments([-2, 0], [2, 0], [0, -2], [0, 2])
        i = all(p==[0, 0])
        self.assertTrue(all([i, not endOfA, not endOfB]))

    def test_intersection2DSegments_T(self):
        p, endOfA, endOfB = pra.geometry.intersection_2D_segments([0, 0], [2, 0], [2, -2], [2, 2])
        i = all(p==[2, 0])
        self.assertTrue(all([i, endOfA, not endOfB]))

    def test_intersection2DSegments_T2(self):
        p, endOfA, endOfB = pra.geometry.intersection_2D_segments([-2, 0], [2, 0], [0, -2], [0, 0])
        i = all(p==[0, 0])
        self.assertTrue(all([i, not endOfA, endOfB]))

    def test_intersection2DSegments_L(self):
        p, endOfA, endOfB = pra.geometry.intersection_2D_segments([0, 0], [2, 0], [2, 2], [2, 0])
        i = all(p==[2, 0])
        self.assertTrue(all([i, endOfA, endOfB]))

    def test_intersection2DSegments_parallel(self):
        p, endOfA, endOfB = pra.geometry.intersection_2D_segments([0, 0], [2, 0], [0, 1], [2, 1])
        i = p is None
        self.assertTrue(all([i, not endOfA, not endOfB]))

    def test_intersection2DSegments_notTouching(self):
        p, endOfA, endOfB = pra.geometry.intersection_2D_segments([0, 0], [2, 0], [1, 4], [1, 1])
        i = p is None
        self.assertTrue(all([i, not endOfA, not endOfB]))

    def test_intersectionSegmentPlane_through(self):
        p, endOfSegment = pra.geometry.intersection_segment_plane([2, 2, 2], [2, 2, -2], [2, 2, 0], [0, 0, -1])
        i = all(p==[2, 2, 0])
        self.assertTrue(all([i, not endOfSegment]))

    def test_intersectionSegmentPlane_touching(self):
        p, endOfSegment = pra.geometry.intersection_segment_plane([2, 2, 2], [2, 2, 0], [2, 2, 0], [0, 0, -1])
        i = all(p==[2, 2, 0])
        self.assertTrue(all([i, endOfSegment]))

    def test_intersectionSegmentPlane_notTouching(self):
        p, endOfSegment = pra.geometry.intersection_segment_plane([2, 2, 2], [2, 2, 1], [2, 2, 0], [0, 0, -1])
        i = p is None
        self.assertTrue(all([i, not endOfSegment]))

    def test_intersectionSegmentPlane_inside(self):
        p, endOfSegment = pra.geometry.intersection_segment_plane([0, 2, 0], [2, 2, 0], [2, 2, 0], [0, 0, -1])
        i = p is None
        self.assertTrue(all([i, not endOfSegment]))

    def test_intersectionSegmentPolygonSurface_through(self):
        wall = pra.Wall([[0, 4, 4, 0], [0, 0, 4, 4], [0, 0, 0, 0]])
        p, endOfSegment, onBorder = pra.geometry.intersection_segment_polygon_surface([2, 2, 2], [2, 2, -2], wall.corners_2d, wall.normal,
                wall.plane_point, wall.plane_basis)
        i = all(p==[2, 2, 0])
        self.assertTrue(all([i, not endOfSegment, not onBorder]))

    def test_intersectionSegmentPolygonSurface_touching(self):
        wall = pra.Wall([[0, 4, 4, 0], [0, 0, 4, 4], [0, 0, 0, 0]])
        p, endOfSegment, onBorder = pra.geometry.intersection_segment_polygon_surface([2, 2, 2], [2, 2, 0], wall.corners_2d, wall.normal,
                wall.plane_point, wall.plane_basis)
        i = all(p==[2, 2, 0])
        self.assertTrue(all([i, endOfSegment, not onBorder]))

    def test_intersectionSegmentPolygonSurface_border(self):
        wall = pra.Wall([[0, 4, 4, 0], [0, 0, 4, 4], [0, 0, 0, 0]])
        p, endOfSegment, onBorder = pra.geometry.intersection_segment_polygon_surface([0, 0, 2], [0, 0, -2], wall.corners_2d, wall.normal,
                wall.plane_point, wall.plane_basis)
        i = all(p==[0, 0, 0])
        self.assertTrue(all([i, not endOfSegment, onBorder]))

    def test_intersectionSegmentPolygonSurface_miss(self):
        wall = pra.Wall([[0, 4, 4, 0], [0, 0, 4, 4], [0, 0, 0, 0]])
        p, endOfSegment, onBorder = pra.geometry.intersection_segment_polygon_surface([-1, -1, 2], [-1, -1, -2], wall.corners_2d, wall.normal,
                wall.plane_point, wall.plane_basis)
        i = p is None
        self.assertTrue(all([i, not endOfSegment, not onBorder]))
        
    def test_isInside2DPolygon_inside(self):
        inside, onBorder = pra.geometry.is_inside_2D_polygon([2, 2], [[0, 4, 4, 0], [0, 0, 4, 4]])
        self.assertTrue(all([inside, not onBorder]))
        
    def test_isInside2DPolygon_onBorder(self):
        inside, onBorder = pra.geometry.is_inside_2D_polygon([0, 2], [[0, 4, 4, 0], [0, 0, 4, 4]])
        self.assertTrue(all([inside, onBorder]))
        
    def test_isInside2DPolygon_onCorner(self):
        inside, onBorder = pra.geometry.is_inside_2D_polygon([4, 4], [[0, 4, 4, 0], [0, 0, 4, 4]])
        self.assertTrue(all([inside, onBorder]))
        
    def test_isInside2DPolygon_outside(self):
        inside, onBorder = pra.geometry.is_inside_2D_polygon([5, 5], [[0, 4, 4, 0], [0, 0, 4, 4]])
        self.assertTrue(all([not inside, not onBorder]))

    def test_isInside2DPolygon_parallel_wall(self):
        corners = [[0, 0, 1, 1, 3, 3], [0, 1, 1, 2, 2, 0]]
        inside, onBorder = pra.geometry.is_inside_2D_polygon([2, 1], corners)
        self.assertTrue(all([inside, not onBorder]))

    def test_isInside2DPolygon_at_corner(self):
        corners = [[0, 1, 1, 3, 3], [0, 1, 2, 2, 0]]
        inside, onBorder = pra.geometry.is_inside_2D_polygon([2, 1], corners)
        self.assertTrue(all([inside, not onBorder]))

    def test_isInside2DPolygon_just_out(self):
        corners = [[0, 1, 1, 3, 3], [0, 1, 2, 2, 0]]
        inside, onBorder = pra.geometry.is_inside_2D_polygon([4, 2], corners)
        self.assertTrue(all([not inside, not onBorder]))
