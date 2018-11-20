/*
 * This file contains the bindings for the libroom ISM model
 */
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "geometry.hpp"
#include "wall.hpp"
#include "room.hpp"

namespace py = pybind11;

float libroom_eps = 1e-5;  // epsilon is set to 0.01 millimeter (10 um)

PYBIND11_MODULE(_libroom, m) {
    m.doc() = "Libroom room simulation extension plugin"; // optional module docstring

    // The Room class
    py::class_<Room>(m, "Room")
      .def(py::init<py::list, py::list, const Eigen::MatrixXf &>())
      .def("image_source_model", &Room::image_source_model)
      .def("get_wall", &Room::get_wall)
      .def_readonly("sources", &Room::sources)
      .def_readonly("attenuations", &Room::attenuations)
      .def_readonly("parents", &Room::parents)
      .def_readonly("gen_walls", &Room::gen_walls)
      .def_readonly("visible_mics", &Room::visible_mics)
      .def_readonly("walls", &Room::walls)
      .def_readonly("obstructing_walls", &Room::obstructing_walls)
      .def_readonly("microphones", &Room::microphones)
      ;

    // The Wall class
    py::class_<Wall>(m, "Wall")
        .def(py::init<const Eigen::MatrixXf &, float>())
        .def("area", &Wall::area)
        .def("intersection", &Wall::intersection)
        .def("side", &Wall::side)
        .def("reflect", &Wall::reflect)
        .def_readonly("dim", &Wall::dim)
        .def_readwrite("absorption", &Wall::absorption)
        .def_readonly("corners", &Wall::corners)
        .def_readonly("origin", &Wall::origin)
        .def_readonly("normal", &Wall::normal)
        .def_readonly("basis", &Wall::basis)
        .def_readonly("flat_corners", &Wall::flat_corners)
        ;

    // The different wall intersection cases
    m.attr("WALL_ISECT_NONE") = WALL_ISECT_NONE;
    m.attr("WALL_ISECT_VALID") = WALL_ISECT_VALID;
    m.attr("WALL_ISECT_VALID_ENDPT") = WALL_ISECT_VALID_ENDPT;
    m.attr("WALL_ISECT_VALID_BNDRY") = WALL_ISECT_VALID_BNDRY;

    // getter and setter for geometric epsilon
    m.def("set_eps", [](const float &eps) { libroom_eps = eps; });
    m.def("get_eps", []() { return libroom_eps; });

    // Routines for the geometry packages
    m.def("ccw3p", &ccw3p, "Determines the orientation of three points");

    m.def("check_intersection_2d_segments",
        &check_intersection_2d_segments,
        "A function that checks if two line segments intersect");

    m.def("intersection_2d_segments",
        &intersection_2d_segments,
        "A function that finds the intersection of two line segments");

    m.def("intersection_3d_segment_plane",
        &intersection_3d_segment_plane,
        "A function that finds the intersection between a line segment and a plane");

    m.def("cross", &cross, "Cross product of two 3D vectors");

    m.def("is_inside_2d_polygon", &is_inside_2d_polygon,
        "Checks if a 2D point lies in or out of a planar polygon");

    m.def("area_2d_polygon", &area_2d_polygon,
        "Compute the signed area of a planar polygon");
}

