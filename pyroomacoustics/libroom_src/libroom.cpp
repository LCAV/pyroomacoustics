/*
 * This file contains the bindings for the libroom ISM model
 */
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

#include "geometry.hpp"
#include "utility.hpp"
#include "wall.hpp"
#include "room.hpp"

namespace py = pybind11;

float libroom_eps = 1e-5;  // epsilon is set to 0.01 millimeter (10 um)

Room<3> *create_room_3D(py::list _walls, py::list _obstructing_walls, const Eigen::MatrixXf &_microphones)
{
  /*
   * This is a factory method to isolate the Room class from pybind code
   */
  Room<3> *room = new Room<3>();


  room->microphones = _microphones.topRows(3);

  for (auto wall : _walls)
    room->walls.push_back(wall.cast<Wall<3>>());

  for (auto owall : _obstructing_walls)
    room->obstructing_walls.push_back(owall.cast<int>());

  room->max_dist = room->get_max_distance();
  return room;
}

Room<2> *create_room_2D(py::list _walls, py::list _obstructing_walls, const Eigen::MatrixXf &_microphones)
{
  /*
   * This is a factory method to isolate the Room class from pybind code
   */
  Room<2> *room = new Room<2>();

  room->microphones = _microphones.topRows(2);

  for (auto wall : _walls)
    room->walls.push_back(wall.cast<Wall<2>>());

  for (auto owall : _obstructing_walls)
    room->obstructing_walls.push_back(owall.cast<int>());
    
  room->max_dist = room->get_max_distance();

  return room;
}


PYBIND11_MODULE(libroom, m) {
    m.doc() = "Libroom room simulation extension plugin"; // optional module docstring

    // The 3D Room class
    py::class_<Room<3>>(m, "Room")
      //.def(py::init<py::list, py::list, const Eigen::MatrixXf &>())
      .def(py::init(&create_room_3D))
      .def("image_source_model", &Room<3>::image_source_model)
      .def("image_source_shoebox", &Room<3>::image_source_shoebox)
      .def("get_wall", &Room<3>::get_wall)
      .def("get_max_distance", &Room<3>::get_max_distance)
      .def("next_wall_hit", &Room<3>::next_wall_hit)
      .def("scat_ray", &Room<3>::scat_ray)
      .def("simul_ray", &Room<3>::simul_ray)
      .def("get_rir_entries", &Room<3>::get_rir_entries)  
      .def("contains", &Room<3>::contains)
      .def_readonly("sources", &Room<3>::sources)
      .def_readonly("orders", &Room<3>::orders)
      .def_readonly("attenuations", &Room<3>::attenuations)
      .def_readonly("gen_walls", &Room<3>::gen_walls)
      .def_readonly("visible_mics", &Room<3>::visible_mics)
      .def_readonly("walls", &Room<3>::walls)
      .def_readonly("obstructing_walls", &Room<3>::obstructing_walls)
      .def_readonly("microphones", &Room<3>::microphones)
      .def_readonly("n_mics", &Room<3>::n_mics)
      .def_readonly("max_dist", &Room<3>::max_dist)
      ;

    // The 2D Room class
    py::class_<Room<2>>(m, "Room2D")
      //.def(py::init<py::list, py::list, const Eigen::MatrixXf &>())
      .def(py::init(&create_room_2D))
      .def("image_source_model", &Room<2>::image_source_model)
      .def("image_source_shoebox", &Room<2>::image_source_shoebox)
      .def("get_wall", &Room<2>::get_wall)
      .def("get_max_distance", &Room<2>::get_max_distance)
      .def("next_wall_hit", &Room<2>::next_wall_hit)
      .def("scat_ray", &Room<2>::scat_ray)
      .def("simul_ray", &Room<2>::simul_ray)
      .def("get_rir_entries", &Room<2>::get_rir_entries)  
      .def("contains", &Room<2>::contains)
      .def_readonly("sources", &Room<2>::sources)
      .def_readonly("orders", &Room<2>::orders)
      .def_readonly("attenuations", &Room<2>::attenuations)
      .def_readonly("gen_walls", &Room<2>::gen_walls)
      .def_readonly("visible_mics", &Room<2>::visible_mics)
      .def_readonly("walls", &Room<2>::walls)
      .def_readonly("obstructing_walls", &Room<2>::obstructing_walls)
      .def_readonly("microphones", &Room<2>::microphones)
      .def_readonly("n_mics", &Room<2>::n_mics)
      .def_readonly("max_dist", &Room<2>::max_dist)
      ;

    // The Wall class
    py::class_<Wall<3>> wall_cls(m, "Wall");

    wall_cls
        .def(py::init<const Eigen::Matrix<float,3,Eigen::Dynamic> &, float, const std::string &>(),
            py::arg("corners"), py::arg("absorption") = 0., py::arg("name") = "")
        .def("area", &Wall<3>::area)
        .def("intersection", &Wall<3>::intersection)
        .def("intersects", &Wall<3>::intersects)
        .def("side", &Wall<3>::side)
        .def("reflect", &Wall<3>::reflect)
        .def("same_as", &Wall<3>::same_as)
        .def_readonly("dim", &Wall<3>::dim)
        .def_readwrite("absorption", &Wall<3>::absorption)
        .def_readwrite("name", &Wall<3>::name)
        .def_readonly("corners", &Wall<3>::corners)
        .def_readonly("origin", &Wall<3>::origin)
        .def_readonly("normal", &Wall<3>::normal)
        .def_readonly("basis", &Wall<3>::basis)
        .def_readonly("flat_corners", &Wall<3>::flat_corners)
        ;

    py::enum_<Wall<3>::Isect>(wall_cls, "Isect")
      .value("NONE", Wall<3>::Isect::NONE)
      .value("VALID", Wall<3>::Isect::VALID)
      .value("ENDPT", Wall<3>::Isect::ENDPT)
      .value("BNDRY", Wall<3>::Isect::BNDRY)
      .export_values();

    // The Wall class
    py::class_<Wall<2>> wall2d_cls(m, "Wall2D");

    wall2d_cls
        .def(py::init<const Eigen::Matrix<float,2,Eigen::Dynamic> &, float, const std::string &>(),
            py::arg("corners"), py::arg("absorption") = 0., py::arg("name") = "")
        .def("area", &Wall<2>::area)
        .def("intersection", &Wall<2>::intersection)
        .def("intersects", &Wall<2>::intersects)
        .def("side", &Wall<2>::side)
        .def("reflect", &Wall<2>::reflect)
        .def("same_as", &Wall<2>::same_as)
        .def_readonly("dim", &Wall<2>::dim)
        .def_readwrite("absorption", &Wall<2>::absorption)
        .def_readwrite("name", &Wall<2>::name)
        .def_readonly("corners", &Wall<2>::corners)
        .def_readonly("origin", &Wall<2>::origin)
        .def_readonly("normal", &Wall<2>::normal)
        .def_readonly("basis", &Wall<2>::basis)
        .def_readonly("flat_corners", &Wall<2>::flat_corners)
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
		
	m.def("cos_angle_between", &cos_angle_between,
		"Computes the angle between two 2D or 3D vectors");
	
	m.def("dist_line_point", &dist_line_point,
		"Computes the distance between a point and an infinite line");
	
	
	// Routines for the utility packages
	m.def("equation", &equation,
		"Computes the a and b coefficients in the expression y=ax+b given two points lying on that line.");
		
	m.def("compute_segment_end", &compute_segment_end,
		"Computes the end point of a segment given the start point, the length, and the orientation");
		
	m.def("compute_reflected_end", &compute_reflected_end,
		"This function operates when we know the vector [start, hit_point]. This function computes the end point E so that [hit_point, E] is the reflected vector of [start, hit_point] with the correct magnitude");
	
	m.def("intersects_mic", &intersects_mic,
		"Determines if a segment intersects the microphone of specified center and radius");
		
	m.def("solve_quad", &solve_quad,
		"Solves the quadratic system and outputs real roots");
		
	m.def("mic_intersection", &mic_intersection,
		"Computes the intersection point between the ray and the microphone");
		
	m.def("test", &test, "Test different functions");
	
}

