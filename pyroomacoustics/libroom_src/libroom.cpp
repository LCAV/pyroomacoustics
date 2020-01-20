/* 
 * Python bindings for libroom
 * Copyright (C) 2019  Robin Scheibler, Cyril Cadoux
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * You should have received a copy of the MIT License along with this program. If
 * not, see <https://opensource.org/licenses/MIT>.
 */

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

#include "common.hpp"
#include "geometry.hpp"
#include "microphone.hpp"
#include "wall.hpp"
#include "room.hpp"

namespace py = pybind11;

float libroom_eps = 1e-5;  // epsilon is set to 0.01 millimeter (10 um)


PYBIND11_MODULE(libroom, m) {
  m.doc() = "Libroom room simulation extension plugin"; // optional module docstring

  // The 3D Room class
  py::class_<Room<3>>(m, "Room")
    .def(py::init<
        const std::vector<Wall<3>> &,
        const std::vector<int> &,
        const std::vector<Microphone<3>> &,
        float, int, float, float, float, float, bool
        >())
    .def(py::init<
        const Vectorf<3> &,
        const Eigen::Array<float,Eigen::Dynamic,6> &,
        const Eigen::Array<float,Eigen::Dynamic,6> &,
        const std::vector<Microphone<3>> &,
        float, int, float, float, float, float, bool
        >())
    .def("set_params", &Room<3>::set_params)
    .def("add_mic", &Room<3>::add_mic)
    .def("reset_mics", &Room<3>::reset_mics)
    .def("image_source_model", &Room<3>::image_source_model)
    .def("get_wall", &Room<3>::get_wall)
    .def("get_max_distance", &Room<3>::get_max_distance)
    .def("next_wall_hit", &Room<3>::next_wall_hit)
    .def("scat_ray", &Room<3>::scat_ray)
    .def("simul_ray", &Room<3>::simul_ray)
    .def("ray_tracing",
        (void (Room<3>::*)(
                             const Eigen::Matrix<float,2,Eigen::Dynamic> &angles,
                             const Vectorf<3> source_pos
                             )
        )
        &Room<3>::ray_tracing)
    .def("ray_tracing",
        (void (Room<3>::*)(
                             size_t nb_phis,
                             size_t nb_thetas,
                             const Vectorf<3> source_pos
                             )
        )
        &Room<3>::ray_tracing)
    .def("ray_tracing",
        (void (Room<3>::*)(
                             size_t nb_rays,
                             const Vectorf<3> source_pos
                             )
        )
        &Room<3>::ray_tracing)
    .def("contains", &Room<3>::contains)
    .def_property("is_hybrid_sim", &Room<3>::get_is_hybrid_sim, &Room<3>::set_is_hybrid_sim)
    .def_property_readonly_static("dim", [](py::object /* self */) { return 3; })
    .def_readonly("walls", &Room<3>::walls)
    .def_readonly("sources", &Room<3>::sources)
    .def_readonly("orders", &Room<3>::orders)
    .def_readonly("attenuations", &Room<3>::attenuations)
    .def_readonly("gen_walls", &Room<3>::gen_walls)
    .def_readonly("visible_mics", &Room<3>::visible_mics)
    .def_readonly("walls", &Room<3>::walls)
    .def_readonly("obstructing_walls", &Room<3>::obstructing_walls)
    .def_readonly("microphones", &Room<3>::microphones)
    .def_readonly("max_dist", &Room<3>::max_dist)
    ;

  // The 2D Room class
  py::class_<Room<2>>(m, "Room2D")
    //.def(py::init<py::list, py::list, const Eigen::MatrixXf &>())
    .def(py::init<
        const std::vector<Wall<2>> &,
        const std::vector<int> &,
        const std::vector<Microphone<2>> &,
        float, int, float, float, float, float, bool
        >())
    .def(py::init<
        const Vectorf<2> &,
        const Eigen::Array<float,Eigen::Dynamic,4> &,
        const Eigen::Array<float,Eigen::Dynamic,4> &,
        const std::vector<Microphone<2>> &,
        float, int, float, float, float, float, bool
        >())
    .def("set_params", &Room<2>::set_params)
    .def("add_mic", &Room<2>::add_mic)
    .def("reset_mics", &Room<2>::reset_mics)
    .def("image_source_model", &Room<2>::image_source_model)
    .def("get_wall", &Room<2>::get_wall)
    .def("get_max_distance", &Room<2>::get_max_distance)
    .def("next_wall_hit", &Room<2>::next_wall_hit)
    .def("scat_ray", &Room<2>::scat_ray)
    .def("simul_ray", &Room<2>::simul_ray)
    .def("ray_tracing",
        (void (Room<2>::*)(
                             const Eigen::Matrix<float,1,Eigen::Dynamic> &angles,
                             const Vectorf<2> source_pos
                            )
        )
        &Room<2>::ray_tracing)
    .def("ray_tracing",
        (void (Room<2>::*)(
                             size_t nb_phis,
                             size_t nb_thetas,
                             const Vectorf<2> source_pos
                            )
        )
        &Room<2>::ray_tracing)
    .def("ray_tracing",
        (void (Room<2>::*)(
                             size_t n_rays,
                             const Vectorf<2> source_pos
                            )
        )
        &Room<2>::ray_tracing)
    .def("contains", &Room<2>::contains)
    .def_property_readonly_static("dim", [](py::object /* self */) { return 2; })
    .def_property("is_hybrid_sim", &Room<2>::get_is_hybrid_sim, &Room<2>::set_is_hybrid_sim)
    .def_readonly("walls", &Room<2>::walls)
    .def_readonly("sources", &Room<2>::sources)
    .def_readonly("orders", &Room<2>::orders)
    .def_readonly("attenuations", &Room<2>::attenuations)
    .def_readonly("gen_walls", &Room<2>::gen_walls)
    .def_readonly("visible_mics", &Room<2>::visible_mics)
    .def_readonly("walls", &Room<2>::walls)
    .def_readonly("obstructing_walls", &Room<2>::obstructing_walls)
    .def_readonly("microphones", &Room<2>::microphones)
    .def_readonly("max_dist", &Room<2>::max_dist)
    ;

  // The Wall class
  py::class_<Wall<3>> wall_cls(m, "Wall");

  wall_cls
    .def(py::init<const Eigen::Matrix<float,3,Eigen::Dynamic> &, const Eigen::ArrayXf &, const Eigen::ArrayXf &, const std::string &>(),
        py::arg("corners"), py::arg("absorption") = Eigen::ArrayXf::Zero(1),
        py::arg("scattering") = Eigen::ArrayXf::Zero(1), py::arg("name") = "")
    .def("area", &Wall<3>::area)
    .def("intersection", &Wall<3>::intersection)
    .def("intersects", &Wall<3>::intersects)
    .def("side", &Wall<3>::side)
    .def("reflect", &Wall<3>::reflect)
    .def("normal_reflect", &Wall<3>::normal_reflect)
    .def("same_as", &Wall<3>::same_as)
    .def_property_readonly_static("dim", [](py::object /* self */) { return 3; })
    .def_readwrite("absorption", &Wall<3>::absorption)
    .def_readwrite("scatter", &Wall<3>::scatter)
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
    .def(py::init<const Eigen::Matrix<float,2,Eigen::Dynamic> &, const Eigen::ArrayXf &, const Eigen::ArrayXf &, std::string &>(),
        py::arg("corners"), py::arg("absorption") = Eigen::ArrayXf::Zero(1),
        py::arg("scattering") = Eigen::ArrayXf::Zero(1), py::arg("name") = "")
    .def("area", &Wall<2>::area)
    .def("intersection", &Wall<2>::intersection)
    .def("intersects", &Wall<2>::intersects)
    .def("side", &Wall<2>::side)
    .def("reflect", &Wall<2>::reflect)
    .def("normal_reflect", &Wall<2>::normal_reflect)
    .def("same_as", &Wall<2>::same_as)
    .def_property_readonly_static("dim", [](py::object /* self */) { return 2; })
    .def_readwrite("absorption", &Wall<2>::absorption)
    .def_readwrite("scatter", &Wall<2>::scatter)
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

  // The microphone class
  py::class_<Microphone<3>>(m, "Microphone")
    .def(py::init<const Vectorf<3> &, int, float, float>())
    .def_readonly("loc", &Microphone<3>::loc)
    .def_readonly("hits", &Microphone<3>::hits)
    .def_readonly("histograms", &Microphone<3>::histograms)
    ;

  py::class_<Microphone<2>>(m, "Microphone2D")
    .def(py::init<const Vectorf<2> &, int, float, float>())
    .def_readonly("loc", &Microphone<2>::loc)
    .def_readonly("hits", &Microphone<2>::hits)
    .def_readonly("histograms", &Microphone<2>::histograms)
    ;

  // The 2D histogram class
  py::class_<Histogram2D>(m, "Histogram2D")
    .def(py::init<int, int>())
    .def("log", &Histogram2D::log)
    .def("bin", &Histogram2D::bin)
    .def("get_hist", &Histogram2D::get_hist)
    .def("reset", &Histogram2D::reset)
    ;

  // Structure to hold detector hit information
  py::class_<Hit>(m, "Hit")
    .def(py::init<int>())
    .def(py::init<const float, const Eigen::ArrayXf &>())
    .def_readonly("transmitted", &Hit::transmitted)
    .def_readonly("distance", &Hit::distance)
    ;

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

}

