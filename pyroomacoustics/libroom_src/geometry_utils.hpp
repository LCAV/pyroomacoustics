#ifndef __GEOMETRY_UTILS_H__
#define __GEOMETRY_UTILS_H__

#include "common.hpp"
#include "wall.hpp"
#include <Eigen/Dense>
#include <tuple>
#include <vector>

/**
 * @brief Triangulates a set of walls into a single mesh.
 *
 * @param walls Vector of Wall objects (3D).
 * @return std::tuple<Eigen::MatrixXf, Eigen::MatrixXi, std::vector<int>>
 *         - V: Matrix of vertices (n_vertices x 3)
 *         - F: Matrix of faces (n_faces x 3)
 *         - face_to_wall: Vector mapping each face to its original wall index.
 */
std::tuple<Eigen::MatrixXf, Eigen::MatrixXi, std::vector<int>>
triangulate_walls(const std::vector<Wall<3>> &walls);

#include "geometry_utils.cpp"

#endif // __GEOMETRY_UTILS_H__
