// Included by geometry_utils.hpp
#include <igl/polygons_to_triangles.h>
#include <igl/remove_duplicate_vertices.h>
#include <iostream>

std::tuple<Eigen::MatrixXf, Eigen::MatrixXi, std::vector<int>>
triangulate_walls(const std::vector<Wall<3>> &walls) {
  std::vector<Eigen::Vector3f> all_vertices;
  std::vector<Eigen::Vector3i> all_faces;
  std::vector<int> face_to_wall;

  for (size_t i = 0; i < walls.size(); ++i) {
    const auto &wall = walls[i];
    int n_corners = wall.flat_corners.cols();

    // Local 2D coordinates for triangulation
    Eigen::MatrixXd V2(n_corners, 2);
    for (int j = 0; j < n_corners; ++j) {
      V2(j, 0) = (double)wall.flat_corners(0, j);
      V2(j, 1) = (double)wall.flat_corners(1, j);
    }

    // Cumulative indices for polygons_to_triangles
    // For a single polygon, C = [0, n_corners]
    Eigen::VectorXi C(2);
    C << 0, n_corners;
    Eigen::VectorXi I(n_corners);
    for (int j = 0; j < n_corners; ++j)
      I(j) = j;

    Eigen::MatrixXi F_wall;
    Eigen::VectorXi J_wall;
    igl::polygons_to_triangles(I, C, F_wall, J_wall);

    // Map wall local vertex indices to global vertex indices
    int base_v_idx = all_vertices.size();
    for (int j = 0; j < n_corners; ++j) {
      all_vertices.push_back(wall.corners.col(j).cast<float>());
    }

    for (int j = 0; j < F_wall.rows(); ++j) {
      all_faces.push_back(Eigen::Vector3i(base_v_idx + F_wall(j, 0),
                                          base_v_idx + F_wall(j, 1),
                                          base_v_idx + F_wall(j, 2)));
      face_to_wall.push_back(i);
    }
  }

  // Convert to Eigen matrices
  Eigen::MatrixXf V(all_vertices.size(), 3);
  for (size_t i = 0; i < all_vertices.size(); ++i)
    V.row(i) = all_vertices[i];

  Eigen::MatrixXi F(all_faces.size(), 3);
  for (size_t i = 0; i < all_faces.size(); ++i)
    F.row(i) = all_faces[i];

  // Remove duplicate vertices to ensure a watertight mesh
  // This is important for AABB tree reliability
  Eigen::MatrixXf V_unique;
  Eigen::MatrixXi F_unique;
  Eigen::VectorXi S, _I;
  igl::remove_duplicate_vertices(V, F, 1e-5, V_unique, S, _I, F_unique);

  return std::make_tuple(V_unique, F_unique, face_to_wall);
}
