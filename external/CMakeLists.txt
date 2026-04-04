include(FetchContent)

# eigen
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen
  GIT_TAG 5.0.1
)

# nanoflann
FetchContent_Declare(
  nanoflann
  GIT_REPOSITORY https://github.com/jlblancoc/nanoflann
  GIT_TAG v1.9.0
)

# pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v3.0.1
)

# eigen
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Disable Eigen pkg-config")
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen testing")
set(BUILD_TESTING OFF CACHE BOOL "Disable testing")

# nanoflann
set(NANOFLANN_BUILD_EXAMPLES OFF CACHE BOOL "Disable nanoflann examples" FORCE)
set(NANOFLANN_BUILD_TESTS OFF CACHE BOOL "Disable nanoflann tests" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable nanoflann benchmarks" FORCE)

# pybind11
set(PYBIND11_TEST OFF CACHE BOOL "Disable pybind11 tests")

FetchContent_MakeAvailable(eigen nanoflann pybind11)
