cmake_minimum_required(VERSION 3.15)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

project(pyroomacoustics)

# Find pybind11 (handled by external/CMakeLists.txt via FetchContent)
set(PYBIND11_FINDPYTHON ON)

# adds the external dependencies
add_subdirectory(external)

# Source directory for libroom
set(LIBROOM_SRC_DIR pyroomacoustics/libroom_src)

# Collect source files
set(LIBROOM_SOURCES
    ${LIBROOM_SRC_DIR}/libroom.cpp
)

# Define the extension module
pybind11_add_module(libroom ${LIBROOM_SOURCES})

# Include directories
target_include_directories(libroom PRIVATE
    .
    ${LIBROOM_SRC_DIR}
)

# Link dependencies from FetchContent
target_link_libraries(libroom PRIVATE
    Eigen3::Eigen
    nanoflann::nanoflann
)

# Compile options
if(MSVC)
    target_compile_options(libroom PRIVATE /EHsc)
else()
    target_compile_options(libroom PRIVATE -O3 -Wall -fvisibility=hidden)
endif()

# Eigen definitions
target_compile_definitions(libroom PRIVATE EIGEN_MPL2_ONLY EIGEN_NO_DEBUG)

# Setting the output name and location
set_target_properties(libroom PROPERTIES
    PREFIX ""
    OUTPUT_NAME "libroom"
)
