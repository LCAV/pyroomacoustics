#ifndef __RIR_BUILDER_HPP__
#define __RIR_BUILDER_HPP__

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void threaded_rir_builder(py::buffer rir, const py::buffer time,
                          const py::buffer alpha, const py::buffer visibility,
                          int fs, size_t fdl, size_t lut_gran,
                          size_t num_threads);

#endif  // __RIR_BUILDER_HPP__
