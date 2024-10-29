/*
 * Threaded function to build RIR from the locations and attenuations of image
 * sources Copyright (C) 2022  Robin Scheibler
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * You should have received a copy of the MIT License along with this program.
 * If not, see <https://opensource.org/licenses/MIT>.
 */

#include "rir_builder.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "threadpool.hpp"

// A constexpr function to give PI as float or double
template <class T>
constexpr T get_pi() {
  return T(3.14159265358979323846);
} /* pi */

template <class T>
void threaded_rir_builder_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> rir,
    const py::array_t<T, py::array::c_style | py::array::forcecast> &time,
    const py::array_t<T, py::array::c_style | py::array::forcecast> &alpha,
    int fs, size_t fdl, size_t lut_gran, size_t num_threads) {

  auto pi = get_pi<T>();

  // accessors for the arrays
  auto rir_acc = rir.mutable_unchecked();
  auto tim_acc = time.unchecked();
  auto alp_acc = alpha.unchecked();

  size_t n_times = time.size();
  size_t fdl2 = (fdl - 1) / 2;
  size_t rir_len = rir.size();

  // error handling
  if (n_times != size_t(alpha.size()))
    throw std::runtime_error("time and alpha arrays should have the same size");
  if (fdl % 2 != 1)
    throw std::runtime_error("the fractional filter length should be odd");

  // find minimum and maximum times
  py::buffer_info tim_buf = time.request();
  T *tim_ptr = (T *)(tim_buf.ptr);
  auto t_max = *std::max_element(tim_ptr, tim_ptr + tim_buf.size);
  auto t_min = *std::min_element(tim_ptr, tim_ptr + tim_buf.size);
  int max_sample = int(std::ceil(fs * t_max)) + fdl2;
  int min_sample = int(std::floor(fs * t_min)) - fdl2;

  if (min_sample < 0)
    throw std::runtime_error("minimum time recorded is less than 0");
  if (max_sample >= int(rir_len))
    throw std::runtime_error(
        "the rir array is too small for the maximum time recorded");

  // create necessary data
  size_t lut_size = (fdl + 2) * lut_gran;
  auto lut_delta = T(1.0) / lut_gran;
  auto lut_gran_f = T(lut_gran);

  // sinc values table
  std::vector<T> sinc_lut(lut_size);
  T n = -T(fdl2) - 1;
  for (size_t idx = 0; idx < lut_size; n += lut_delta, idx++) {
    if (n == 0.0)
      sinc_lut[idx] = 1.0;
    else
      sinc_lut[idx] = std::sin(pi * n) / (pi * n);
  }

  // hann window
  std::vector<T> hann(fdl);
  for (size_t idx = 0; idx < fdl; idx++)
    hann[idx] = T(0.5) - T(0.5) * std::cos((pi * T(2 * idx)) / T(fdl - 1));

  // divide into equal size blocks for thread processing
  std::vector<T> rir_out(num_threads * rir_len);
  size_t block_size = size_t(std::ceil(double(n_times) / double(num_threads)));

  // relase the GIL from here on
  py::gil_scoped_release release;

  // build the RIR
  ThreadPool pool(num_threads);
  std::vector<std::future<void>> results;
  for (size_t t_idx = 0; t_idx < num_threads; t_idx++) {
    size_t t_start = t_idx * block_size;
    size_t t_end = std::min(t_start + block_size, n_times);
    size_t offset = t_idx * rir_len;

    results.emplace_back(pool.enqueue(
        [&](size_t t_start, size_t t_end, size_t offset) {
          for (size_t idx = t_start; idx < t_end; idx++) {
            // decompose integral/fractional parts
            T sample_frac = fs * tim_acc(idx);
            T time_ip = std::floor(sample_frac);
            T time_fp = sample_frac - time_ip;

            // LUT interpolation
            T x_off_frac = (1. - time_fp) * lut_gran_f;
            T lut_gran_off = std::floor(x_off_frac);
            T x_off = (x_off_frac - lut_gran_off);

            int lut_pos = int(lut_gran_off);
            int f = int(time_ip) - fdl2;
            for (size_t k = 0; k < fdl; lut_pos += lut_gran, f++, k++)
              rir_out[offset + f] +=
                  alp_acc(idx) * hann[k] *
                  (sinc_lut[lut_pos] +
                   x_off * (sinc_lut[lut_pos + 1] - sinc_lut[lut_pos]));
          }
        },
        t_start, t_end, offset));
  }

  for (auto &&result : results) result.get();

  block_size = size_t(std::ceil(double(rir_len) / double(num_threads)));

  std::vector<std::future<void>> sum_results;
  for (size_t t_idx = 0; t_idx < num_threads; t_idx++) {
    size_t t_start = t_idx * block_size;
    size_t t_end = std::min(t_start + block_size, rir_len);

    sum_results.emplace_back(pool.enqueue(
        [&](size_t t_start, size_t t_end) {
          for (size_t idx = t_start; idx < t_end; idx++)
            for (size_t t_idx = 0; t_idx < num_threads; t_idx++)
              rir_acc(idx) += rir_out[t_idx * rir_len + idx];
        },
        t_start, t_end));
  }

  for (auto &&result : sum_results) result.get();
}

void rir_builder(py::buffer rir, const py::buffer &time, const py::buffer &alpha,
                 int fs, size_t fdl, size_t lut_gran, size_t num_threads) {
  // dispatch to correct implementation depending on input type
  auto buf = pybind11::array::ensure(rir);
  if (py::isinstance<py::array_t<float>>(buf)) {
    threaded_rir_builder_impl<float>(rir, time, alpha, fs, fdl,
                                     lut_gran, num_threads);
  } else if (py::isinstance<py::array_t<double>>(buf)) {
    threaded_rir_builder_impl<double>(rir, time, alpha, fs, fdl,
                                      lut_gran, num_threads);
  } else {
    std::runtime_error("wrong type array for rir builder");
  }
}

template <class T>
void threaded_delay_sum_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast> irs,
    const py::array_t<int, py::array::c_style | py::array::forcecast> delays,
    py::array_t<T, py::array::c_style | py::array::forcecast> output,
    size_t num_threads) {
  // accessors for the arrays
  auto irs_acc = irs.unchecked();                // short impulse responses
  auto delays_acc = delays.unchecked();          // integer delays
  auto output_acc = output.mutable_unchecked();  // output array

  size_t n_irs = irs.shape(0);
  size_t n_taps = irs.shape(1);
  size_t out_len = output.shape(0);

  // error handling
  if (n_irs != (size_t)delays.shape(0))
    throw std::runtime_error("time and alpha arrays should have the same size");

  // check that the output buffer is long enough
  py::buffer_info del_buf = delays.request();
  int *del_ptr = (int *)(del_buf.ptr);
  auto d_min = *std::min_element(del_ptr, del_ptr + del_buf.size);
  auto d_max = *std::max_element(del_ptr, del_ptr + del_buf.size);
  if (d_min < 0) throw std::runtime_error("delays should be positive");
  if (d_max + n_taps > out_len)
    throw std::runtime_error("the output buffer is too small");

  // divide into equal size blocks for thread processing
  std::vector<T> out_buffers(num_threads * out_len, 0);
  size_t block_size = size_t(std::ceil(double(n_irs) / double(num_threads)));

  // release the GIL from here on
  py::gil_scoped_release release;

  // build the RIR
  ThreadPool pool(num_threads);
  std::vector<std::future<void>> results;
  for (size_t t_idx = 0; t_idx < num_threads; t_idx++) {
    size_t t_start = t_idx * block_size;
    size_t t_end = std::min(t_start + block_size, (size_t)n_irs);
    size_t offset = t_idx * out_len;

    results.emplace_back(pool.enqueue(
        [&](size_t t_start, size_t t_end, size_t offset) {
          for (size_t idx = t_start; idx < t_end; idx++) {
            for (size_t k = 0; k < n_taps; k++) {
              out_buffers[offset + delays_acc(idx) + k] += irs_acc(idx, k);
            }
          }
        },
        t_start, t_end, offset));
  }

  for (auto &&result : results) result.get();

  // now sum all the parts
  block_size = size_t(std::ceil(double(out_len) / double(num_threads)));

  std::vector<std::future<void>> sum_results;
  for (size_t t_idx = 0; t_idx < num_threads; t_idx++) {
    size_t t_start = t_idx * block_size;
    size_t t_end = std::min(t_start + block_size, (size_t)out_len);

    sum_results.emplace_back(pool.enqueue(
        [&](size_t t_start, size_t t_end) {
          for (size_t idx = t_start; idx < t_end; idx++)
            for (size_t t_idx = 0; t_idx < num_threads; t_idx++)
              output_acc(idx) += out_buffers[t_idx * out_len + idx];
        },
        t_start, t_end));
  }

  for (auto &&result : sum_results) result.get();
}

void delay_sum(const py::buffer irs, const py::buffer delays, py::buffer output,
               size_t num_threads) {
  // dispatch to correct implementation depending on input type
  auto buf = pybind11::array::ensure(irs);
  if (py::isinstance<py::array_t<float>>(buf)) {
    threaded_delay_sum_impl<float>(irs, delays, output, num_threads);
  } else if (py::isinstance<py::array_t<double>>(buf)) {
    threaded_delay_sum_impl<double>(irs, delays, output, num_threads);
  } else {
    std::runtime_error("wrong type array for delay-sum");
  }
}

template <class T>
void threaded_fractional_delay_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> out,
    const py::array_t<T, py::array::c_style | py::array::forcecast> time,
    size_t lut_gran, size_t num_threads) {
  auto pi = get_pi<T>();

  // accessors for the arrays
  auto out_acc = out.mutable_unchecked();
  auto tim_acc = time.unchecked();

  // error handling
  if (out_acc.ndim() != 2)
    throw std::runtime_error("output array needs to be 2D");
  if (tim_acc.shape(0) != out_acc.shape(0))
    throw std::runtime_error(
        "time and output arrays should have the same size");

  size_t n_times = time.size();
  size_t fdl = out_acc.shape(1);
  if (fdl % 2 == 0)  // force to be odd length
    fdl = fdl - 1;
  size_t fdl2 = (fdl - 1) / 2;

  // create necessary data
  size_t lut_size = (fdl + 2) * lut_gran;
  auto lut_delta = T(1.0) / lut_gran;
  auto lut_gran_f = T(lut_gran);

  // sinc values table
  std::vector<T> sinc_lut(lut_size);
  T n = -T(fdl2) - 1;
  for (size_t idx = 0; idx < lut_size; n += lut_delta, idx++) {
    if (n == 0.0)
      sinc_lut[idx] = 1.0;
    else
      sinc_lut[idx] = std::sin(pi * n) / (pi * n);
  }

  // hann window
  std::vector<T> hann(fdl);
  for (size_t idx = 0; idx < fdl; idx++)
    hann[idx] = T(0.5) - T(0.5) * std::cos((T(2 * idx) * pi) / T(fdl - 1));

  // divide into equal size blocks for thread processing
  size_t block_size = size_t(std::ceil(double(n_times) / double(num_threads)));

  // relase the GIL from here on
  py::gil_scoped_release release;

  // build the RIR
  ThreadPool pool(num_threads);
  std::vector<std::future<void>> results;
  for (size_t t_idx = 0; t_idx < num_threads; t_idx++) {
    size_t t_start = t_idx * block_size;
    size_t t_end = std::min(t_start + block_size, n_times);

    results.emplace_back(pool.enqueue(
        [&](size_t t_start, size_t t_end) {
          for (size_t idx = t_start; idx < t_end; idx++) {
            // LUT interpolation
            T x_off_frac = (1. - tim_acc(idx)) * lut_gran_f;
            T lut_gran_off = std::floor(x_off_frac);
            T x_off = (x_off_frac - lut_gran_off);

            int lut_pos = int(lut_gran_off);
            for (size_t k = 0; k < fdl; lut_pos += lut_gran, k++)
              out_acc(idx, k) =
                  hann[k] *
                  (sinc_lut[lut_pos] +
                   x_off * (sinc_lut[lut_pos + 1] - sinc_lut[lut_pos]));
          }
        },
        t_start, t_end));
  }

  for (auto &&result : results) result.get();
}

void fractional_delay(py::buffer out, const py::buffer time, size_t lut_gran,
                      size_t num_threads) {
  // dispatch to correct implementation depending on input type
  auto buf = pybind11::array::ensure(out);
  if (py::isinstance<py::array_t<float>>(buf)) {
    threaded_fractional_delay_impl<float>(out, time, lut_gran, num_threads);
  } else if (py::isinstance<py::array_t<double>>(buf)) {
    threaded_fractional_delay_impl<double>(out, time, lut_gran, num_threads);
  } else {
    std::runtime_error("wrong type array for fractional delays");
  }
}

