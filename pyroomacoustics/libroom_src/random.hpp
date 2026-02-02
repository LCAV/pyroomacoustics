/*
 * Package-wide random number generator.
 * Copyright (C) 2026  Robin Scheibler
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

#ifndef __RANDOM_HPP__
#define __RANDOM_HPP__

#include <random>
#include <cstdint>

namespace rng {
    // Use the 64-bit version of Mersenne Twister
    inline std::mt19937_64& get_engine() {
        static std::mt19937_64 engine;
        return engine;
    }

    inline void set_seed(std::uint64_t seed) {
        get_engine().seed(seed);
    }

    // Helper for uniform real numbers in [min, max).
    inline float uniform(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(get_engine());
    }
}

#endif // __RANDOM_HPP__
