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
}

#endif // __RANDOM_HPP__
