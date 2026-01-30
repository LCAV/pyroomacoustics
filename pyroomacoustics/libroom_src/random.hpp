#ifndef __RANDOM_HPP__
#define __RANDOM_HPP__

#include <random>

namespace rng {
    // We use a static engine so it persists across calls
    inline std::mt19937& get_engine() {
        static std::mt19937 engine;
        return engine;
    }

    inline void set_seed(unsigned int seed) {
        get_engine().seed(seed);
    }
}

#endif // __RANDOM_HPP__
