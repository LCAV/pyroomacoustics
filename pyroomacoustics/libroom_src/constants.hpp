#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__

namespace constants {
    static constexpr float PI = 3.14159265358979323846f;
    static constexpr float TWO_PI = 2.0f * PI;
    static constexpr float HALF_PI = PI / 2.0f;
    // Initial energy of a particule.
    // The value 2.0 is necessary to match the scale of the ISM.
    static constexpr float ENERGY_0 = 2.0f;
}

#endif // __CONSTANTS_HPP__
