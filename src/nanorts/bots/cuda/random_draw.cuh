#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace nanorts::random_draw {

constexpr uint32_t kParity = 0x1BD11BDAu;

constexpr int kMantissaShift = 9;
constexpr uint32_t kOneBits = 0x3F800000u;

constexpr float kTiny = FLT_MIN;
constexpr float kSpan = 1.0f - kTiny;
constexpr float kUnmarked = -HUGE_VALF;

struct Key {
    uint32_t k0;
    uint32_t k1;
};

__host__ __device__ inline uint32_t rotate_left(uint32_t x, uint32_t distance) {
    return (x << distance) | (x >> (32u - distance));
}

__host__ __device__ inline void threefry2x32(Key key, uint32_t& x0, uint32_t& x1) {
    constexpr uint32_t kRotations[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};
    const uint32_t schedule[3] = {key.k0, key.k1, key.k0 ^ key.k1 ^ kParity};
    x0 += schedule[0];
    x1 += schedule[1];

#pragma unroll
    for (int block = 0; block < 5; ++block) {
#pragma unroll
        for (int round = 0; round < 4; ++round) {
            x0 += x1;
            x1 = rotate_left(x1, kRotations[block % 2][round]);
            x1 ^= x0;
        }

        x0 += schedule[(block + 1) % 3];
        x1 += schedule[(block + 2) % 3] + static_cast<uint32_t>(block + 1);
    }
}

__host__ __device__ inline Key split(Key key, uint32_t index) {
    uint32_t x0 = 0u;
    uint32_t x1 = index;
    threefry2x32(key, x0, x1);
    return Key{x0, x1};
}

__host__ __device__ inline uint32_t bits(Key key, uint32_t index) {
    uint32_t x0 = 0u;
    uint32_t x1 = index;
    threefry2x32(key, x0, x1);
    return x0 ^ x1;
}

__device__ inline float uniform(Key key, uint32_t index) {
    const float floats = __fsub_rn(__uint_as_float((bits(key, index) >> kMantissaShift) | kOneBits), 1.0f);
    return fmaxf(kTiny, __fadd_rn(__fmul_rn(floats, kSpan), kTiny));
}

__device__ inline float gumbel_of(float u) {
    return -logf(-logf(u));
}

__device__ inline float score(Key key, uint32_t index, float logit) {
    return __fadd_rn(gumbel_of(uniform(key, index)), logit);
}

struct Best {
    float value = kUnmarked;
    int32_t index = 0;

    __device__ void offer(float candidate, int32_t at) {
        if (candidate > value) {
            value = candidate;
            index = at;
        }
    }
};

}
