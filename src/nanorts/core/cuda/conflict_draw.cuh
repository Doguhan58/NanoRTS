#pragma once

#include <cstdint>

namespace nanorts::draws {

constexpr int32_t kConflictEvent = 0;

constexpr uint32_t kParity = 0x1BD11BDAu;

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

__host__ __device__ inline Key fold_in(Key key, int32_t data) {
    uint32_t x0 = 0u;
    uint32_t x1 = static_cast<uint32_t>(data);
    threefry2x32(key, x0, x1);
    return Key{x0, x1};
}

__host__ __device__ inline int32_t randint01(Key key) {
    uint32_t s0 = 0u;
    uint32_t s1 = 1u;
    threefry2x32(key, s0, s1);
    uint32_t b0 = 0u;
    uint32_t b1 = 0u;
    threefry2x32(Key{s0, s1}, b0, b1);
    return static_cast<int32_t>((b0 ^ b1) % 2u);
}

__host__ __device__ inline Key conflict_address(Key root, int32_t time, int32_t birth) {
    return fold_in(fold_in(fold_in(root, time), birth), kConflictEvent);
}

__host__ __device__ inline int32_t conflict_draw(Key address, int32_t ordinal) {
    return randint01(fold_in(address, ordinal));
}

}
