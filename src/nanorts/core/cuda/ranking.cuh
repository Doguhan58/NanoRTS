#pragma once

#include <cstdint>

namespace nanorts::core {

__device__ inline uint64_t ranked_entry(int32_t primary, int32_t secondary, int32_t local) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(primary)) << 48) | (static_cast<uint64_t>(static_cast<uint32_t>(secondary) ^ 0x80000000u) << 16) | static_cast<uint32_t>(local);
}

__device__ inline int32_t ranked_local(uint64_t entry) {
    return static_cast<int32_t>(entry & 0xffffu);
}

__device__ inline unsigned lanes_below() {
    return (1u << (threadIdx.x % 32)) - 1u;
}

}
