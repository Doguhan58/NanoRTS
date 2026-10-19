#pragma once

#include <cstdint>

#include "state.cuh"

namespace nanorts::core {

constexpr unsigned kWarp = 0xffffffffu;

template <typename Slot, typename Cell>
__device__ bool lowest_free_slot(const LaneState<Slot, Cell>& state, const Extents& size, int32_t player, int32_t& newborn) {
    const int32_t block_start = size.neutral + player * size.capacity;
    const int lane = threadIdx.x % 32;

    for (int32_t base = 0; base < size.capacity; base += 32) {
        const int32_t local = base + lane;
        const bool free = local < size.capacity && !state.active(block_start + local);
        const unsigned free_lanes = __ballot_sync(kWarp, free);

        if (free_lanes != 0) {
            newborn = block_start + base + __ffs(free_lanes) - 1;
            return true;
        }
    }

    newborn = 0;
    return false;
}

}
