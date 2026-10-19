#pragma once

#include <cstdint>

#include "state.cuh"
#include "units.cuh"

namespace nanorts::core {

template <typename Slot, typename Cell>
__device__ void decide(const LaneState<Slot, Cell>& state, const Extents& size, uint8_t* game_over, int32_t* winner) {
    const int lane = threadIdx.x % 32;
    bool surviving[kPlayers];

    for (int player = 0; player < kPlayers; ++player) {
        const int32_t block_start = size.neutral + player * size.capacity;
        bool has_active_unit = false;

        for (int32_t base = 0; base < size.capacity; base += 32) {
            const int32_t local = base + lane;
            has_active_unit |= __any_sync(kWarp, local < size.capacity && state.active(block_start + local));
        }

        surviving[player] = has_active_unit;
    }

    const int survivors = surviving[0] + surviving[1];

    if (lane == 0) {
        *game_over = survivors <= 1;
        *winner = survivors == 1 ? (surviving[0] ? 0 : 1) : kAbsent;
    }

    __syncwarp();
}

}
