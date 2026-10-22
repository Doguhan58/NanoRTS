#pragma once

#include <cstdint>

#include "ranking.cuh"
#include "state.cuh"
#include "units.cuh"

namespace nanorts::core {

template <typename Slot, typename Cell>
__device__ int32_t order_holders(const LaneState<Slot, Cell>& state, const Extents& size, Slot* held, uint64_t* listed) {
    const int lane = threadIdx.x % 32;
    int32_t holders = 0;

    for (int32_t base = 0; base < size.owned; base += 32) {
        const int32_t local = base + lane;
        const int32_t order = local < size.owned ? state.pending_order[size.neutral + local] : kAbsent;
        const bool holds = order != kAbsent;
        const unsigned holding_lanes = __ballot_sync(kWarp, holds);

        if (holds) {
            listed[holders + __popc(holding_lanes & lanes_below())] = ranked_entry(0, order, local);
        }

        holders += __popc(holding_lanes);
    }

    __syncwarp();

    for (int32_t position = lane; position < holders; position += 32) {
        const uint64_t entry = listed[position];
        int32_t place = 0;

        for (int32_t other = 0; other < holders; ++other) {
            place += listed[other] < entry;
        }

        held[place] = static_cast<Slot>(size.neutral + ranked_local(entry));
    }

    __syncwarp();
    return holders;
}

template <typename Slot, typename Cell>
__device__ int32_t batch_order(const LaneState<Slot, Cell>& state, const Slot* held, int32_t holders, Slot* members) {
    const int lane = threadIdx.x % 32;
    const int32_t time = *state.time;
    int32_t count = 0;

    for (int32_t base = 0; base < holders; base += 32) {
        const int32_t position = base + lane;
        bool acting = false;
        int32_t slot = 0;

        if (position < holders) {
            slot = held[position];
            const int32_t kind = state.pending_kind[slot];
            acting = kind != kKindAbsent && kind != kKindNone && state.pending_completes_at[slot] <= time;
        }

        const unsigned acting_lanes = __ballot_sync(kWarp, acting);

        if (acting) {
            members[count + __popc(acting_lanes & ((1u << lane) - 1u))] = static_cast<Slot>(slot);
        }

        count += __popc(acting_lanes);
    }

    __syncwarp();
    return count;
}

}
