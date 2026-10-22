#pragma once

#include <cstdint>

#include "../kinds/cuda/action_kind.cuh"
#include "conflict_draw.cuh"
#include "state.cuh"
#include "units.cuh"

namespace nanorts::core {

enum Strategy : int {
    kCancelBoth = 0,
    kCancelRandom = 1,
    kCancelAlternating = 2,
};

constexpr int32_t kNoCap = 0x7fffffff;

constexpr int32_t kCrossCycleNoneDuration = -1;

struct Reservation {
    bool claims_position;
    int32_t position;
    int32_t cost;
    int32_t player;
};

__device__ inline bool reservations_conflict(const Reservation& existing, const Reservation& requested, const int32_t* balance) {
    const bool shared_position = existing.claims_position && requested.claims_position && existing.position == requested.position;
    const int32_t existing_cost_of_same_player = existing.player == requested.player ? existing.cost : 0;
    const int32_t claimed = wrapping_add(existing_cost_of_same_player, requested.cost);
    const bool over_budget = requested.cost != 0 && claimed > 0 && claimed > balance[requested.player];
    return shared_position || over_budget;
}

__device__ inline bool may_conflict(const Reservation& requested) {
    return requested.claims_position || requested.cost != 0;
}

struct PendingClaim {
    Reservation reservation;
    int32_t duration;
};

struct RequestedAction {
    int32_t kind;
    int32_t none_duration;
    int32_t duration;
    Reservation reservation;
    int32_t birth;
};

struct Resolved {
    int32_t kind;
    int32_t none_duration;
    int32_t draws_consumed;
};

__device__ inline int32_t warp_inclusive_min(int32_t value, int lane) {
#pragma unroll
    for (int distance = 1; distance < 32; distance *= 2) {
        const int32_t earlier = __shfl_up_sync(kWarp, value, distance);

        if (lane >= distance) {
            value = min(value, earlier);
        }
    }

    return value;
}

template <typename Slot, typename Cell>
__device__ void cancel_pending_action(const LaneState<Slot, Cell>& state, int32_t slot, int32_t settled_duration) {
    const int32_t cancelled_duration = wrapping_add(wrapping_mul(settled_duration, 2), 1) >> 1;
    state.pending_kind[slot] = static_cast<int8_t>(kKindNone);
    state.pending_direction[slot] = static_cast<int8_t>(kAbsent);
    state.pending_none_duration[slot] = cancelled_duration;
    state.pending_target_cell[slot] = static_cast<Cell>(kAbsent);
    state.pending_produced_type[slot] = static_cast<int8_t>(kAbsent);
    state.pending_completes_at[slot] = wrapping_add(state.pending_issued_at[slot], cancelled_duration);
}

template <typename Slot, typename Cell, typename PendingClaimOf>
__device__ Resolved resolve(const LaneState<Slot, Cell>& state, const Extents& size, const Slot* entries, int32_t count, Strategy strategy, draws::Key root, const RequestedAction& request, PendingClaimOf pending_claim_of) {
    if (!may_conflict(request.reservation)) {
        return Resolved{request.kind, request.none_duration, 0};
    }

    const int lane = threadIdx.x % 32;
    const unsigned lanes_below_mask = (1u << lane) - 1u;
    const int32_t time = *state.time;
    const int32_t alternating_count = *state.alternating_conflict_count;
    int32_t same_cycle_conflicts = 0;
    bool any_cross_cycle = false;
    int32_t request_duration_cap = kNoCap;

    for (int32_t base = 0; base < count; base += 32) {
        const int32_t position = base + lane;
        int32_t entry = 0;
        bool same_cycle_conflict = false;
        bool cross_cycle_conflict = false;
        int32_t pending_duration = 0;

        if (position < count) {
            entry = entries[position];
            const bool holds_action = state.pending_kind[entry] != kKindAbsent;
            const PendingClaim pending = pending_claim_of(entry);
            const bool conflicts = holds_action && reservations_conflict(pending.reservation, request.reservation, state.player_resources);
            const bool issued_earlier = state.pending_issued_at[entry] != time;
            const int32_t packed = wrapping_add(wrapping_mul(pending.duration, 4), (conflicts && !issued_earlier ? 2 : 0) + (conflicts && issued_earlier ? 1 : 0));
            cross_cycle_conflict = (packed & 1) != 0;
            same_cycle_conflict = ((packed >> 1) & 1) != 0;
            pending_duration = packed >> 2;
        }

        const unsigned same_cycle_lanes = __ballot_sync(kWarp, same_cycle_conflict);
        const unsigned cross_cycle_lanes = __ballot_sync(kWarp, cross_cycle_conflict);

        if (same_cycle_lanes == 0 && cross_cycle_lanes == 0) {
            continue;
        }

        const int32_t same_cycle_before = same_cycle_conflicts + __popc(same_cycle_lanes & lanes_below_mask);
        const bool cross_cycle_before = any_cross_cycle || (cross_cycle_lanes & lanes_below_mask) != 0;
        bool cancels_request = true;
        bool cancels_pending = true;

        if (strategy == kCancelAlternating) {
            cancels_request = floor_mod(wrapping_add(alternating_count, same_cycle_before), 2) == 0;
            cancels_pending = !cancels_request;
        } else if (strategy == kCancelRandom) {
            cancels_request = same_cycle_conflict && draws::conflict_draw(draws::conflict_address(root, time, request.birth), same_cycle_before) == 0;
            cancels_pending = !cancels_request;
        }

        const int32_t cap_through_entry = min(warp_inclusive_min(same_cycle_conflict && cancels_request ? pending_duration : kNoCap, lane), request_duration_cap);
        int32_t cap_before_entry = __shfl_up_sync(kWarp, cap_through_entry, 1);

        if (lane == 0) {
            cap_before_entry = request_duration_cap;
        }

        const int32_t request_duration_before = cross_cycle_before ? kCrossCycleNoneDuration : min(request.duration, cap_before_entry);

        if (same_cycle_conflict && cancels_pending) {
            cancel_pending_action(state, entry, min(pending_duration, request_duration_before));
        }

        same_cycle_conflicts += __popc(same_cycle_lanes);
        any_cross_cycle = any_cross_cycle || cross_cycle_lanes != 0;
        request_duration_cap = __shfl_sync(kWarp, cap_through_entry, 31);
    }

    __syncwarp();
    const bool request_cancelled = any_cross_cycle || request_duration_cap != kNoCap;
    const int32_t final_request_duration = any_cross_cycle ? kCrossCycleNoneDuration : min(request.duration, request_duration_cap);

    if (strategy == kCancelAlternating && lane == 0) {
        *state.alternating_conflict_count = wrapping_add(alternating_count, same_cycle_conflicts);
    }

    return Resolved{
        request_cancelled ? kKindNone : request.kind,
        request_cancelled ? final_request_duration : request.none_duration,
        strategy == kCancelRandom ? same_cycle_conflicts : 0,
    };
}

}
