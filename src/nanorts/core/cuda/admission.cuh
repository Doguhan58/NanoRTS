#pragma once

#include <cstdint>

#include "../kinds/cuda/action_kind.cuh"
#include "../kinds/cuda/attack.cuh"
#include "../kinds/cuda/harvest.cuh"
#include "../kinds/cuda/move.cuh"
#include "../kinds/cuda/none.cuh"
#include "../kinds/cuda/produce.cuh"
#include "../kinds/cuda/return_.cuh"
#include "ranking.cuh"
#include "reservations.cuh"
#include "state.cuh"

namespace nanorts::core {

struct Command {
    int32_t kind;
    int32_t direction;
    int32_t none_duration;
    int32_t target_cell;
    int32_t produced_type;
};

struct Admitted {
    int32_t* harvest;
    int32_t* return_;
    int32_t* produce;
    int32_t* attack_by_target_owner;
};

constexpr int kTargetOwners = 3;

struct CommandTable {
    const int32_t* kind;
    const int32_t* direction;
    const int32_t* none_duration;
    const int32_t* target_cell;
    const int32_t* produced_type;
    const int32_t* request_order;

    __device__ Command at(int32_t local) const {
        return Command{kind[local], direction[local], none_duration[local], target_cell[local], produced_type[local]};
    }
};

template <typename Slot, typename Cell>
__device__ int32_t order_requests(const LaneState<Slot, Cell>& state, const Extents& size, const CommandTable& commands, Slot* requests, uint64_t* listed) {
    const int lane = threadIdx.x % 32;
    const int rank_bits = max(1, 32 - __clz(size.capacity - 1));
    const int32_t highest_rank = (1 << rank_bits) - 1;
    auto rank = [&](int32_t local) {
        const int32_t requested = commands.request_order[local];
        return clip(requested == kAbsent ? 0 : requested, 0, highest_rank);
    };

    auto issued = [&](int32_t local) { return state.active(size.neutral + local) && commands.kind[local] != kKindAbsent; };
    int32_t ordered = 0;

    for (int player = 0; player < kPlayers; ++player) {
        const int32_t start = player * size.capacity;
        int32_t player_requests = 0;

        for (int32_t base = 0; base < size.capacity; base += 32) {
            const int32_t local = start + base + lane;
            const bool issues = base + lane < size.capacity && issued(local);
            const unsigned issuing_lanes = __ballot_sync(kWarp, issues);

            if (issues) {
                listed[player_requests + __popc(issuing_lanes & lanes_below())] = ranked_entry(rank(local), state.birth[size.neutral + local], local);
            }

            player_requests += __popc(issuing_lanes);
        }

        __syncwarp();

        for (int32_t position = lane; position < player_requests; position += 32) {
            const uint64_t entry = listed[position];
            int32_t place = 0;

            for (int32_t other = 0; other < player_requests; ++other) {
                place += listed[other] < entry;
            }

            requests[ordered + place] = static_cast<Slot>(size.neutral + ranked_local(entry));
        }

        __syncwarp();
        ordered += player_requests;
    }

    return ordered;
}

__device__ inline bool claims_position(int32_t kind) {
    return kind == kKindMove || kind == kKindProduce;
}

__device__ inline Reservation reservation(const Content& content, int32_t width, int32_t cell, int32_t owner, int32_t kind, int32_t direction, int32_t produced_type, bool has_action) {
    return Reservation{
        has_action && claims_position(kind),
        reserved_position(cell, direction, width),
        has_action && kind == kKindProduce ? produce_cost(content, produced_type) : 0,
        clip(owner, 0, kPlayers - 1),
    };
}

__device__ inline int32_t action_duration(const Content& content, int32_t unit_type, int32_t kind, int32_t requested_none_duration, int32_t produced_type) {
    const int actor_type = clip(unit_type, 0, content.table.types - 1);

    switch (kind) {
        case kKindNone:
            return none_duration(requested_none_duration);
        case kKindMove:
            return move_duration(content, actor_type);
        case kKindHarvest:
            return harvest_duration(content, actor_type);
        case kKindReturn:
            return return_duration(content, actor_type);
        case kKindProduce:
            return produce_duration(content, produced_type);
        case kKindAttack:
            return attack_duration(content, actor_type);
        default:
            return 0;
    }
}

template <typename Slot>
__device__ bool can_execute_action(const Content& content, const Board<Slot>& board, const Actor& actor, int32_t balance, int32_t cell, const Command& command) {
    const Neighbour adjacent = directed_neighbour(board, cell, command.direction);

    switch (command.kind) {
        case kKindNone:
            return none_is_valid();
        case kKindMove:
            return move_is_valid(content.table, actor, adjacent);
        case kKindHarvest:
            return harvest_is_valid(content.table, actor, adjacent);
        case kKindReturn:
            return return_is_valid(content.table, actor, adjacent);
        case kKindProduce:
            return produce_type_is_valid(content.table, actor, balance, command.produced_type) && adjacent.free;
        case kKindAttack:
            return attack_is_valid(content.table, board, actor, cell, command.target_cell);
        default:
            return false;
    }
}

template <typename Slot, typename Cell>
__device__ bool position_is_blocked(const LaneState<Slot, Cell>& state, const Extents& size, int32_t position) {
    const int32_t cell = gather_index(position, size.cells);
    return state.terrain[cell] != 0 || static_cast<int32_t>(state.entity_at[cell]) != kAbsent;
}

template <typename Slot, typename Cell>
__device__ void count_admitted(const LaneState<Slot, Cell>& state, const Extents& size, const Admitted& admitted, int32_t player, int32_t kind, const Command& command, bool issued) {
    if (!issued) {
        return;
    }

    if (kind == kKindHarvest) {
        admitted.harvest[player] += 1;
    } else if (kind == kKindReturn) {
        admitted.return_[player] += 1;
    } else if (kind == kKindProduce) {
        admitted.produce[player * size.types + clip(command.produced_type, 0, size.types - 1)] += 1;
    } else if (kind == kKindAttack) {
        const int32_t occupant = static_cast<int32_t>(state.entity_at[clip(command.target_cell, 0, size.cells - 1)]);
        const int32_t target_owner = occupant == kAbsent ? kAbsent : static_cast<int32_t>(state.owner[clip(occupant, 0, size.entities - 1)]);
        const int32_t target_owner_index = target_owner + 1;

        if (target_owner_index >= 0 && target_owner_index < kTargetOwners) {
            admitted.attack_by_target_owner[player * kTargetOwners + target_owner_index] += 1;
        }
    }
}

template <typename Slot, typename Cell>
__device__ PendingClaim pending_claim(const LaneState<Slot, Cell>& state, const Extents& size, const Content& content, int32_t slot) {
    const int32_t held_kind = state.pending_kind[slot];
    return PendingClaim{
        reservation(content, size.width, static_cast<int32_t>(state.cell[slot]), state.owner[slot], held_kind, state.pending_direction[slot], state.pending_produced_type[slot], held_kind != kKindAbsent),
        action_duration(content, state.unit_type[slot], held_kind, state.pending_none_duration[slot], state.pending_produced_type[slot]),
    };
}

template <typename Slot, typename Cell>
__device__ void assign_pending_action(const LaneState<Slot, Cell>& state, int32_t slot, const Command& command, const Resolved& resolved, int32_t time, int32_t duration, int32_t order) {
    const int32_t kind = resolved.kind;
    const bool keeps_direction = kind == kKindMove || kind == kKindHarvest || kind == kKindReturn || kind == kKindProduce;
    state.pending_kind[slot] = static_cast<int8_t>(kind);
    state.pending_direction[slot] = static_cast<int8_t>(keeps_direction ? command.direction : kAbsent);
    state.pending_none_duration[slot] = kind == kKindNone ? resolved.none_duration : 0;
    state.pending_target_cell[slot] = static_cast<Cell>(kind == kKindAttack ? command.target_cell : kAbsent);
    state.pending_produced_type[slot] = static_cast<int8_t>(kind == kKindProduce ? command.produced_type : kAbsent);
    state.pending_issued_at[slot] = time;
    state.pending_completes_at[slot] = wrapping_add(time, duration);
    state.pending_order[slot] = order;
}

template <typename Slot, typename Cell>
__device__ void admit_one(const LaneState<Slot, Cell>& state, const Extents& size, const Content& content, const Slot* entries, int32_t count, Strategy strategy, draws::Key root, const Admitted& admitted, int32_t slot, const Command& command) {
    const bool leader = threadIdx.x % 32 == 0;
    const int32_t player = (slot - size.neutral) / size.capacity;
    const int32_t time = *state.time;
    const bool issued = state.active(slot) && command.kind != kKindAbsent;
    const int32_t unit_type = state.unit_type[slot];
    const int32_t cell = static_cast<int32_t>(state.cell[slot]);
    const int32_t owner = state.owner[slot];
    const Actor actor{clip(unit_type, 0, size.types - 1), state.owner[slot], state.carried_resources[slot]};
    const int32_t balance = state.player_resources[clip(owner, 0, kPlayers - 1)];

    const int32_t requested_duration = action_duration(content, unit_type, command.kind, command.none_duration, command.produced_type);
    const bool legal = can_execute_action(content, state.board(size), actor, balance, cell, command);
    int32_t kind = legal ? command.kind : kKindNone;
    int32_t none_duration = legal ? command.none_duration : requested_duration;
    const int32_t reserved = reserved_position(cell, command.direction, size.width);

    if (claims_position(kind) && position_is_blocked(state, size, reserved)) {
        kind = kKindNone;
        none_duration = requested_duration;
    }

    if (leader) {
        count_admitted(state, size, admitted, player, kind, command, issued);
    }

    const RequestedAction request{
        kind, none_duration, requested_duration, reservation(content, size.width, cell, owner, kind, command.direction, command.produced_type, issued), state.birth[slot],
    };

    const Resolved resolved = resolve(state, size, entries, count, strategy, root, request, [&](int32_t entry) { return pending_claim(state, size, content, entry); });

    const int32_t duration = resolved.kind == kKindNone ? resolved.none_duration : requested_duration;
    const int32_t held_order = state.pending_order[slot];
    const bool appends = issued && held_order == kAbsent;
    const int32_t order = held_order == kAbsent ? *state.next_pending_order : held_order;

    if (leader && issued && is_in_range(slot, size.entities)) {
        assign_pending_action(state, slot, command, resolved, time, duration, order);
    }

    if (leader && appends) {
        *state.next_pending_order = wrapping_add(*state.next_pending_order, 1);
    }

    __syncwarp();
}

}
