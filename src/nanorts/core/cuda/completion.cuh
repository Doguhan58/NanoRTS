#pragma once

#include <cstdint>

#include "../kinds/cuda/action_kind.cuh"
#include "../kinds/cuda/attack.cuh"
#include "../kinds/cuda/harvest.cuh"
#include "../kinds/cuda/move.cuh"
#include "../kinds/cuda/none.cuh"
#include "../kinds/cuda/produce.cuh"
#include "../kinds/cuda/return_.cuh"
#include "events.cuh"
#include "state.cuh"

namespace nanorts::core {

template <typename Slot, typename Cell>
__device__ RetainedActor retained_actor_at(const LaneState<Slot, Cell>& state, int32_t slot) {
    return RetainedActor{
        state.pending_kind[slot], state.pending_direction[slot], static_cast<int32_t>(state.pending_target_cell[slot]), state.pending_produced_type[slot], state.birth[slot], state.owner[slot], state.unit_type[slot], static_cast<int32_t>(state.cell[slot]), state.carried_resources[slot],
    };
}

template <typename Slot, typename Cell>
__device__ void remove_unit(const Member<Slot, Cell>& member, const Removal& removal, Row& row) {
    const LaneState<Slot, Cell>& state = member.state;
    const Extents& size = member.size;
    const int32_t removed = gather_index(removal.slot, size.entities);
    const int32_t cell = static_cast<int32_t>(state.cell[removed]);
    member.record(row, kDeathRecord);
    row.death_slot = removal.slot;
    row.death_birth = state.birth[removed];
    row.death_owner = state.owner[removed];
    row.death_unit_type = state.unit_type[removed];
    member.write(state.entity_at, cell, size.cells, true, kAbsent);

    if (member.leader && is_in_range(removal.slot, size.entities)) {
        state.remove(removal.slot);
    }
}

template <typename Slot, typename Cell>
__device__ void take_turn(const LaneState<Slot, Cell>& state, const Extents& size, const Content& content, int32_t slot, const RetainedActor& retained, Row& row) {
    const bool leader = threadIdx.x % 32 == 0;
    const Member<Slot, Cell> member{
        state, size, content, slot, retained, clip(retained.unit_type, 0, size.types - 1), clip(retained.owner, 0, kPlayers - 1), state.active(slot) && state.birth[slot] == retained.birth, reserved_position(retained.cell, retained.direction, size.width), directed_neighbour(state.board(size), retained.cell, retained.direction), leader,
    };

    row = Row{};
    Removal removal{false, 0};

    switch (retained.kind) {
        case kKindMove:
            removal = move_complete(member);
            break;
        case kKindHarvest:
            removal = harvest_complete(member, row);
            break;
        case kKindReturn:
            removal = return_complete(member, row);
            break;
        case kKindProduce:
            removal = produce_complete(member, row);
            break;
        case kKindAttack:
            removal = attack_complete(member, row);
            break;
        default:
            break;
    }

    if (removal.when) {
        remove_unit(member, removal, row);
    }

    __syncwarp();
}

}
