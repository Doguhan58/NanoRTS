#pragma once

#include "action_kind.cuh"

namespace nanorts::core {

__device__ inline int32_t move_duration(const Content& content, int actor_type) {
    return content.timing(actor_type, kUnitMoveTime);
}

__device__ inline bool move_is_valid(const UnitTable& table, const Actor& actor, const Neighbour& adjacent) {
    return table.flag(actor.unit_type, kCanMove) && adjacent.free;
}

template <typename Slot, typename Cell>
__device__ Removal move_complete(const Member<Slot, Cell>& member) {
    const LaneState<Slot, Cell>& state = member.state;
    const int32_t cells = member.size.cells;
    const bool moving = member.performs(kKindMove) && member.alive;
    member.write(state.entity_at, member.destination, cells, moving, member.slot);
    member.write(state.entity_at, member.retained.cell, cells, moving, kAbsent);
    member.write(state.cell, member.slot, member.size.entities, moving, member.destination);
    return Removal{false, 0};
}

}
