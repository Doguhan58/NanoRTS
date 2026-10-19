#pragma once

#include "action_kind.cuh"

namespace nanorts::core {

__device__ inline int32_t return_duration(const Content& content, int actor_type) {
    return content.timing(actor_type, kUnitMoveTime);
}

__device__ inline bool return_is_valid(const UnitTable& table, const Actor& actor, const Neighbour& adjacent) {
    return table.flag(actor.unit_type, kCanHarvest) && actor.carried > 0 && adjacent.occupied && table.flag(adjacent.unit_type, kIsStockpile) && adjacent.owner == actor.owner;
}

template <typename Slot, typename Cell>
__device__ Removal return_complete(const Member<Slot, Cell>& member, Row& row) {
    const LaneState<Slot, Cell>& state = member.state;
    const int32_t entities = member.size.entities;
    const Neighbour& stockpile = member.neighbour;
    const int32_t carried = member.retained.carried;
    const bool returning = member.performs(kKindReturn) && stockpile.occupied && member.content.table.flag(stockpile.unit_type, kIsStockpile) && carried > 0;
    const int32_t credit = returning ? carried : 0;
    const int32_t balance = wrapping_add(state.player_resources[member.player], credit);
    const int32_t target_birth = state.birth[gather_index(stockpile.slot, entities)];
    member.write(state.carried_resources, member.slot, entities, returning && member.alive, 0);
    member.write(state.player_resources, member.player, kPlayers, returning, balance);

    if (returning) {
        member.record(row, kReturnRecord);
        row.target_slot = stockpile.slot;
        row.target_birth = target_birth;
        row.amount = credit;
        row.balance = balance;
    }

    return Removal{false, 0};
}

}
