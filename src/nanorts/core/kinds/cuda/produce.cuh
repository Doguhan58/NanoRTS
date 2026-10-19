#pragma once

#include "../../cuda/units.cuh"
#include "action_kind.cuh"

namespace nanorts::core {

__device__ inline int32_t produce_duration(const Content& content, int32_t produced_type) {
    return content.timing(content.known_type(produced_type), kUnitProduceTime);
}

__device__ inline int32_t produce_cost(const Content& content, int32_t produced_type) {
    return content.table.at(content.known_type(produced_type), kCost);
}

__device__ inline bool produce_type_is_valid(const UnitTable& table, const Actor& actor, int32_t balance, int produced_type) {
    return produced_type >= 0 && produced_type < table.types && table.produces(actor.unit_type, produced_type) && balance >= table.at(produced_type, kCost);
}

template <typename Slot, typename Cell>
__device__ Removal produce_complete(const Member<Slot, Cell>& member, Row& row) {
    const LaneState<Slot, Cell>& state = member.state;
    const Extents& size = member.size;
    const int player = member.player;
    const int newborn_type = member.content.known_type(member.retained.produced_type);
    const int32_t balance = wrapping_sub(state.player_resources[player], produce_cost(member.content, member.retained.produced_type));
    const bool affordable = member.performs(kKindProduce) && balance >= 0;
    int32_t newborn = 0;
    const bool has_free_slot = lowest_free_slot(state, size, player, newborn);
    const bool spawns = affordable && has_free_slot;
    const bool overflows = affordable && !has_free_slot;
    const int32_t birth = *state.next_birth;
    member.write(state.player_resources, player, kPlayers, spawns, balance);
    member.write(state.entity_at, member.destination, size.cells, spawns, newborn);
    member.write(state.birth, newborn, size.entities, spawns, birth);
    member.write(state.unit_type, newborn, size.entities, spawns, newborn_type);
    member.write(state.owner, newborn, size.entities, spawns, player);
    member.write(state.cell, newborn, size.entities, spawns, member.destination);
    member.write(state.hit_points, newborn, size.entities, spawns, member.content.timing(newborn_type, kUnitHitPoints));
    member.increment(state.next_birth, 0, 1, spawns);
    member.increment(state.spawn_overflow, player, kPlayers, overflows);

    if (spawns) {
        member.record(row, kSpawnRecord);
        row.produced_type = newborn_type;
        row.spawn_slot = newborn;
        row.spawn_birth = birth;
        row.spawn_cell = member.destination;
        row.balance = balance;
    }

    return Removal{false, 0};
}

}
