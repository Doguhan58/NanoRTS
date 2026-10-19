#pragma once

#include "action_kind.cuh"

namespace nanorts::core {

__device__ inline int32_t attack_duration(const Content& content, int actor_type) {
    return content.timing(actor_type, kUnitAttackTime);
}

__device__ inline bool can_attack(const UnitTable& table, const Actor& actor, int8_t target_owner, bool in_range) {
    return table.flag(actor.unit_type, kCanAttack) && target_owner >= 0 && target_owner != actor.owner && in_range;
}

template <typename Slot>
__device__ bool attack_is_valid(const UnitTable& table, const Board<Slot>& board, const Actor& actor, int32_t cell, int32_t target_cell) {
    const int32_t cells = board.width * board.height;
    const bool on_board = target_cell >= 0 && target_cell < cells;
    const int32_t target_slot = static_cast<int32_t>(board.entity_at[on_board ? target_cell : 0]);
    const bool holds_unit = on_board && target_slot != kAbsent;
    const int8_t target_owner = holds_unit ? board.owner[gather_index(target_slot, board.entity_capacity)] : kOwnerNeutral;
    const int32_t attack_range = table.at(actor.unit_type, kAttackRange);
    const int32_t dx = wrapping_sub(floor_mod(target_cell, board.width), floor_mod(cell, board.width));
    const int32_t dy = wrapping_sub(floor_div(target_cell, board.width), floor_div(cell, board.width));
    const int32_t squared_distance = wrapping_add(wrapping_mul(dx, dx), wrapping_mul(dy, dy));
    return can_attack(table, actor, target_owner, squared_distance <= wrapping_mul(attack_range, attack_range));
}

template <typename Slot, typename Cell>
__device__ Removal attack_complete(const Member<Slot, Cell>& member, Row& row) {
    const LaneState<Slot, Cell>& state = member.state;
    const Extents& size = member.size;
    const int32_t occupant = static_cast<int32_t>(state.entity_at[clip(member.retained.target_cell, 0, size.cells - 1)]);
    const bool hit = member.performs(kKindAttack) && occupant != kAbsent;
    const int32_t victim = hit ? occupant : 0;
    const int32_t damage = member.content.timing(member.actor_type, kUnitAttackDamage);
    const int32_t victim_slot = gather_index(victim, size.entities);
    const int32_t hit_points_left = wrapping_sub(static_cast<int32_t>(state.hit_points[victim_slot]), damage);
    const int32_t target_birth = state.birth[victim_slot];
    member.write(state.hit_points, victim, size.entities, hit, hit_points_left);

    if (hit) {
        member.record(row, kHitRecord);
        row.target_slot = victim;
        row.target_birth = target_birth;
        row.amount = damage;
    }

    return Removal{hit && hit_points_left <= 0, victim};
}

}
