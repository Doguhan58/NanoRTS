#pragma once

#include "action_kind.cuh"

namespace nanorts::core {

__device__ inline int32_t harvest_duration(const Content& content, int actor_type) {
    return content.timing(actor_type, kUnitHarvestTime);
}

__device__ inline bool harvest_is_valid(const UnitTable& table, const Actor& actor, const Neighbour& adjacent) {
    return table.flag(actor.unit_type, kCanHarvest) && actor.carried == 0 && adjacent.occupied && table.flag(adjacent.unit_type, kIsResource);
}

template <typename Slot, typename Cell>
__device__ Removal harvest_complete(const Member<Slot, Cell>& member, Row& row) {
    const LaneState<Slot, Cell>& state = member.state;
    const int32_t entities = member.size.entities;
    const Neighbour& stack = member.neighbour;
    const bool harvesting = member.performs(kKindHarvest) && stack.occupied && member.content.table.flag(stack.unit_type, kIsResource);
    const int32_t amount = member.content.timing(member.actor_type, kUnitHarvestAmount);
    const int32_t stack_slot = gather_index(stack.slot, entities);
    const int32_t left_in_stack = wrapping_sub(state.carried_resources[stack_slot], amount);
    const bool depleted = harvesting && left_in_stack <= 0;
    const int32_t target_birth = state.birth[stack_slot];
    member.write(state.carried_resources, stack.slot, entities, harvesting, left_in_stack);
    member.write(state.carried_resources, member.slot, entities, harvesting && member.alive, amount);

    if (harvesting) {
        member.record(row, depleted ? kHarvestRecord | kHarvestDepleted : kHarvestRecord);
        row.target_slot = stack.slot;
        row.target_birth = target_birth;
        row.amount = amount;
    }

    return Removal{depleted, stack.slot};
}

}
