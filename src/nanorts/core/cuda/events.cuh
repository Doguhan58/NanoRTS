#pragma once

#include <cstdint>

namespace nanorts::core {

enum RecordFlag : int {
    kHitRecord = 1 << 0,
    kDeathRecord = 1 << 1,
    kHarvestRecord = 1 << 2,
    kHarvestDepleted = 1 << 3,
    kReturnRecord = 1 << 4,
    kSpawnRecord = 1 << 5,
};

enum Int32Column : int {
    kActorBirth,
    kTargetBirth,
    kAmount,
    kBalance,
    kSpawnBirth,
    kDeathBirth,
    kInt32Columns,
};

enum IndexColumn : int {
    kTargetSlot,
    kSpawnSlot,
    kSpawnCell,
    kDeathSlot,
    kIndexColumns,
};

enum Int8Column : int {
    kFlags,
    kActorOwner,
    kActorUnitType,
    kProducedType,
    kDeathOwner,
    kDeathUnitType,
    kInt8Columns,
};

struct Row {
    int32_t flags;
    int32_t actor_birth;
    int32_t actor_owner;
    int32_t actor_unit_type;
    int32_t target_slot;
    int32_t target_birth;
    int32_t amount;
    int32_t balance;
    int32_t produced_type;
    int32_t spawn_slot;
    int32_t spawn_birth;
    int32_t spawn_cell;
    int32_t death_slot;
    int32_t death_birth;
    int32_t death_owner;
    int32_t death_unit_type;

    template <typename Index>
    __device__ void store(int32_t* wide, Index* index, int8_t* narrow) const {
        wide[kActorBirth] = actor_birth;
        wide[kTargetBirth] = target_birth;
        wide[kAmount] = amount;
        wide[kBalance] = balance;
        wide[kSpawnBirth] = spawn_birth;
        wide[kDeathBirth] = death_birth;
        index[kTargetSlot] = static_cast<Index>(target_slot);
        index[kSpawnSlot] = static_cast<Index>(spawn_slot);
        index[kSpawnCell] = static_cast<Index>(spawn_cell);
        index[kDeathSlot] = static_cast<Index>(death_slot);
        narrow[kFlags] = static_cast<int8_t>(flags);
        narrow[kActorOwner] = static_cast<int8_t>(actor_owner);
        narrow[kActorUnitType] = static_cast<int8_t>(actor_unit_type);
        narrow[kProducedType] = static_cast<int8_t>(produced_type);
        narrow[kDeathOwner] = static_cast<int8_t>(death_owner);
        narrow[kDeathUnitType] = static_cast<int8_t>(death_unit_type);
    }
};

}
