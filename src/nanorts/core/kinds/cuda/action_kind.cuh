#pragma once

#include <cstdint>

#include "../../cuda/board.cuh"
#include "../../cuda/events.cuh"
#include "../../cuda/state.cuh"

namespace nanorts::core {

enum Column : int {
    kCanMove,
    kCanHarvest,
    kIsResource,
    kIsStockpile,
    kCanAttack,
    kAttackRange,
    kCost,
    kProduces,
};

struct UnitTable {
    const int32_t* rows;
    int types;

    __device__ int32_t at(int type, int column) const {
        return rows[type * (kProduces + types) + column];
    }

    __device__ bool flag(int type, int column) const {
        return at(type, column) != 0;
    }

    __device__ bool produces(int producer, int produced) const {
        return flag(producer, kProduces + produced);
    }
};

struct Actor {
    int unit_type;
    int8_t owner;
    int32_t carried;
};

enum Timing : int {
    kUnitMoveTime,
    kUnitHarvestTime,
    kUnitAttackTime,
    kUnitProduceTime,
    kUnitHitPoints,
    kUnitAttackDamage,
    kUnitHarvestAmount,
    kTimings,
};

struct Content {
    UnitTable table;
    const int32_t* timings;

    __device__ int32_t timing(int type, int column) const {
        return timings[type * kTimings + column];
    }

    __device__ int known_type(int32_t produced_type) const {
        return clip(produced_type, 0, table.types - 1);
    }
};

struct RetainedActor {
    int32_t kind;
    int32_t direction;
    int32_t target_cell;
    int32_t produced_type;
    int32_t birth;
    int32_t owner;
    int32_t unit_type;
    int32_t cell;
    int32_t carried;
};

struct Removal {
    bool when;
    int32_t slot;
};

template <typename Slot, typename Cell>
struct Member {
    const LaneState<Slot, Cell>& state;
    const Extents& size;
    const Content& content;
    int32_t slot;
    RetainedActor retained;
    int actor_type;
    int player;
    bool alive;
    int32_t destination;
    Neighbour neighbour;
    bool leader;

    __device__ bool performs(int32_t code) const {
        return retained.kind == code;
    }

    template <typename T, typename V>
    __device__ void write(T* leaf, int32_t index, int32_t extent, bool when, V value) const {
        __syncwarp();

        if (leader && when && is_in_range(index, extent)) {
            leaf[index] = static_cast<T>(value);
        }
    }

    __device__ void increment(int32_t* counter, int32_t index, int32_t extent, bool when) const {
        __syncwarp();

        if (leader && when && is_in_range(index, extent)) {
            counter[index] = wrapping_add(counter[index], 1);
        }
    }

    __device__ void record(Row& row, int flag) const {
        row.flags |= flag;
        row.actor_birth = retained.birth;
        row.actor_owner = retained.owner;
        row.actor_unit_type = retained.unit_type;
    }
};

}
