#pragma once

#include <cstdint>

#include "board.cuh"

namespace nanorts::core {

constexpr int32_t kKindAbsent = -1;
constexpr int32_t kKindNone = 0;
constexpr int32_t kKindMove = 1;
constexpr int32_t kKindHarvest = 2;
constexpr int32_t kKindReturn = 3;
constexpr int32_t kKindProduce = 4;
constexpr int32_t kKindAttack = 5;

constexpr int kPlayers = 2;

struct Extents {
    int32_t width;
    int32_t height;
    int32_t cells;
    int32_t entities;
    int32_t neutral;
    int32_t capacity;
    int32_t owned;
    int32_t types;
};

template <typename Slot, typename Cell>
struct LaneState {
    int32_t* birth;
    int8_t* unit_type;
    int8_t* owner;
    Cell* cell;
    int16_t* hit_points;
    int32_t* carried_resources;
    int8_t* pending_kind;
    int8_t* pending_direction;
    int32_t* pending_none_duration;
    Cell* pending_target_cell;
    int8_t* pending_produced_type;
    int32_t* pending_issued_at;
    int32_t* pending_completes_at;
    int32_t* pending_order;
    Slot* entity_at;
    const uint8_t* terrain;
    int32_t* player_resources;
    int32_t* time;
    int32_t* next_birth;
    int32_t* next_pending_order;
    int32_t* alternating_conflict_count;
    int32_t* spawn_overflow;

    __device__ bool active(int32_t slot) const {
        return birth[slot] != kAbsent;
    }

    __device__ Board<Slot> board(const Extents& size) const {
        return Board<Slot>{size.width, size.height, size.entities, size.types, entity_at, terrain, unit_type, owner};
    }

    __device__ void remove(int32_t slot) const {
        birth[slot] = kAbsent;
        unit_type[slot] = static_cast<int8_t>(kAbsent);
        owner[slot] = static_cast<int8_t>(kAbsent);
        cell[slot] = static_cast<Cell>(kAbsent);
        hit_points[slot] = 0;
        carried_resources[slot] = 0;
        clear_pending(slot);
    }

    __device__ void clear_pending(int32_t slot) const {
        pending_kind[slot] = static_cast<int8_t>(kKindAbsent);
        pending_direction[slot] = static_cast<int8_t>(kAbsent);
        pending_none_duration[slot] = 0;
        pending_target_cell[slot] = static_cast<Cell>(kAbsent);
        pending_produced_type[slot] = static_cast<int8_t>(kAbsent);
        pending_issued_at[slot] = 0;
        pending_completes_at[slot] = 0;
        pending_order[slot] = kAbsent;
    }
};

__device__ inline int32_t wrapping_add(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
}

__device__ inline int32_t wrapping_sub(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) - static_cast<uint32_t>(b));
}

__device__ inline int32_t wrapping_mul(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) * static_cast<uint32_t>(b));
}

}
