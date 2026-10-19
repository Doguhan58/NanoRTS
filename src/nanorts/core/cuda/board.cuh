#pragma once

#include <cstdint>

namespace nanorts::core {

constexpr int32_t kAbsent = -1;
constexpr int8_t kOwnerNeutral = -1;

constexpr int kDirections = 4;

__device__ constexpr int direction_dx(int direction) {
    return direction == 1 ? 1 : direction == 3 ? -1 : 0;
}

__device__ constexpr int direction_dy(int direction) {
    return direction == 0 ? -1 : direction == 2 ? 1 : 0;
}

__device__ inline int32_t floor_div(int32_t value, int32_t divisor) {
    const int32_t quotient = value / divisor;
    return value % divisor < 0 ? quotient - 1 : quotient;
}

__device__ inline int32_t floor_mod(int32_t value, int32_t divisor) {
    const int32_t remainder = value % divisor;
    return remainder < 0 ? remainder + divisor : remainder;
}

__device__ inline int32_t clip(int32_t value, int32_t low, int32_t high) {
    return value < low ? low : value > high ? high : value;
}

__device__ inline int32_t gather_index(int32_t index, int32_t size) {
    return clip(index < 0 ? index + size : index, 0, size - 1);
}

__device__ inline bool is_in_range(int32_t index, int32_t size) {
    return index >= 0 && index < size;
}

__device__ inline int32_t reserved_position(int32_t cell, int32_t direction, int32_t width) {
    const int known_direction = clip(direction, 0, kDirections - 1);
    return cell + direction_dx(known_direction) + direction_dy(known_direction) * width;
}

struct Neighbour {
    bool free;
    bool occupied;
    int unit_type;
    int8_t owner;
    int32_t slot;
};

template <typename Slot>
struct Board {
    int32_t width;
    int32_t height;
    int32_t entity_capacity;
    int32_t types;
    const Slot* entity_at;
    const uint8_t* terrain;
    const int8_t* unit_type;
    const int8_t* owner;
};

template <typename Slot>
__device__ Neighbour neighbour(const Board<Slot>& board, int32_t cell, int direction) {
    const int32_t neighbour_x = floor_mod(cell, board.width) + direction_dx(direction);
    const int32_t neighbour_y = floor_div(cell, board.width) + direction_dy(direction);

    if (neighbour_x < 0 || neighbour_x >= board.width || neighbour_y < 0 || neighbour_y >= board.height) {
        return Neighbour{false, false, 0, kOwnerNeutral, 0};
    }

    const int32_t neighbour_cell = neighbour_y * board.width + neighbour_x;
    const int32_t occupant = static_cast<int32_t>(board.entity_at[neighbour_cell]);
    const bool occupied = occupant != kAbsent;
    const int32_t slot = occupied ? gather_index(occupant, board.entity_capacity) : 0;
    return Neighbour{
        !occupied && board.terrain[neighbour_cell] == 0, occupied, clip(board.unit_type[slot], 0, board.types - 1), board.owner[slot], occupied ? occupant : 0,
    };
}

template <typename Slot>
__device__ Neighbour directed_neighbour(const Board<Slot>& board, int32_t cell, int32_t direction) {
    if (direction < 0 || direction >= kDirections) {
        return Neighbour{false, false, 0, kOwnerNeutral, 0};
    }

    return neighbour(board, cell, direction);
}

}
