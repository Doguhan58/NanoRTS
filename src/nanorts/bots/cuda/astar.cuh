#pragma once

#include <cstdint>

namespace nanorts::astar {

constexpr int32_t kAbsent = -1;
constexpr int32_t kUp = 0;
constexpr int32_t kRight = 1;
constexpr int32_t kDown = 2;
constexpr int32_t kLeft = 3;

constexpr uint8_t kUnseen = 0;
constexpr uint8_t kBlocked = 1;
constexpr uint8_t kSeen = 2;

constexpr int kScratchBytesPerCell = 3 * sizeof(int16_t) + sizeof(uint8_t);

struct Path {
    int32_t found;
    int32_t step;
    int32_t distance;
    int32_t pops;
};

struct Scratch {
    int16_t* top;
    int16_t* below;
    int16_t* parent;
    uint8_t* seen;

    __device__ static Scratch carve(void* shared, int cells) {
        int16_t* halves = static_cast<int16_t*>(shared);
        return Scratch{halves, halves + cells, halves + 2 * cells, reinterpret_cast<uint8_t*>(halves + 3 * cells)};
    }

    __device__ void clear(int cell, bool free) {
        top[cell] = -1;
        seen[cell] = free ? kUnseen : kBlocked;
    }

    __device__ void push(int stack, int cell, int from) {
        seen[cell] = kSeen;
        parent[cell] = static_cast<int16_t>(from);
        below[cell] = top[stack];
        top[stack] = static_cast<int16_t>(cell);
    }
};

__device__ inline int floor_mod(int value, int width) {
    const int remainder = value % width;
    return remainder < 0 ? remainder + width : remainder;
}

__device__ inline int floor_div(int value, int width) {
    return (value - floor_mod(value, width)) / width;
}

__device__ inline int manhattan(int x, int y, int target_x, int target_y) {
    return abs(x - target_x) + abs(y - target_y);
}

__device__ inline Path search(Scratch& s, int start, int target, int range, int width, int height) {
    const int cells = width * height;

    if (start < 0 || start >= cells) {
        return Path{0, kAbsent, kAbsent, 0};
    }

    const int target_x = floor_mod(target, width);
    const int target_y = floor_div(target, width);
    const int squared_range = static_cast<int>(static_cast<uint32_t>(range) * static_cast<uint32_t>(range));
    s.push(0, start, start);
    int lowest = 0;
    int open = 1;
    int pops = 0;

    while (open > 0) {
        ++pops;

        while (s.top[lowest] == -1) {
            ++lowest;
        }

        const int pos = s.top[lowest];
        s.top[lowest] = s.below[pos];
        --open;
        const int y = pos / width;
        const int x = pos - y * width;
        const int dx = x - target_x;
        const int dy = y - target_y;

        if (dx * dx + dy * dy <= squared_range) {
            int at = pos;
            int above = s.parent[pos];
            int last = at;
            int distance = 0;

            while (above != at) {
                last = at;
                at = above;
                above = s.parent[at];
                ++distance;
            }

            int32_t step = kAbsent;

            if (last == at + width) {
                step = kDown;
            } else if (last == at - 1) {
                step = kLeft;
            } else if (last == at - width) {
                step = kUp;
            } else if (last == at + 1) {
                step = kRight;
            }

            return Path{1, step, distance, pops};
        }

        const int h = manhattan(x, y, target_x, target_y);
        const bool up = y > 0 && s.seen[pos - width] == kUnseen;
        const bool right = x < width - 1 && s.seen[pos + 1] == kUnseen;
        const bool down = y < height - 1 && s.seen[pos + width] == kUnseen;
        const bool left = x > 0 && s.seen[pos - 1] == kUnseen;

        if (up) {
            s.push(lowest + (1 + manhattan(x, y - 1, target_x, target_y) - h) / 2, pos - width, pos);
            ++open;
        }

        if (right) {
            s.push(lowest + (1 + manhattan(x + 1, y, target_x, target_y) - h) / 2, pos + 1, pos);
            ++open;
        }

        if (down) {
            s.push(lowest + (1 + manhattan(x, y + 1, target_x, target_y) - h) / 2, pos + width, pos);
            ++open;
        }

        if (left) {
            s.push(lowest + (1 + manhattan(x - 1, y, target_x, target_y) - h) / 2, pos - 1, pos);
            ++open;
        }
    }

    return Path{0, kAbsent, kAbsent, pops};
}

}
