#include <cstdint>

#include "astar.cuh"
#include "whole_act.cuh"

namespace {

namespace ffi = xla::ffi;
namespace astar = nanorts::astar;
using nanorts::whole_act::Extents;
using nanorts::whole_act::Lanes;
using nanorts::whole_act::StateRows;

constexpr int kThreads = nanorts::whole_act::kWarp;
constexpr unsigned kEveryThread = 0xffffffffu;
constexpr int32_t kMaxRows = kThreads;
constexpr int32_t kMaxWidth = 32;

constexpr int64_t kBytesPerCell = 8;
static_assert(kBytesPerCell == astar::kScratchBytesPerCell + 1);
constexpr int64_t kLaneBytes = 1024;
constexpr int64_t kBytesPerSlot = 74;
constexpr int64_t kBytesPerEntity = 37;

constexpr int32_t kAbsent = -1;
constexpr int32_t kHighest = 2147483647;
constexpr int32_t kLowest = -kHighest - 1;
constexpr int64_t kNoKey = 9223372036854775807LL;
constexpr int64_t kBeforeEveryKey = -kNoKey - 1;
constexpr int64_t kKeyRadix = 4294967296LL;
constexpr int32_t kWait = 1;
constexpr int kDirections = 4;

constexpr int32_t kKindAbsent = -1;
constexpr int32_t kKindNone = 0;
constexpr int32_t kKindMove = 1;
constexpr int32_t kKindHarvest = 2;
constexpr int32_t kKindReturn = 3;
constexpr int32_t kKindProduce = 4;
constexpr int32_t kKindAttack = 5;

constexpr int32_t kTypeResource = 0;
constexpr int32_t kTypeBase = 1;
constexpr int32_t kTypeBarracks = 2;
constexpr int32_t kTypeWorker = 3;
constexpr int32_t kTypeLight = 4;
constexpr int32_t kTypeHeavy = 5;
constexpr int32_t kTypeRanged = 6;

constexpr int32_t kUp = 0;
constexpr int32_t kRight = 1;
constexpr int32_t kDown = 2;
constexpr int32_t kLeft = 3;

constexpr int32_t kFarFromRanged = 3;
constexpr int32_t kManyCombatants = 24;
constexpr int32_t kTargets = 24;
constexpr int32_t kFewTargets = 12;
constexpr int32_t kFewUnits = 32;
constexpr int32_t kShortRadius = 42;
constexpr int32_t kLongRadius = 64;
constexpr int32_t kSmallWidth = 12;
constexpr double kCrowdingDivisor = 2.9;
constexpr int32_t kLate = 1000;
constexpr int32_t kLevelWeight = 1000;
constexpr int32_t kSites = 13;
constexpr int32_t kAttempts = 2;
constexpr int32_t kMaxChases = kTargets;

constexpr int kCanAttack = 3;
constexpr int kAttackRange = 4;
constexpr int kCost = 5;
constexpr int kTableColumns = 6;

constexpr int kAttackDamage = 0;
constexpr int kAttackTime = 1;
constexpr int kHitPoints = 2;
constexpr int kProduceTime = 3;
constexpr int kMoveTime = 4;
constexpr int kTurnColumns = 5;

constexpr int kCommandKind = 0;
constexpr int kCommandDirection = 1;
constexpr int kCommandNoneDuration = 2;
constexpr int kCommandTargetCell = 3;
constexpr int kCommandProducedType = 4;
constexpr int kCommandRequestOrder = 5;

constexpr uint16_t kPresent = 1u << 0;
constexpr uint16_t kMine = 1u << 1;
constexpr uint16_t kEnemy = 1u << 2;
constexpr uint16_t kResource = 1u << 3;
constexpr uint16_t kCarrying = 1u << 4;
constexpr uint16_t kArmed = 1u << 5;
constexpr uint16_t kApart = 1u << 6;
constexpr uint16_t kUnassigned = 1u << 7;
constexpr uint16_t kOpenBase = 1u << 8;

constexpr uint8_t kWall = 1u << 0;
constexpr uint8_t kOccupied = 1u << 1;
constexpr uint8_t kClaimed = 1u << 2;
constexpr uint8_t kFree = 1u << 3;
constexpr uint8_t kTaken = 1u << 4;
constexpr uint8_t kFutureBarracks = 1u << 5;
constexpr uint8_t kBlocking = 1u << 6;
constexpr uint8_t kBuilding = 1u << 7;

constexpr uint8_t kSucceeds = 1u << 0;
constexpr uint8_t kSoon = 1u << 1;
constexpr uint8_t kThreatened = 1u << 2;

__constant__ int32_t kStepX[kDirections] = {0, 1, 0, -1};
__constant__ int32_t kStepY[kDirections] = {-1, 0, 1, 0};
__constant__ int32_t kListed[kDirections] = {kUp, kDown, kLeft, kRight};
__constant__ int32_t kAroundX[kDirections] = {-1, 0, 0, 1};
__constant__ int32_t kAroundY[kDirections] = {0, 1, -1, 0};
__constant__ int32_t kAroundDirection[kDirections] = {kLeft, kDown, kUp, kRight};
__constant__ int32_t kSiteX[kSites] = {0, -1, 0, 0, 1, -2, -1, -1, 0, 0, 1, 1, 2};
__constant__ int32_t kSiteY[kSites] = {0, 0, 1, -1, 0, 0, 1, -1, 2, -2, 1, -1, 0};

constexpr int64_t shared_bytes(int64_t cells, int64_t capacity, int64_t entities) {
    return kBytesPerCell * cells + kLaneBytes + kBytesPerSlot * capacity + kBytesPerEntity * entities;
}

__device__ int32_t clip(int32_t value, int32_t low, int32_t high) {
    return value < low ? low : value > high ? high : value;
}

__device__ int32_t wrapped_add(int32_t left, int32_t right) {
    return static_cast<int32_t>(static_cast<uint32_t>(left) + static_cast<uint32_t>(right));
}

__device__ int32_t wrapped_sub(int32_t left, int32_t right) {
    return static_cast<int32_t>(static_cast<uint32_t>(left) - static_cast<uint32_t>(right));
}

__device__ int32_t wrapped_mul(int32_t left, int32_t right) {
    return static_cast<int32_t>(static_cast<uint32_t>(left) * static_cast<uint32_t>(right));
}

__device__ int32_t warp_sum(int32_t value) {
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(kEveryThread, value, offset);
    }

    return value;
}

__device__ int32_t warp_min(int32_t value) {
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        value = min(value, __shfl_xor_sync(kEveryThread, value, offset));
    }

    return value;
}

__device__ int32_t warp_max(int32_t value) {
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        value = max(value, __shfl_xor_sync(kEveryThread, value, offset));
    }

    return value;
}

__device__ bool warp_any(bool value) {
    return __any_sync(kEveryThread, value) != 0;
}

struct Pick {
    int64_t key;
    int32_t index;
};

__device__ Pick nothing_picked() {
    return Pick{kNoKey, kHighest};
}

__device__ void offer(Pick& best, int64_t key, int32_t index) {
    if (key < best.key || (key == best.key && index < best.index)) {
        best.key = key;
        best.index = index;
    }
}

__device__ Pick least(Pick best) {
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        const int64_t key = __shfl_xor_sync(kEveryThread, best.key, offset);
        const int32_t index = __shfl_xor_sync(kEveryThread, best.index, offset);
        offer(best, key, index);
    }

    return best;
}

__device__ int32_t picked(const Pick& best) {
    return best.index == kHighest ? kAbsent : best.index;
}

__device__ int64_t keyed(int32_t primary, int32_t secondary) {
    return static_cast<int64_t>(primary) * kKeyRadix + static_cast<int64_t>(static_cast<uint32_t>(secondary));
}

__device__ int32_t floor_sqrt(int32_t value, int32_t bound) {
    int32_t root = 0;

    while (root < bound && (root + 1) * (root + 1) <= value) {
        ++root;
    }

    return root;
}

template <typename Slot, typename Cell>
struct Inputs {
    StateRows<Slot, Cell> state;
    Lanes<int32_t> table;
    Lanes<int32_t> player;
    Lanes<int16_t> hit_points;
    Lanes<Cell> pending_target_cell;
    Lanes<int32_t> pending_completes_at;
    Lanes<int32_t> time;
    Lanes<int32_t> turn_table;
    Lanes<uint8_t> ranks;
    Lanes<int32_t> holder;
    Lanes<int32_t> order;
    Lanes<int32_t> next_order;
};

struct Outputs {
    int32_t* holder;
    int32_t* order;
    int32_t* next_order;
    int32_t* commands;
};

struct Shared {
    astar::Scratch scratch;
    uint8_t* cells;
    uint32_t* wall_rows;
    uint32_t* occupied_rows;
    uint32_t* free_rows;
    uint32_t* taken_rows;
    uint32_t* target_rows;
    int16_t* chases;
    int32_t* birth;
    int32_t* remaining;
    int32_t* damage;
    int32_t* value;
    int32_t* target;
    int32_t* holder;
    int32_t* held;
    int32_t* site_score;
    int16_t* cell;
    int16_t* future;
    int16_t* hit_points;
    int16_t* counts;
    uint16_t* bits;
    int16_t* to_resource;
    int16_t* catalog;
    int16_t* rank;
    int16_t* by_birth;
    int16_t* walk;
    int8_t* x;
    int8_t* y;
    int8_t* fx;
    int8_t* fy;
    uint8_t* type;
    int8_t* pending;
    uint8_t* marks;
    int8_t* kind;
    int8_t* direction;
    int8_t* produced;
    int8_t* none_duration;

    __device__ static Shared carve(uint64_t* memory, const Extents& size) {
        const int32_t cells = size.cells;
        const int32_t capacity = size.capacity;
        const int32_t entities = size.entities;
        Shared carved;
        carved.scratch = astar::Scratch::carve(memory, cells);
        carved.cells = carved.scratch.seen + cells;
        char* const lane = reinterpret_cast<char*>(memory) + kBytesPerCell * cells;
        uint32_t* rows = reinterpret_cast<uint32_t*>(lane);
        carved.wall_rows = rows;
        carved.occupied_rows = rows + kMaxRows;
        carved.free_rows = rows + 2 * kMaxRows;
        carved.taken_rows = rows + 3 * kMaxRows;
        carved.target_rows = rows + 4 * kMaxRows;
        carved.chases = reinterpret_cast<int16_t*>(rows + 5 * kMaxRows);
        int32_t* words = reinterpret_cast<int32_t*>(lane + kLaneBytes);
        carved.birth = words;
        carved.remaining = words + entities;
        carved.damage = words + 2 * entities;
        carved.value = words + 3 * entities;
        words += 4 * entities;
        carved.target = words;
        carved.holder = words + capacity;
        carved.held = words + 2 * capacity;
        carved.site_score = words + 3 * capacity;
        int16_t* halves = reinterpret_cast<int16_t*>(words + (3 + kSites) * capacity);
        carved.cell = halves;
        carved.future = halves + entities;
        carved.hit_points = halves + 2 * entities;
        carved.counts = halves + 3 * entities;
        carved.bits = reinterpret_cast<uint16_t*>(halves + 4 * entities);
        carved.to_resource = halves + 5 * entities;
        carved.catalog = halves + 6 * entities;
        halves += 7 * entities;
        carved.rank = halves;
        carved.by_birth = halves + capacity;
        carved.walk = halves + 2 * capacity;
        int8_t* bytes = reinterpret_cast<int8_t*>(halves + 3 * capacity);
        carved.x = bytes;
        carved.y = bytes + entities;
        carved.fx = bytes + 2 * entities;
        carved.fy = bytes + 3 * entities;
        carved.type = reinterpret_cast<uint8_t*>(bytes + 4 * entities);
        carved.pending = bytes + 5 * entities;
        carved.marks = reinterpret_cast<uint8_t*>(bytes + 6 * entities);
        bytes += 7 * entities;
        carved.kind = bytes;
        carved.direction = bytes + capacity;
        carved.produced = bytes + 2 * capacity;
        carved.none_duration = bytes + 3 * capacity;
        return carved;
    }
};

static_assert(5 * kMaxRows * sizeof(uint32_t) + kMaxChases * sizeof(int16_t) <= kLaneBytes, "the lane's rows outgrew their bytes");
static_assert(4 * 4 + 7 * 2 + 7 == kBytesPerEntity, "the per-entity arrays and their bytes disagree");
static_assert((3 + kSites) * 4 + 3 * 2 + 4 == kBytesPerSlot, "the per-slot arrays and their bytes disagree");

struct Plan {
    int32_t direction;
    int32_t target;
    bool moved;
};

struct Turn {
    Shared s;
    const int32_t* table;
    const int32_t* turn_table;
    const uint8_t* ranks;
    int32_t lane;
    int32_t width;
    int32_t height;
    int32_t cells;
    int32_t entities;
    int32_t capacity;
    int32_t types;
    int32_t player;
    int32_t other;
    int32_t start;
    int32_t time;
    int32_t balance;
    int32_t enemy_balance;
    int32_t reserved;
    int32_t enemy_reserved;
    int32_t radius;
    int32_t diagonal_root;
    bool decisive;
    int32_t own_count;
    int32_t resource_count;
    int32_t own_base_count;
    int32_t enemy_base_count;
    int32_t enemies;
    int32_t workers;
    int32_t barracks;
    int32_t combatants;
    int32_t enemy_workers;
    int32_t enemy_future_heavies;
    int32_t occupancy;
    int32_t power;
    int32_t enemy_power;
    int32_t enemy_heavies;
    int32_t lone_hit_points;
    bool enemy_ranged;
    bool own_heavy;
    bool own_ranged;
    bool weak;
    int32_t requests;
    int32_t used;
    int32_t future_heavies;
    int32_t draws;
    int32_t next_order;

    __device__ int32_t column(int32_t type, int at) const {
        return table[clip(type, 0, types - 1) * kTableColumns + at];
    }

    __device__ int32_t turn_column(int32_t type, int at) const {
        return turn_table[clip(type, 0, types - 1) * kTurnColumns + at];
    }

    __device__ int32_t cost(int32_t type) const {
        return column(type, kCost);
    }

    __device__ int32_t reach(int32_t type) const {
        return column(type, kAttackRange);
    }

    __device__ bool armed(int32_t type) const {
        return column(type, kCanAttack) != 0;
    }

    __device__ int32_t x_of(int32_t cell) const {
        return astar::floor_mod(cell, width);
    }

    __device__ int32_t y_of(int32_t cell) const {
        return astar::floor_div(cell, width);
    }

    __device__ int32_t manhattan(int32_t cell, int32_t other_cell) const {
        return abs(x_of(cell) - x_of(other_cell)) + abs(y_of(cell) - y_of(other_cell));
    }

    __device__ int32_t squared(int32_t cell, int32_t other_cell) const {
        const int32_t dx = x_of(cell) - x_of(other_cell);
        const int32_t dy = y_of(cell) - y_of(other_cell);
        return dx * dx + dy * dy;
    }

    __device__ int32_t distance_to(int32_t e, int32_t x, int32_t y) const {
        return abs(s.x[e] - x) + abs(s.y[e] - y);
    }

    __device__ int32_t squared_to(int32_t e, int32_t x, int32_t y) const {
        const int32_t dx = s.x[e] - x;
        const int32_t dy = s.y[e] - y;
        return dx * dx + dy * dy;
    }

    __device__ int32_t squared_to_future(int32_t e, int32_t x, int32_t y) const {
        const int32_t dx = s.fx[e] - x;
        const int32_t dy = s.fy[e] - y;
        return dx * dx + dy * dy;
    }

    __device__ bool on_board(int32_t x, int32_t y) const {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    __device__ int32_t offset_cell(int32_t cell, int32_t dx, int32_t dy) const {
        const int32_t x = x_of(cell) + dx;
        const int32_t y = y_of(cell) + dy;
        return on_board(x, y) ? y * width + x : kAbsent;
    }

    __device__ int32_t step(int32_t cell, int32_t direction) const {
        if (direction < 0 || direction >= kDirections) {
            return kAbsent;
        }

        return offset_cell(cell, kStepX[direction], kStepY[direction]);
    }

    __device__ int32_t future(int32_t cell, int32_t direction) const {
        const int32_t stepped = step(cell, direction);
        return stepped == kAbsent ? cell : stepped;
    }

    __device__ bool at(uint8_t bit, int32_t cell) const {
        return cell != kAbsent && (s.cells[clip(cell, 0, cells - 1)] & bit) != 0;
    }

    __device__ bool has(int32_t e, uint16_t bit) const {
        return (s.bits[e] & bit) != 0;
    }

    __device__ bool is(int32_t e, uint16_t bit, int32_t type) const {
        return (s.bits[e] & bit) != 0 && s.type[e] == type;
    }

    __device__ bool busy(int32_t slot) const {
        return s.kind[clip(slot - start, 0, capacity - 1)] != kKindAbsent || s.pending[slot] != kKindAbsent;
    }

    __device__ bool busy_all(int32_t e) const {
        const int32_t local = e - start;
        return (local >= 0 && local < capacity && s.kind[local] != kKindAbsent) || s.pending[e] != kKindAbsent;
    }

    __device__ bool consistent(int32_t cell, int32_t price) const {
        return !at(kClaimed, cell) && (reserved == 0 || price + reserved <= balance) && (enemy_reserved == 0 || enemy_reserved <= enemy_balance);
    }

    __device__ bool move_allowed(int32_t target) const {
        return target != kAbsent && !at(kWall, target) && !at(kOccupied, target) && consistent(target, 0);
    }

    __device__ bool pos_free(int32_t cell) const {
        return !at(kTaken, cell) && !at(kOccupied, cell) && !at(kWall, cell);
    }

    __device__ bool openable(int32_t cell) const {
        return cell != kAbsent && !at(kWall, cell) && !at(kOccupied, cell);
    }

    __device__ int32_t toward(int32_t origin, int32_t goal) const {
        const int32_t dx = x_of(goal) - x_of(origin);
        const int32_t dy = y_of(goal) - y_of(origin);
        const int32_t horizontal = dx > 0 ? kRight : kLeft;
        const int32_t vertical = dy > 0 ? kDown : kUp;
        return abs(dx) > abs(dy) ? horizontal : vertical;
    }

    __device__ int32_t to_resource(int32_t e) const {
        const int32_t distance = s.to_resource[e];
        return distance < 0 ? kHighest : distance;
    }

    __device__ void issue(int32_t slot, bool when, int32_t kind, int32_t direction, int32_t target_cell, int32_t produced_type, int32_t none_duration) {
        const int32_t local = slot - start;

        if (!when || local < 0 || local >= capacity) {
            return;
        }

        const bool first = s.kind[local] == kKindAbsent;
        __syncwarp();

        if (lane == 0) {
            s.kind[local] = static_cast<int8_t>(kind);
            s.direction[local] = static_cast<int8_t>(direction);
            s.target[local] = target_cell;
            s.produced[local] = static_cast<int8_t>(produced_type);
            s.none_duration[local] = static_cast<int8_t>(none_duration);

            if (first) {
                s.rank[local] = static_cast<int16_t>(requests);
            }
        }

        __syncwarp();

        if (first) {
            ++requests;
        }
    }

    __device__ void lock(int32_t cell, bool when, uint8_t bit) {
        if (!when || cell < 0 || cell >= cells) {
            return;
        }

        if (lane == 0) {
            s.cells[cell] |= bit;

            if (bit == kTaken) {
                s.taken_rows[cell / width] |= 1u << (cell % width);
            }
        }

        __syncwarp();
    }

    template <typename Among>
    __device__ int32_t closest(int32_t origin, Among among) const {
        const int32_t ox = x_of(origin);
        const int32_t oy = y_of(origin);
        Pick best = nothing_picked();

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (among(e)) {
                offer(best, keyed(distance_to(e, ox, oy), s.birth[e]), e);
            }
        }

        return picked(least(best));
    }

    __device__ int32_t closest_with(int32_t origin, uint16_t bit) const {
        return closest(origin, [&](int32_t e) { return has(e, bit); });
    }

    __device__ int32_t closest_of(int32_t origin, uint16_t bit, int32_t type) const {
        return closest(origin, [&](int32_t e) { return is(e, bit, type); });
    }

    template <typename Keep>
    __device__ int32_t list_in_unit_order(Keep keep, int32_t written) {
        for (int32_t base = 0; base < own_count; base += kThreads) {
            const int32_t index = base + lane;
            const int32_t slot = index < own_count ? s.by_birth[index] : 0;
            const bool kept = index < own_count && keep(slot);
            const unsigned ballot = __ballot_sync(kEveryThread, kept);

            if (kept) {
                s.walk[written + __popc(ballot & ((1u << lane) - 1u))] = static_cast<int16_t>(slot);
            }

            written += __popc(ballot);
        }

        __syncwarp();
        return written;
    }

    __device__ astar::Path search(int32_t origin, int32_t goal) {
        for (int32_t c = lane; c < cells; c += kThreads) {
            s.scratch.clear(c, (s.cells[c] & (kFree | kTaken)) == kFree);
        }

        __syncwarp();
        astar::Path path{0, kAbsent, kAbsent, 0};

        if (lane == 0) {
            path = astar::search(s.scratch, origin, goal, 1, width, height);
        }

        __syncwarp();
        path.found = __shfl_sync(kEveryThread, path.found, 0);
        path.step = __shfl_sync(kEveryThread, path.step, 0);
        return path;
    }

    __device__ uint32_t full_row() const {
        return width >= kMaxWidth ? 0xffffffffu : (1u << width) - 1u;
    }

    __device__ uint32_t spread(uint32_t front) const {
        const uint32_t above = __shfl_up_sync(kEveryThread, front, 1);
        const uint32_t below = __shfl_down_sync(kEveryThread, front, 1);
        return ((front << 1) & full_row()) | (front >> 1) | (lane > 0 ? above : 0u) | (lane < kThreads - 1 ? below : 0u);
    }

    __device__ bool row_bit(uint32_t row, int32_t cell) const {
        const uint32_t word = __shfl_sync(kEveryThread, row, cell / width);
        return ((word >> (cell % width)) & 1u) != 0;
    }

    __device__ void proximity(int32_t goal, const int32_t (&neighbour)[kDirections], int32_t (&near)[kDirections]) {
        const bool owns_row = lane < height;
        const uint32_t blocked = owns_row ? (s.wall_rows[lane] | s.taken_rows[lane] | s.occupied_rows[lane]) : 0u;
        const bool goal_on_board = goal >= 0 && goal < cells;
        uint32_t front = goal_on_board && owns_row && lane == goal / width ? 1u << (goal % width) : 0u;
        uint32_t reached = front;
        bool opening[kDirections];

        for (int k = 0; k < kDirections; ++k) {
            near[k] = neighbour[k] != kAbsent && neighbour[k] == goal ? 0 : kHighest;
            opening[k] = openable(neighbour[k]);
        }

        int32_t level = 0;

        while (true) {
            bool going = level < radius && warp_any(front != 0u);

            if (decisive) {
                for (int k = 0; k < kDirections; ++k) {
                    going = going && !(opening[k] && near[k] != kHighest);
                }
            }

            if (!going) {
                break;
            }

            const uint32_t grown = spread(front);
            const uint32_t fresh = owns_row ? grown & ~blocked & ~reached : 0u;
            reached |= fresh;

            for (int k = 0; k < kDirections; ++k) {
                if (near[k] == kHighest && neighbour[k] != kAbsent && row_bit(fresh, neighbour[k])) {
                    near[k] = level + 1;
                }
            }

            front = fresh;
            ++level;
        }
    }

    __device__ bool breadth_first_direction(int32_t origin, int32_t goal, int32_t& direction) {
        int32_t neighbour[kDirections];

        for (int k = 0; k < kDirections; ++k) {
            neighbour[k] = offset_cell(origin, kAroundX[k], kAroundY[k]);
        }

        int32_t near[kDirections];
        proximity(goal, neighbour, near);
        bool any_open = false;
        int64_t best_key = kNoKey;
        int best = 0;

        for (int k = 0; k < kDirections; ++k) {
            if (openable(neighbour[k]) && near[k] != kHighest) {
                const int32_t fit = -kLevelWeight * near[k] - floor_sqrt(squared(neighbour[k], goal), diagonal_root);
                const int64_t key = -static_cast<int64_t>(fit);

                if (!any_open || key < best_key) {
                    best_key = key;
                    best = k;
                }

                any_open = true;
            }
        }

        direction = kAroundDirection[best];
        return any_open;
    }

    __device__ Plan plan_move(int32_t slot, int32_t goal) {
        const int32_t origin = s.cell[slot];
        const astar::Path path = search(origin, goal);

        if (path.found == 0 || path.step == kAbsent) {
            return Plan{kAbsent, kAbsent, false};
        }

        int32_t toward_goal = kAbsent;
        const bool open = breadth_first_direction(origin, goal, toward_goal);
        const int32_t direction = open ? toward_goal : path.step;
        const int32_t target = step(origin, direction);
        return Plan{direction, target, move_allowed(target) && !at(kTaken, target)};
    }

    __device__ void apply_move(int32_t slot, const Plan& plan) {
        issue(slot, plan.moved, kKindMove, plan.direction, kAbsent, kAbsent, 0);
        lock(plan.target, plan.moved, kTaken);
    }

    __device__ bool move_towards(int32_t slot, int32_t goal, bool active) {
        if (!active) {
            return false;
        }

        const Plan plan = plan_move(slot, goal);
        apply_move(slot, plan);
        return plan.moved;
    }

    __device__ void shuffled_move(int32_t slot, const bool (&passing)[kDirections]) {
        if (!(passing[0] || passing[1] || passing[2] || passing[3])) {
            return;
        }

        const uint32_t packed = ranks[min(draws, 2 * capacity - 1)];
        int32_t drawn = 0;
        uint32_t place = kDirections;

        for (int d = 0; d < kDirections; ++d) {
            const uint32_t at_place = (packed >> (2 * d)) & 3u;

            if (passing[d] && at_place < place) {
                place = at_place;
                drawn = d;
            }
        }

        const int32_t target = step(s.cell[slot], drawn);
        issue(slot, true, kKindMove, drawn, kAbsent, kAbsent, 0);
        lock(target, true, kTaken);
        ++draws;
    }

    __device__ void move_away(int32_t slot, bool active) {
        if (!active) {
            return;
        }

        bool passing[kDirections];

        for (int d = 0; d < kDirections; ++d) {
            const int32_t target = step(s.cell[slot], d);
            passing[d] = pos_free(target) && move_allowed(target);
        }

        shuffled_move(slot, passing);
    }

    __device__ void move_closer(int32_t slot, int32_t goal, bool active) {
        if (!active) {
            return;
        }

        const int32_t origin = s.cell[slot];
        const int32_t here = manhattan(origin, goal);
        bool passing[kDirections];

        for (int d = 0; d < kDirections; ++d) {
            const int32_t target = step(origin, d);
            passing[d] = manhattan(max(target, 0), goal) < here && pos_free(target) && move_allowed(target);
        }

        shuffled_move(slot, passing);
    }

    __device__ bool producible(int32_t slot, int32_t target, int32_t price) const {
        return !busy(slot) && balance - used >= price && pos_free(target) && consistent(target, price);
    }

    __device__ bool produce(int32_t slot, int32_t direction, int32_t produced_type, bool active) {
        if (!active) {
            return false;
        }

        const int32_t price = cost(produced_type);
        const int32_t target = future(s.cell[slot], direction);

        if (!producible(slot, target, price)) {
            return false;
        }

        issue(slot, true, kKindProduce, direction, kAbsent, produced_type, 0);
        lock(target, true, kTaken);

        if (produced_type == kTypeBarracks) {
            lock(target, true, kFutureBarracks);
        }

        if (produced_type == kTypeHeavy) {
            ++future_heavies;
        }

        used += price;
        return true;
    }

    __device__ bool produce_first(int32_t slot, const int32_t (&directions)[kDirections], int32_t produced_type, bool active) {
        if (!active) {
            return false;
        }

        const int32_t price = cost(produced_type);

        for (int k = 0; k < kDirections; ++k) {
            if (producible(slot, future(s.cell[slot], directions[k]), price)) {
                return produce(slot, directions[k], produced_type, true);
            }
        }

        return false;
    }

    __device__ bool separated(int32_t slot, bool active) {
        if (!active) {
            return true;
        }

        const int32_t origin = s.cell[slot];
        const int32_t ox = x_of(origin);
        const int32_t oy = y_of(origin);

        if (lane < kMaxRows) {
            s.target_rows[lane] = 0u;
        }

        __syncwarp();

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (!has(e, kEnemy) || squared_to(e, ox, oy) <= 1) {
                continue;
            }

            for (int d = 0; d < kDirections; ++d) {
                const int32_t beside = offset_cell(s.cell[e], kStepX[d], kStepY[d]);

                if (beside != kAbsent) {
                    atomicOr(&s.target_rows[beside / width], 1u << (beside % width));
                }
            }
        }

        __syncwarp();
        const bool owns_row = lane < height;
        const uint32_t targets = owns_row ? s.target_rows[lane] : 0u;
        const uint32_t open = owns_row ? s.free_rows[lane] & ~s.taken_rows[lane] : 0u;
        uint32_t seeds = 0u;

        for (int d = 0; d < kDirections; ++d) {
            const int32_t beside = offset_cell(origin, kStepX[d], kStepY[d]);

            if (beside != kAbsent && owns_row && lane == beside / width) {
                seeds |= 1u << (beside % width);
            }
        }

        seeds &= open;
        uint32_t previous = seeds;
        uint32_t reached = open & (seeds | spread(seeds));

        while (warp_any(reached != previous) && !warp_any((reached & targets) != 0u)) {
            previous = reached;
            reached = open & (reached | spread(reached));
        }

        return !warp_any((reached & targets) != 0u);
    }

    __device__ int32_t targets_tried() const {
        const int32_t cut = combatants > kManyCombatants ? kFewTargets : kTargets;
        return min(enemies, cut);
    }

    __device__ Pick next_target(int64_t after) const {
        Pick best = nothing_picked();

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (has(e, kEnemy)) {
                const int64_t key = keyed(-s.value[e], s.birth[e]);

                if (key > after) {
                    offer(best, key, e);
                }
            }
        }

        return least(best);
    }

    __device__ void attack_nearby(int32_t slot) {
        const int32_t unit_type = s.type[slot];

        if (!has(slot, kMine) || busy(slot) || !armed(unit_type)) {
            return;
        }

        const int32_t ox = s.x[slot];
        const int32_t oy = s.y[slot];
        const int32_t range = reach(unit_type);
        const int32_t ranged_reach = reach(kTypeRanged) + kFarFromRanged;
        const int32_t own_damage = turn_column(unit_type, kAttackDamage);
        const int32_t own_attack_time = turn_column(unit_type, kAttackTime);

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (!has(e, kEnemy)) {
                continue;
            }

            const int32_t target_type = s.type[e];
            const int32_t distance = squared_to(e, ox, oy);
            const bool succeeds = distance <= range * range && !(s.remaining[e] <= own_attack_time);
            const int32_t target_reach = reach(target_type);
            const bool threatened = distance <= target_reach * target_reach;
            const bool kills = own_damage > s.hit_points[e];
            const int32_t to_worker = unit_type == kTypeRanged && target_type == kTypeWorker ? 1 : 0;
            const int32_t score = 1000 * (succeeds ? 1 : 0) + 100 * (kills ? 1 : 0) + (to_worker + turn_column(target_type, kAttackDamage)) * 10 + (threatened ? 1 : 0);
            const bool hopeless = s.hit_points[e] <= s.damage[e] || distance > ranged_reach * ranged_reach;
            const bool soon = squared_to_future(e, ox, oy) <= range * range;
            s.value[e] = hopeless ? -1 : score;
            s.marks[e] = (succeeds ? kSucceeds : 0) | (soon ? kSoon : 0) | (threatened ? kThreatened : 0);
        }

        __syncwarp();
        const int32_t limit = targets_tried();
        const bool allowed = consistent(kAbsent, 0);
        int32_t first_decisive = limit;
        int32_t decisive = kAbsent;
        bool decisive_hits = false;
        bool decisive_waits = false;
        int32_t chase_count = 0;
        int64_t after = kBeforeEveryKey;

        for (int32_t tried = 0; tried < limit; ++tried) {
            const Pick next = next_target(after);
            const int32_t e = picked(next);

            if (e == kAbsent) {
                break;
            }

            after = next.key;
            const uint8_t marks = s.marks[e];
            const bool succeeds = (marks & kSucceeds) != 0;
            const bool soon = (marks & kSoon) != 0;
            const bool hits = succeeds && allowed;
            const bool waits = !succeeds && soon;

            if (hits || waits) {
                first_decisive = tried;
                decisive = e;
                decisive_hits = hits;
                decisive_waits = waits;
                break;
            }

            if (!succeeds && !soon && (marks & kThreatened) != 0) {
                if (lane == 0) {
                    s.chases[chase_count] = static_cast<int16_t>(e);
                }

                ++chase_count;
            }
        }

        __syncwarp();
        Plan plan{kAbsent, kAbsent, false};

        for (int32_t chased = 0; chased < chase_count && !plan.moved; ++chased) {
            plan = plan_move(slot, s.cell[s.chases[chased]]);
        }

        apply_move(slot, plan);
        const bool acts = !plan.moved && first_decisive < limit;

        if (acts && decisive_hits) {
            issue(slot, true, kKindAttack, kAbsent, s.cell[decisive], kAbsent, 0);

            if (lane == 0) {
                s.damage[decisive] += turn_column(unit_type, kHitPoints);
            }

            __syncwarp();
        }

        issue(slot, acts && decisive_waits, kKindNone, kAbsent, kAbsent, kAbsent, kWait);
    }

    __device__ bool in_contact(int32_t slot) const {
        const int32_t ox = s.x[slot];
        const int32_t oy = s.y[slot];
        const int32_t range = reach(s.type[slot]);
        bool touching = false;

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (!has(e, kEnemy)) {
                continue;
            }

            const int32_t distance = squared_to(e, ox, oy);
            const int32_t target_reach = reach(s.type[e]);
            touching = touching || distance <= range * range || squared_to_future(e, ox, oy) <= range * range || distance <= target_reach * target_reach;
        }

        return warp_any(touching);
    }

    __device__ void all_attack_nearby() {
        const int32_t listed = list_in_unit_order([&](int32_t slot) { return armed(s.type[slot]) && !busy_all(slot); }, 0);
        int32_t acting = 0;

        for (int32_t index = 0; index < listed; ++index) {
            const int32_t slot = s.walk[index];
            const bool touching = in_contact(slot);
            __syncwarp();

            if (touching) {
                if (lane == 0) {
                    s.walk[acting] = static_cast<int16_t>(slot);
                }

                ++acting;
            }

            __syncwarp();
        }

        for (int32_t index = 0; index < acting; ++index) {
            attack_nearby(s.walk[index]);
        }
    }

    __device__ int32_t planned_barracks() const {
        int32_t planned = 0;

        for (int32_t c = lane; c < cells; c += kThreads) {
            planned += (s.cells[c] & kFutureBarracks) != 0 ? 1 : 0;
        }

        return warp_sum(planned);
    }

    __device__ int32_t distance_to_cells(int32_t origin, uint8_t bit) const {
        const int32_t ox = x_of(origin);
        const int32_t oy = y_of(origin);
        int32_t best = kHighest;

        for (int32_t c = lane; c < cells; c += kThreads) {
            if ((s.cells[c] & bit) != 0) {
                best = min(best, abs(c % width - ox) + abs(c / width - oy));
            }
        }

        return warp_min(best);
    }

    __device__ int32_t distance_to_own(int32_t origin, int32_t type) const {
        const int32_t ox = x_of(origin);
        const int32_t oy = y_of(origin);
        int32_t best = kHighest;

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (is(e, kMine, type)) {
                best = min(best, distance_to(e, ox, oy));
            }
        }

        return warp_min(best);
    }

    __device__ bool needs_barracks() {
        const int32_t planned = planned_barracks();

        if (!(barracks + planned < own_base_count)) {
            return false;
        }

        const int32_t reach_width = width / 4;
        bool needed = false;

        for (int32_t index = 0; index < own_base_count; ++index) {
            const int32_t base = s.catalog[resource_count + index];
            const int32_t origin = s.cell[base];
            const bool far_from_planned = distance_to_cells(origin, kFutureBarracks) > reach_width;
            const bool near_built = distance_to_own(origin, kTypeBarracks) < reach_width;
            needed = needed || (far_from_planned && !near_built);
        }

        return needed;
    }

    __device__ int32_t danger_time() const {
        const int32_t worker_move = turn_column(kTypeWorker, kMoveTime);
        return turn_column(kTypeBarracks, kProduceTime) / worker_move + turn_column(kTypeHeavy, kProduceTime) / worker_move;
    }

    struct Site {
        int32_t x;
        int32_t y;
        int32_t enemy_x;
        int32_t enemy_y;
        int32_t enemy_distance;
        int32_t danger;
    };

    __device__ Site site_at(int32_t site) const {
        const int32_t enemy = max(closest_with(site, kEnemy), 0);
        const int32_t enemy_cell = s.cell[enemy];
        return Site{x_of(site), y_of(site), x_of(enemy_cell), y_of(enemy_cell), max(manhattan(site, enemy_cell), 1), danger_time()};
    }

    __device__ int32_t site_worker_score(int32_t w, const Site& site) const {
        if (!is(w, kMine, kTypeWorker) || busy_all(w)) {
            return kLowest;
        }

        const int32_t wx = s.x[w];
        const int32_t wy = s.y[w];
        const bool beside = !((wx < site.x && site.enemy_x < site.x) || (wx > site.x && site.enemy_x > site.x) || (wy < site.y && site.enemy_y < site.y) || (wy > site.y && site.enemy_y > site.y));
        const int32_t penalty = site.enemy_distance < site.danger ? (2 * site.danger) / site.enemy_distance - (beside ? 3 : 0) : 0;
        return -penalty - (abs(wx - site.x) + abs(wy - site.y)) / 2;
    }

    __device__ int32_t best_site_worker(const Site& site) const {
        int32_t best = kLowest;

        for (int32_t e = lane; e < entities; e += kThreads) {
            best = max(best, site_worker_score(e, site));
        }

        return warp_max(best);
    }

    __device__ int32_t block_penalty(int32_t sx, int32_t sy, bool diagonals) const {
        int32_t total = 0;

        for (int k = 1; k < kSites; ++k) {
            const int32_t distance = abs(kSiteX[k]) + abs(kSiteY[k]);

            if (distance == 2 && !diagonals) {
                continue;
            }

            const int32_t x = sx + kSiteX[k];
            const int32_t y = sy + kSiteY[k];

            if (on_board(x, y) && (s.cells[y * width + x] & kBlocking) != 0) {
                total += distance == 1 ? 4 : 1;
            }
        }

        return total;
    }

    __device__ int32_t site_of(int32_t row, int32_t k) const {
        const int32_t base = s.walk[row];
        const int32_t sx = s.x[base] + kSiteX[k];
        const int32_t sy = s.y[base] + kSiteY[k];
        return on_board(sx, sy) ? sy * width + sx : kAbsent;
    }

    __device__ int32_t score_sites() {
        const int32_t rows = list_in_unit_order([&](int32_t slot) { return s.type[slot] == kTypeBase; }, 0);

        for (int32_t row = 0; row < rows; ++row) {
            const int32_t base = s.walk[row];
            const bool apart = separated(base, true);

            if (lane == 0) {
                s.bits[base] = apart ? (s.bits[base] | kApart) : (s.bits[base] & ~kApart);
            }

            __syncwarp();
        }

        const bool any_barracks = barracks > 0 || planned_barracks() > 0;

        for (int32_t row = 0; row < rows; ++row) {
            for (int32_t k = 0; k < kSites; ++k) {
                const int32_t site = site_of(row, k);
                int32_t score = kLowest;

                if (site != kAbsent && workers > 0 && !at(kWall, site) && !at(kBuilding, site)) {
                    const int32_t near = max(closest_of(site, kMine, kTypeBase), 0);
                    const int32_t near_cell = s.cell[near];
                    const int32_t deserted = any_barracks ? min(distance_to_cells(near_cell, kFutureBarracks), distance_to_own(near_cell, kTypeBarracks)) / 2 : 0;
                    const int32_t best = best_site_worker(site_at(site));
                    const int32_t sx = x_of(site);
                    const int32_t sy = y_of(site);
                    const int32_t penalty = block_penalty(sx, sy, false);
                    const int32_t diagonal_penalty = block_penalty(sx, sy, true);
                    const int32_t open_score = wrapped_mul(10, wrapped_add(deserted - penalty, best));
                    score = has(near, kApart) ? -diagonal_penalty * 10 : open_score;
                }

                if (lane == 0) {
                    s.site_score[row * kSites + k] = score;
                }
            }
        }

        __syncwarp();
        return rows;
    }

    __device__ void build_barracks() {
        const bool affordable = balance - used - 1 >= cost(kTypeBarracks);

        if (!(affordable && own_base_count > 0 && workers > 0 && needs_barracks())) {
            return;
        }

        const int32_t rows = score_sites();
        const int32_t entries = rows * kSites;
        int32_t removed[kAttempts] = {kAbsent, kAbsent};
        bool done = false;

        for (int32_t tries = 0; tries < kAttempts && !done; ++tries) {
            Pick best = nothing_picked();

            for (int32_t index = lane; index < entries; index += kThreads) {
                bool remaining = true;

                for (int32_t r = 0; r < tries; ++r) {
                    remaining = remaining && index != removed[r];
                }

                offer(best, -static_cast<int64_t>(remaining ? s.site_score[index] : kLowest), index);
            }

            best = least(best);
            const int32_t index = entries > 0 ? best.index : 0;
            const int32_t chosen_score = entries - tries > 0 ? s.site_score[index] : kLowest;
            const int32_t site = index < entries ? site_of(index / kSites, index % kSites) : kAbsent;
            bool goes = false;
            bool moved = false;
            bool built = false;

            if (chosen_score != kLowest) {
                const Site known = site_at(max(site, 0));
                const int32_t top = best_site_worker(known);
                Pick first = nothing_picked();

                for (int32_t e = lane; e < entities; e += kThreads) {
                    if (is(e, kMine, kTypeWorker) && site_worker_score(e, known) == top) {
                        offer(first, s.birth[e], e);
                    }
                }

                const int32_t worker = max(picked(least(first)), 0);
                goes = site_worker_score(worker, known) != kLowest;

                if (goes) {
                    const bool beside = manhattan(s.cell[worker], site) == 1;
                    moved = move_towards(worker, site, !beside);
                    built = produce(worker, toward(s.cell[worker], site), kTypeBarracks, beside);
                }
            }

            done = !goes || moved || built;
            removed[tries] = index;
        }
    }

    __device__ void build_base() {
        if (!(workers > 0 && own_base_count == 0 && resource_count > 0 && balance - used >= cost(kTypeBase))) {
            return;
        }

        Pick best = nothing_picked();

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (is(e, kMine, kTypeWorker)) {
                offer(best, keyed(busy_all(e) ? kHighest : to_resource(e), s.birth[e]), e);
            }
        }

        const int32_t nearest = max(picked(least(best)), 0);

        if (busy_all(nearest)) {
            return;
        }

        const int32_t price = cost(kTypeBase);

        for (int k = 0; k < kDirections; ++k) {
            const int32_t site = future(s.cell[nearest], kListed[k]);

            if (!at(kWall, site) && !at(kBuilding, site) && pos_free(site) && consistent(site, price)) {
                produce(nearest, kListed[k], kTypeBase, true);
                return;
            }
        }
    }

    __device__ void combat_directions(int32_t slot, int32_t (&directions)[kDirections]) const {
        int32_t origin[kDirections];
        int32_t distance[kDirections];

        for (int k = 0; k < kDirections; ++k) {
            const int32_t cell = future(s.cell[slot], kListed[k]);
            origin[k] = cell;
            distance[k] = kHighest;
        }

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (has(e, kEnemy)) {
                for (int k = 0; k < kDirections; ++k) {
                    distance[k] = min(distance[k], distance_to(e, x_of(origin[k]), y_of(origin[k])));
                }
            }
        }

        int order[kDirections];

        for (int k = 0; k < kDirections; ++k) {
            distance[k] = warp_min(distance[k]);
            order[k] = k;
        }

        for (int i = 1; i < kDirections; ++i) {
            const int held = order[i];
            int j = i - 1;

            while (j >= 0 && distance[order[j]] > distance[held]) {
                order[j + 1] = order[j];
                --j;
            }

            order[j + 1] = held;
        }

        for (int k = 0; k < kDirections; ++k) {
            directions[k] = kListed[order[k]];
        }
    }

    __device__ void barracks_action() {
        const int32_t listed = list_in_unit_order([&](int32_t slot) { return s.type[slot] == kTypeBarracks; }, 0);
        const bool exhausted = resource_count == 0;

        for (int32_t index = 0; index < listed; ++index) {
            const int32_t slot = s.walk[index];

            if (busy(slot)) {
                continue;
            }

            const bool apart = separated(slot, true);
            int32_t directions[kDirections];
            combat_directions(slot, directions);
            bool done = produce_first(slot, directions, kTypeRanged, apart);
            done = produce_first(slot, directions, kTypeHeavy, !done) || done;
            done = produce_first(slot, directions, kTypeRanged, !done && weak) || done;
            const bool poor = balance - used < cost(kTypeHeavy);
            produce_first(slot, directions, kTypeRanged, !done && exhausted && poor);
        }
    }

    __device__ int32_t workers_per_base(int32_t slot) {
        if (width < 9) {
            return 15;
        }

        if (width > 16) {
            return 2;
        }

        const bool apart = separated(slot, true);
        const int32_t below = enemy_workers / max(enemy_base_count, 1);
        return apart || time > kLate ? 2 : max(below, 2);
    }

    __device__ int32_t build_worker_direction(int32_t slot) const {
        const int32_t origin = s.cell[slot];
        const int32_t enemy = closest_with(origin, kEnemy);
        const int32_t resource = closest_with(origin, kResource);
        const bool toward_enemy = resource == kAbsent || workers >= 2 * own_base_count;
        const int32_t goal = toward_enemy ? enemy : resource;
        const int32_t goal_cell = s.cell[max(goal, 0)];
        int64_t best_key = kNoKey;
        int best = 0;
        bool any = false;

        for (int k = 0; k < kDirections; ++k) {
            const int32_t near = future(origin, kListed[k]);
            const int32_t beyond = future(near, kListed[k]);
            const bool usable = !at(kWall, near) && !at(kOccupied, near) && pos_free(beyond) && enemy != kAbsent;

            if (usable) {
                const int64_t key = static_cast<int64_t>(manhattan(near, goal_cell));

                if (!any || key < best_key) {
                    best_key = key;
                    best = k;
                }

                any = true;
            }
        }

        return any ? kListed[best] : kUp;
    }

    __device__ void bases_action() {
        const int32_t listed = list_in_unit_order([&](int32_t slot) { return s.type[slot] == kTypeBase; }, 0);
        int32_t producing = 0;

        for (int32_t index = lane; index < listed; index += kThreads) {
            producing += s.pending[s.walk[index]] != kKindAbsent ? 1 : 0;
        }

        producing = warp_sum(producing);
        int32_t made_workers = 0;

        for (int32_t index = 0; index < listed; ++index) {
            const int32_t slot = s.walk[index];

            if (busy(slot)) {
                continue;
            }

            const bool only = resource_count == 0 && balance - used == 1;
            produce_first(slot, kListed, kTypeWorker, only);

            if (only) {
                continue;
            }

            const bool enough = width >= 9 && workers + made_workers + producing >= workers_per_base(slot) * own_base_count;

            if (enough) {
                continue;
            }

            const int32_t direction = build_worker_direction(slot);
            const bool made = produce(slot, direction, kTypeWorker, true);
            const bool anywhere = produce_first(slot, kListed, kTypeWorker, !made);
            made_workers += made || anywhere ? 1 : 0;
        }
    }

    __device__ void go_harvesting(int32_t slot, bool active) {
        if (!active) {
            return;
        }

        const int32_t origin = s.cell[slot];
        const int32_t resource = closest_with(origin, kResource);

        if (resource == kAbsent) {
            return;
        }

        const int32_t goal = s.cell[resource];
        const bool beside = manhattan(origin, goal) == 1;
        issue(slot, beside && !busy(slot) && consistent(kAbsent, 0), kKindHarvest, toward(origin, goal), kAbsent, kAbsent, 0);
        const bool moved = move_towards(slot, goal, !beside);
        move_away(slot, !beside && !moved);
    }

    __device__ int32_t harvest_score(int32_t e) const {
        if (busy_all(e) || has(e, kCarrying)) {
            return kHighest;
        }

        int32_t to_base = kHighest;

        for (int32_t index = 0; index < own_base_count; ++index) {
            const int32_t base = s.catalog[resource_count + index];

            if (has(base, kOpenBase)) {
                to_base = min(to_base, abs(s.x[e] - s.x[base]) + abs(s.y[e] - s.y[base]));
            }
        }

        return to_resource(e) + to_base;
    }

    __device__ void remember_order() {
        for (int32_t local = lane; local < capacity; local += kThreads) {
            const int32_t order = s.held[local];

            if (s.holder[local] == kAbsent) {
                continue;
            }

            int32_t place = 0;

            for (int32_t other_local = 0; other_local < capacity; ++other_local) {
                const int32_t other_order = s.held[other_local];
                place += s.holder[other_local] != kAbsent && (other_order < order || (other_order == order && other_local < local)) ? 1 : 0;
            }

            s.walk[place] = static_cast<int16_t>(start + local);
        }

        __syncwarp();
    }

    __device__ void worker_action() {
        const bool crowded = width <= kSmallWidth && occupancy > static_cast<int32_t>(static_cast<double>(width * height) / kCrowdingDivisor);
        const int32_t per_base = crowded ? 1 : 2;
        int32_t remembered = 0;

        for (int32_t local = lane; local < capacity; local += kThreads) {
            remembered += s.holder[local] != kAbsent ? 1 : 0;
        }

        remembered = warp_sum(remembered);
        remember_order();

        for (int32_t e = lane; e < entities; e += kThreads) {
            s.counts[e] = 0;
            s.bits[e] = is(e, kMine, kTypeWorker) ? (s.bits[e] | kUnassigned) : (s.bits[e] & ~kUnassigned);
        }

        __syncwarp();

        for (int32_t index = 0; index < remembered; ++index) {
            const int32_t slot = s.walk[index];
            const int32_t base = closest_of(s.cell[slot], kMine, kTypeBase);
            go_harvesting(slot, !busy(slot) && !has(slot, kCarrying));

            if (lane == 0) {
                s.bits[slot] &= ~kUnassigned;

                if (base != kAbsent) {
                    ++s.counts[base];
                }
            }

            __syncwarp();
        }

        bool any_open = false;

        for (int32_t index = lane; index < own_base_count; index += kThreads) {
            const int32_t base = s.catalog[resource_count + index];
            const bool open = s.counts[base] < per_base;
            s.bits[base] = open ? (s.bits[base] | kOpenBase) : (s.bits[base] & ~kOpenBase);
            any_open = any_open || open;
        }

        any_open = warp_any(any_open);
        __syncwarp();
        bool any_unassigned = false;

        for (int32_t e = lane; e < entities; e += kThreads) {
            any_unassigned = any_unassigned || has(e, kUnassigned);
        }

        any_unassigned = warp_any(any_unassigned);

        while (any_open && any_unassigned && resource_count > 0) {
            Pick best = nothing_picked();

            for (int32_t e = lane; e < entities; e += kThreads) {
                if (has(e, kUnassigned)) {
                    offer(best, keyed(harvest_score(e), s.birth[e]), e);
                }
            }

            best = least(best);

            if (best.index == kHighest || (best.key >> 32) == kHighest) {
                break;
            }

            const int32_t worker = best.index;
            const int32_t base = max(closest(s.cell[worker], [&](int32_t e) { return has(e, kOpenBase); }), 0);
            go_harvesting(worker, true);
            const int32_t local = worker - start;

            if (lane == 0) {
                if (local >= 0 && local < capacity) {
                    s.holder[local] = s.birth[worker];
                    s.held[local] = next_order;
                }

                if (s.counts[base] == 0) {
                    s.counts[base] = 1;
                }

                if (s.counts[base] >= per_base) {
                    s.bits[base] &= ~kOpenBase;
                }

                s.bits[worker] &= ~kUnassigned;
            }

            ++next_order;
            __syncwarp();
            any_open = false;
            any_unassigned = false;

            for (int32_t e = lane; e < entities; e += kThreads) {
                any_open = any_open || has(e, kOpenBase);
                any_unassigned = any_unassigned || has(e, kUnassigned);
            }

            any_open = warp_any(any_open);
            any_unassigned = warp_any(any_unassigned);
        }

        if (own_base_count == 0) {
            return;
        }

        const int32_t delivering = list_in_unit_order([&](int32_t slot) { return s.type[slot] == kTypeWorker && !busy_all(slot) && has(slot, kCarrying); }, 0);

        for (int32_t index = 0; index < delivering; ++index) {
            const int32_t slot = s.walk[index];
            const int32_t base = closest_of(s.cell[slot], kMine, kTypeBase);

            if (busy(slot) || !has(slot, kCarrying) || base == kAbsent) {
                continue;
            }

            const int32_t origin = s.cell[slot];
            const int32_t goal = s.cell[base];
            const bool beside = manhattan(origin, goal) <= 1;
            issue(slot, beside && consistent(kAbsent, 0), kKindReturn, toward(origin, goal), kAbsent, kAbsent, 0);
            move_towards(slot, goal, !beside);
        }
    }

    __device__ bool workers_attack() const {
        if (width <= kSmallWidth) {
            return true;
        }

        return weak && !enemy_ranged && !own_heavy && future_heavies == 0 && !own_ranged;
    }

    __device__ bool crowded_cell(int32_t e) const {
        for (int k = 0; k < kDirections; ++k) {
            if (at(kOccupied, offset_cell(s.cell[e], kAroundX[k], kAroundY[k]))) {
                return true;
            }
        }

        return false;
    }

    __device__ void free_blocks() {
        const int32_t listed = list_in_unit_order([&](int32_t slot) { return s.type[slot] == kTypeWorker && !busy_all(slot) && crowded_cell(slot); }, 0);

        for (int32_t index = 0; index < listed; ++index) {
            const int32_t slot = s.walk[index];
            move_away(slot, !busy(slot) && crowded_cell(slot));
        }
    }

    __device__ void combat_visit(int32_t slot, int32_t tried_at_most, bool overpowering) {
        const int32_t unit_type = s.type[slot];

        if (busy(slot) || !armed(unit_type)) {
            return;
        }

        const int32_t ox = s.x[slot];
        const int32_t oy = s.y[slot];
        const bool ranged = unit_type == kTypeRanged;
        const bool heavy_or_ranged = unit_type == kTypeHeavy || ranged;

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (!has(e, kEnemy)) {
                continue;
            }

            int32_t score = -distance_to(e, ox, oy);

            if (width > 9 && ranged && s.type[e] == kTypeRanged) {
                score += 2;
            }

            if (width >= 16 && heavy_or_ranged && s.type[e] == kTypeBarracks) {
                score += width;
            }

            s.value[e] = score;
        }

        __syncwarp();
        bool boxed = true;

        for (int k = 0; k < kDirections; ++k) {
            const int32_t beside = offset_cell(s.cell[slot], kAroundX[k], kAroundY[k]);

            if (beside != kAbsent && (s.cells[beside] & (kFree | kTaken)) == kFree) {
                boxed = false;
            }
        }

        const int32_t limit = boxed ? 0 : tried_at_most;
        const Pick first = next_target(kBeforeEveryKey);
        Pick cursor = first;
        Plan plan{kAbsent, kAbsent, false};
        int32_t index = 0;

        while (!plan.moved && index < limit) {
            plan = plan_move(slot, s.future[max(picked(cursor), 0)]);

            if (!plan.moved) {
                ++index;
                cursor = next_target(cursor.key);
            }
        }

        apply_move(slot, plan);
        const int32_t tried = boxed ? tried_at_most : index;
        const bool stuck = enemies > 0 && tried >= enemies && ranged;
        move_away(slot, stuck && overpowering);
        move_closer(slot, s.cell[max(picked(first), 0)], stuck);
    }

    __device__ void go_combat(bool attack) {
        const auto able = [&](int32_t slot) { return !busy_all(slot) && armed(s.type[slot]); };
        int32_t listed = 0;

        if (attack) {
            listed = list_in_unit_order([&](int32_t slot) { return s.type[slot] == kTypeWorker && able(slot); }, listed);
        }

        listed = list_in_unit_order([&](int32_t slot) { return s.type[slot] == kTypeHeavy && able(slot); }, listed);
        listed = list_in_unit_order([&](int32_t slot) { return s.type[slot] == kTypeRanged && able(slot); }, listed);
        listed = list_in_unit_order([&](int32_t slot) { return s.type[slot] == kTypeLight && able(slot); }, listed);
        const int32_t tried_at_most = targets_tried();
        const bool overpowering = power - enemy_power > 0;

        for (int32_t index = 0; index < listed; ++index) {
            combat_visit(s.walk[index], tried_at_most, overpowering);
        }
    }

    __device__ void fill_with_nones() {
        int32_t placed = requests;

        for (int32_t base = 0; base < own_count; base += kThreads) {
            const int32_t index = base + lane;
            const int32_t slot = index < own_count ? s.by_birth[index] : 0;
            const int32_t local = slot - start;
            const bool waits = index < own_count && s.pending[slot] == kKindAbsent && s.kind[local] == kKindAbsent;
            const unsigned ballot = __ballot_sync(kEveryThread, waits);

            if (waits) {
                s.kind[local] = kKindNone;
                s.none_duration[local] = kWait;
                s.rank[local] = static_cast<int16_t>(placed + __popc(ballot & ((1u << lane) - 1u)));
            }

            placed += __popc(ballot);
        }

        __syncwarp();
        requests = placed;
    }

    __device__ __forceinline__ void run() {
        all_attack_nearby();
        build_barracks();
        build_base();
        barracks_action();
        bases_action();
        worker_action();
        const bool attack = workers_attack();

        if (!attack) {
            free_blocks();
        }

        go_combat(attack);
        fill_with_nones();
    }
};

__device__ void or_cell(uint8_t* bits, int32_t cell, uint8_t bit) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(bits + cell);
    atomicOr(reinterpret_cast<uint32_t*>(address & ~uintptr_t{3}), static_cast<uint32_t>(bit) << (8 * (address & 3)));
}

__device__ int32_t scattered(int32_t index, int32_t size) {
    const int32_t wrapped = index < 0 ? index + size : index;
    return wrapped >= 0 && wrapped < size ? wrapped : kAbsent;
}

template <typename Slot, typename Cell>
__device__ void load(Turn& a, const Inputs<Slot, Cell>& in, int64_t lane_index) {
    const Shared& s = a.s;
    const int32_t* birth = in.state.birth.at(lane_index);
    const int8_t* unit_type = in.state.unit_type.at(lane_index);
    const int8_t* owner = in.state.owner.at(lane_index);
    const Cell* cell = in.state.cell.at(lane_index);
    const int32_t* carried = in.state.carried.at(lane_index);
    const int8_t* pending_kind = in.state.pending_kind.at(lane_index);
    const int8_t* pending_direction = in.state.pending_direction.at(lane_index);
    const int8_t* pending_produced_type = in.state.pending_produced_type.at(lane_index);
    const Slot* entity_at = in.state.entity_at.at(lane_index);
    const bool* terrain = in.state.terrain.at(lane_index);
    const int16_t* hit_points = in.hit_points.at(lane_index);
    const Cell* pending_target_cell = in.pending_target_cell.at(lane_index);
    const int32_t* pending_completes_at = in.pending_completes_at.at(lane_index);
    const int32_t* holder = in.holder.at(lane_index);
    const int32_t* order = in.order.at(lane_index);
    const int32_t lane = a.lane;

    for (int32_t c = lane; c < a.cells; c += kThreads) {
        s.cells[c] = (terrain[c] ? kWall : 0) | (entity_at[c] != kAbsent ? kOccupied : 0);
    }

    for (int32_t e = lane; e < a.entities; e += kThreads) {
        const int32_t born = birth[e];
        const bool present = born != kAbsent;
        const int32_t type = clip(unit_type[e], 0, a.types - 1);
        const int32_t holder_of = owner[e];
        const int32_t at = static_cast<int32_t>(cell[e]);
        const int32_t kind = pending_kind[e];
        const bool resource = present && type == kTypeResource;
        const bool mine = present && !resource && holder_of == a.player;
        const bool enemy = present && !resource && holder_of != kAbsent && holder_of != a.player;
        const bool moving = present && kind == kKindMove;
        const int32_t future = moving ? a.future(at, pending_direction[e]) : at;
        s.birth[e] = born;
        s.cell[e] = static_cast<int16_t>(at);
        s.future[e] = static_cast<int16_t>(future);
        s.x[e] = static_cast<int8_t>(a.x_of(at));
        s.y[e] = static_cast<int8_t>(a.y_of(at));
        s.fx[e] = static_cast<int8_t>(a.x_of(future));
        s.fy[e] = static_cast<int8_t>(a.y_of(future));
        s.hit_points[e] = hit_points[e];
        s.type[e] = static_cast<uint8_t>(type);
        s.pending[e] = static_cast<int8_t>(present ? kind : kKindAbsent);
        s.remaining[e] = moving ? wrapped_sub(pending_completes_at[e], a.time) : kHighest;
        s.damage[e] = 0;
        s.bits[e] = (present ? kPresent : 0) | (mine ? kMine : 0) | (enemy ? kEnemy : 0) | (resource ? kResource : 0) | (carried[e] > 0 ? kCarrying : 0) | (a.armed(type) ? kArmed : 0);
    }

    for (int32_t local = lane; local < a.capacity; local += kThreads) {
        s.kind[local] = kKindAbsent;
        s.direction[local] = kAbsent;
        s.target[local] = kAbsent;
        s.produced[local] = kAbsent;
        s.none_duration[local] = 0;
        s.rank[local] = kAbsent;
    }

    __syncwarp();
    int32_t reserved[nanorts::whole_act::kPlayers] = {0, 0};
    int32_t used = 0;
    int32_t future_heavies = 0;
    int32_t enemy_future_heavies = 0;
    bool based = false;

    for (int32_t e = lane; e < a.entities; e += kThreads) {
        const uint16_t bits = s.bits[e];

        if ((bits & kPresent) == 0) {
            continue;
        }

        const int32_t at = s.cell[e];
        const int32_t kind = s.pending[e];
        const int32_t direction = pending_direction[e];
        const int32_t produced = max(static_cast<int32_t>(pending_produced_type[e]), 0);
        const bool resource = (bits & kResource) != 0;
        const bool mine = (bits & kMine) != 0;
        const int32_t type = s.type[e];

        if (kind == kKindMove || kind == kKindProduce) {
            const int32_t claimed = scattered(a.step(at, direction), a.cells);

            if (claimed != kAbsent) {
                or_cell(s.cells, claimed, kClaimed);
            }

            const int32_t raw = at + kStepY[clip(direction, 0, kDirections - 1)] * a.width + kStepX[clip(direction, 0, kDirections - 1)];
            const int32_t raw_claimed = scattered(raw, a.cells);

            if (raw_claimed != kAbsent) {
                or_cell(s.cells, raw_claimed, kFree);
            }
        }

        if (kind == kKindProduce && e >= a.entities - nanorts::whole_act::kPlayers * a.capacity) {
            reserved[(e - (a.entities - nanorts::whole_act::kPlayers * a.capacity)) / a.capacity] += a.cost(produced);
        }

        if (!resource && kind == kKindProduce) {
            const int32_t built = a.future(at, direction);
            or_cell(s.cells, built, kTaken);

            if (mine) {
                used += a.cost(produced);
                future_heavies += produced == kTypeHeavy ? 1 : 0;

                if (produced == kTypeBarracks) {
                    or_cell(s.cells, built, kFutureBarracks);
                }
            }
        }

        if (kind == kKindProduce && produced == kTypeHeavy && (bits & kEnemy) != 0) {
            ++enemy_future_heavies;
        }

        if (!resource && kind == kKindAttack) {
            const int32_t target = static_cast<int32_t>(pending_target_cell[e]);
            const int32_t victim = static_cast<int32_t>(entity_at[clip(target, 0, a.cells - 1)]);

            if (target != kAbsent && victim != kAbsent) {
                atomicAdd(&s.damage[victim], a.turn_column(type, kAttackDamage));
            }
        }

        if ((mine || (bits & kEnemy) != 0 || resource) && (type == kTypeResource || type == kTypeBase)) {
            or_cell(s.cells, at, kBlocking);
        }

        if ((mine || (bits & kEnemy) != 0) && (type == kTypeBase || type == kTypeBarracks)) {
            or_cell(s.cells, at, kBuilding);
        }

        based = based || (mine && type == kTypeBase);
    }

    __syncwarp();

    for (int32_t c = lane; c < a.cells; c += kThreads) {
        const uint8_t bits = s.cells[c];
        const bool free = (bits & (kWall | kOccupied | kFree)) == 0;
        s.cells[c] = (bits & ~kFree) | (free ? kFree : 0);
    }

    __syncwarp();

    if (lane < kMaxRows) {
        uint32_t wall = 0u;
        uint32_t occupied = 0u;
        uint32_t free = 0u;
        uint32_t taken = 0u;

        if (lane < a.height) {
            for (int32_t x = 0; x < a.width; ++x) {
                const uint8_t bits = s.cells[lane * a.width + x];
                const uint32_t bit = 1u << x;
                wall |= (bits & kWall) != 0 ? bit : 0u;
                occupied |= (bits & kOccupied) != 0 ? bit : 0u;
                free |= (bits & kFree) != 0 ? bit : 0u;
                taken |= (bits & kTaken) != 0 ? bit : 0u;
            }
        }

        s.wall_rows[lane] = wall;
        s.occupied_rows[lane] = occupied;
        s.free_rows[lane] = free;
        s.taken_rows[lane] = taken;
    }

    a.reserved = warp_sum(reserved[a.player]);
    a.enemy_reserved = warp_sum(reserved[a.other]);
    a.used = warp_sum(used);
    a.future_heavies = warp_sum(future_heavies);
    a.enemy_future_heavies = warp_sum(enemy_future_heavies);
    based = warp_any(based);

    for (int32_t local = lane; local < a.capacity; local += kThreads) {
        const int32_t held = holder[local];
        const bool alive = held != kAbsent && held == birth[a.start + local] && based;
        s.holder[local] = alive ? held : kAbsent;
        s.held[local] = alive ? order[local] : kAbsent;
    }

    int32_t present = 0;
    int32_t enemies = 0;
    int32_t workers = 0;
    int32_t barracks = 0;
    int32_t combatants = 0;
    int32_t enemy_workers = 0;
    int32_t occupancy = 0;
    int32_t power = 0;
    int32_t enemy_power = 0;
    int32_t enemy_heavies = 0;
    int32_t lone_hit_points = kLowest;
    bool enemy_ranged = false;
    bool own_heavy = false;
    bool own_ranged = false;

    for (int32_t e = lane; e < a.entities; e += kThreads) {
        const uint16_t bits = s.bits[e];
        const int32_t type = s.type[e];
        const bool mine = (bits & kMine) != 0;
        const bool enemy = (bits & kEnemy) != 0;
        const bool armed = (bits & kArmed) != 0;
        const bool unit = mine || enemy;
        present += (bits & kPresent) != 0 ? 1 : 0;
        enemies += enemy ? 1 : 0;
        workers += mine && type == kTypeWorker ? 1 : 0;
        barracks += mine && type == kTypeBarracks ? 1 : 0;
        combatants += enemy && armed ? 1 : 0;
        enemy_workers += enemy && type == kTypeWorker ? 1 : 0;
        occupancy += (unit && type == kTypeWorker ? 1 : 0) + (unit && armed ? 1 : 0) + (unit && type == kTypeBase ? 1 : 0) + (unit && type == kTypeBarracks ? 1 : 0) + ((bits & kResource) != 0 ? 1 : 0);
        power += mine && armed ? a.turn_column(type, kAttackDamage) : 0;
        enemy_power += enemy && armed ? a.turn_column(type, kAttackDamage) : 0;
        enemy_heavies += enemy && type == kTypeHeavy ? 1 : 0;
        lone_hit_points = enemy && type == kTypeHeavy ? max(lone_hit_points, static_cast<int32_t>(s.hit_points[e])) : lone_hit_points;
        enemy_ranged = enemy_ranged || (enemy && type == kTypeRanged);
        own_heavy = own_heavy || (mine && type == kTypeHeavy);
        own_ranged = own_ranged || (mine && type == kTypeRanged);
    }

    a.radius = warp_sum(present) > kFewUnits ? kShortRadius : kLongRadius;
    a.enemies = warp_sum(enemies);
    a.workers = warp_sum(workers);
    a.barracks = warp_sum(barracks);
    a.combatants = warp_sum(combatants);
    a.enemy_workers = warp_sum(enemy_workers);
    a.occupancy = warp_sum(occupancy);
    a.power = warp_sum(power);
    a.enemy_power = warp_sum(enemy_power);
    a.enemy_heavies = warp_sum(enemy_heavies);
    a.lone_hit_points = warp_max(lone_hit_points);
    a.enemy_ranged = warp_any(enemy_ranged);
    a.own_heavy = warp_any(own_heavy);
    a.own_ranged = warp_any(own_ranged);

    int32_t listed = 0;
    const auto catalogue = [&](auto keep) {
        for (int32_t base = 0; base < a.entities; base += kThreads) {
            const int32_t e = base + lane;
            const bool kept = e < a.entities && keep(e);
            const unsigned ballot = __ballot_sync(kEveryThread, kept);

            if (kept) {
                s.catalog[listed + __popc(ballot & ((1u << lane) - 1u))] = static_cast<int16_t>(e);
            }

            listed += __popc(ballot);
        }
    };
    catalogue([&](int32_t e) { return a.has(e, kResource); });
    a.resource_count = listed;
    catalogue([&](int32_t e) { return a.is(e, kMine, kTypeBase); });
    a.own_base_count = listed - a.resource_count;
    catalogue([&](int32_t e) { return a.is(e, kEnemy, kTypeBase); });
    a.enemy_base_count = listed - a.resource_count - a.own_base_count;
    __syncwarp();

    for (int32_t e = lane; e < a.entities; e += kThreads) {
        int32_t nearest = kHighest;

        for (int32_t index = 0; index < a.resource_count; ++index) {
            const int32_t r = s.catalog[index];
            nearest = min(nearest, abs(s.x[e] - s.x[r]) + abs(s.y[e] - s.y[r]));
        }

        s.to_resource[e] = static_cast<int16_t>(nearest == kHighest ? -1 : nearest);
    }

    __syncwarp();
    int32_t near_workers = 0;

    for (int32_t e = lane; e < a.entities; e += kThreads) {
        if (!a.is(e, kEnemy, kTypeWorker)) {
            continue;
        }

        int32_t to_base = kHighest;

        for (int32_t index = 0; index < a.enemy_base_count; ++index) {
            const int32_t b = s.catalog[a.resource_count + a.own_base_count + index];
            to_base = min(to_base, abs(s.x[e] - s.x[b]) + abs(s.y[e] - s.y[b]));
        }

        const int32_t to_resource = a.has(e, kCarrying) ? 0 : a.to_resource(e);
        near_workers += wrapped_add(to_base, to_resource) < a.width / 2 ? 1 : 0;
    }

    const int32_t total = wrapped_add(a.enemy_balance, warp_sum(near_workers));
    a.weak = a.enemy_future_heavies == 0 && a.enemy_heavies <= 1 && (a.enemy_heavies == 0 || a.lone_hit_points <= 3) && total < a.cost(kTypeHeavy);

    for (int32_t local = lane; local < a.capacity; local += kThreads) {
        const int32_t slot = a.start + local;

        if (!a.has(slot, kMine)) {
            continue;
        }

        const int32_t born = s.birth[slot];
        int32_t place = 0;

        for (int32_t other_local = 0; other_local < a.capacity; ++other_local) {
            const int32_t other_slot = a.start + other_local;
            place += a.has(other_slot, kMine) && s.birth[other_slot] < born ? 1 : 0;
        }

        s.by_birth[place] = static_cast<int16_t>(slot);
    }

    int32_t own = 0;

    for (int32_t local = lane; local < a.capacity; local += kThreads) {
        own += a.has(a.start + local, kMine) ? 1 : 0;
    }

    a.own_count = warp_sum(own);
    __syncwarp();
}

template <typename Slot, typename Cell>
__global__ void __launch_bounds__(kThreads) act_kernel(Inputs<Slot, Cell> in, Outputs out, Extents size) {
    extern __shared__ uint64_t memory[];
    const int64_t lane_index = blockIdx.x;
    Turn a;
    a.s = Shared::carve(memory, size);
    a.table = in.table.at(lane_index);
    a.turn_table = in.turn_table.at(lane_index);
    a.ranks = in.ranks.at(lane_index);
    a.lane = threadIdx.x;
    a.width = size.width;
    a.height = size.height;
    a.cells = size.cells;
    a.entities = size.entities;
    a.capacity = size.capacity;
    a.types = size.types;
    a.player = in.player.at(lane_index)[0];
    a.other = 1 - a.player;
    a.start = size.neutral + a.player * size.capacity;
    a.time = in.time.at(lane_index)[0];
    const int32_t* resources = in.state.resources.at(lane_index);
    a.balance = resources[clip(a.player, 0, 1)];
    a.enemy_balance = resources[clip(a.other, 0, 1)];
    const int32_t diagonal = (size.width - 1) * (size.width - 1) + (size.height - 1) * (size.height - 1);
    a.diagonal_root = floor_sqrt(diagonal, diagonal) + 1;
    a.decisive = diagonal < kLevelWeight * kLevelWeight;
    a.requests = 0;
    a.draws = 0;
    a.next_order = in.next_order.at(lane_index)[0];
    load(a, in, lane_index);
    a.run();

    using nanorts::whole_act::kCommandFields;
    using nanorts::whole_act::kPlayers;
    const Shared& s = a.s;
    int32_t* commands = out.commands + lane_index * kCommandFields * kPlayers * size.capacity;

    for (int32_t local = a.lane; local < size.capacity; local += kThreads) {
        int32_t issued[kCommandFields];
        issued[kCommandKind] = s.kind[local];
        issued[kCommandDirection] = s.direction[local];
        issued[kCommandNoneDuration] = s.none_duration[local];
        issued[kCommandTargetCell] = s.target[local];
        issued[kCommandProducedType] = s.produced[local];
        issued[kCommandRequestOrder] = s.rank[local];
        int32_t unissued[kCommandFields];
        unissued[kCommandKind] = kKindAbsent;
        unissued[kCommandDirection] = kAbsent;
        unissued[kCommandNoneDuration] = 0;
        unissued[kCommandTargetCell] = kAbsent;
        unissued[kCommandProducedType] = kAbsent;
        unissued[kCommandRequestOrder] = kAbsent;

        for (int field = 0; field < kCommandFields; ++field) {
            for (int32_t who = 0; who < kPlayers; ++who) {
                commands[(field * kPlayers + who) * size.capacity + local] = who == a.player ? issued[field] : unissued[field];
            }
        }

        out.holder[lane_index * size.capacity + local] = s.holder[local];
        out.order[lane_index * size.capacity + local] = s.held[local];
    }

    if (a.lane == 0) {
        out.next_order[lane_index] = a.next_order;
    }
}

template <typename Slot, typename Cell>
ffi::Error launch(cudaStream_t stream, nanorts::whole_act::Operands& operands, const Outputs& outputs, const Extents& size, int64_t lanes) {
    Inputs<Slot, Cell> in{};
    const int64_t entities = size.entities;
    const int64_t capacity = size.capacity;

    if (auto e = nanorts::whole_act::bind_state(operands, size, in.state); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{size.types} * kTableColumns, in.table); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.player); e.failure()) {
        return e;
    }

    if (auto e = operands.next(entities, in.hit_points); e.failure()) {
        return e;
    }

    if (auto e = operands.next(entities, in.pending_target_cell); e.failure()) {
        return e;
    }

    if (auto e = operands.next(entities, in.pending_completes_at); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.time); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{size.types} * kTurnColumns, in.turn_table); e.failure()) {
        return e;
    }

    if (auto e = operands.next(2 * capacity, in.ranks); e.failure()) {
        return e;
    }

    if (auto e = operands.next(capacity, in.holder); e.failure()) {
        return e;
    }

    if (auto e = operands.next(capacity, in.order); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.next_order); e.failure()) {
        return e;
    }

    const int64_t shared = shared_bytes(size.cells, size.capacity, size.entities);
    static nanorts::whole_act::SharedOptIn opt_in;

    if (auto e = opt_in.allow(operands, nanorts::kernel(act_kernel<Slot, Cell>), shared); e.failure()) {
        return e;
    }

    act_kernel<Slot, Cell><<<static_cast<unsigned>(lanes), kThreads, static_cast<size_t>(shared), stream>>>(in, outputs, size);
    return nanorts::launched();
}

constexpr int64_t kExtraOperands = 6;
constexpr int64_t kReadOperands = nanorts::whole_act::kStateOperands + 2 + kExtraOperands;
constexpr int64_t kCarryLeaves = 3;
constexpr int64_t kHitPointsOperand = nanorts::whole_act::kStateOperands + 2;
constexpr int64_t kTargetCellOperand = kHitPointsOperand + 1;
constexpr int64_t kTurnTableOperand = kHitPointsOperand + 4;
constexpr int64_t kRanksOperand = kHitPointsOperand + 5;

ffi::Error Act(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets results, std::string_view build_id) {
    using ffi::DataType;
    constexpr std::string_view kTarget = "nanorts_mayari_act";

    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (args.size() != kReadOperands + kCarryLeaves || results.size() != kCarryLeaves + 1) {
        return ffi::Error::InvalidArgument(
            "nanorts_mayari_act takes the State leaves, the unit tables, the player, the shuffle "
            "ranks and the carry, and returns the carry and the Commands");
    }

    auto commands = results.get<ffi::AnyBuffer>(kCarryLeaves);

    if (!commands.has_value()) {
        return commands.error();
    }

    ffi::AnyBuffer table_rows = *args.get<ffi::AnyBuffer>(nanorts::whole_act::kStateOperands);
    ffi::AnyBuffer turn_rows = *args.get<ffi::AnyBuffer>(kTurnTableOperand);
    ffi::AnyBuffer rank_rows = *args.get<ffi::AnyBuffer>(kRanksOperand);
    const auto issued = (*commands)->dimensions();
    const auto table = table_rows.dimensions();
    const auto turned = turn_rows.dimensions();

    if ((*commands)->element_type() != ffi::S32 || issued.size() != 4 || issued[1] != nanorts::whole_act::kCommandFields || issued[2] != nanorts::whole_act::kPlayers || table.size() != 3 || table[2] != kTableColumns || table_rows.element_type() != ffi::S32 || turned.size() != 3 || turned[1] != table[1] || turned[2] != kTurnColumns || turn_rows.element_type() != ffi::S32 || rank_rows.element_type() != ffi::U8) {
        return ffi::Error::InvalidArgument(
            "the Commands are `[lanes, 6, 2, C]` int32, the unit tables `[1, T, 6]` and "
            "`[1, T, 5]` int32, and the shuffle ranks uint8");
    }

    const int64_t lanes = issued[0];
    nanorts::whole_act::Operands operands(kTarget, args, lanes);
    Extents size{};
    DataType slot = DataType::INVALID;
    DataType cell = DataType::INVALID;

    if (auto e = nanorts::whole_act::read_state(operands, size, slot, cell); e.failure()) {
        return e;
    }

    size.capacity = static_cast<int32_t>(issued[3]);
    size.neutral = size.entities - nanorts::whole_act::kPlayers * size.capacity;
    size.types = static_cast<int32_t>(table[1]);
    const int64_t shared = shared_bytes(size.cells, size.capacity, size.entities);

    if (size.cells <= 0 || size.capacity <= 0 || size.neutral < 0 || size.types <= kTypeRanged || size.height > kMaxRows || size.width > kMaxWidth || shared > nanorts::whole_act::kSharedOptInLimit || nanorts::whole_act::threads_for(size.capacity) > nanorts::whole_act::kMaxThreads) {
        return ffi::Error::InvalidArgument(
            "a table holding every unit type mayari names, over a lane of at most 32 rows and "
            "32 columns whose act fits one block's shared memory");
    }

    auto hit_points = operands.at(kHitPointsOperand);
    auto target_cell = operands.at(kTargetCellOperand);

    if (!hit_points.has_value()) {
        return hit_points.error();
    }

    if (!target_cell.has_value()) {
        return target_cell.error();
    }

    if (hit_points->element_type() != DataType::S16 || target_cell->element_type() != cell) {
        return ffi::Error::InvalidArgument("hit points are int16 and pending targets share the cells' dtype");
    }

    Outputs outputs{};
    int64_t result = 0;
    int64_t operand = kReadOperands;
    const auto bind = [&](int64_t elements, auto*& into) { return nanorts::whole_act::carry_result(operands, results, result++, operand++, lanes, elements, into); };

    if (auto e = bind(size.capacity, outputs.holder); e.failure()) {
        return e;
    }

    if (auto e = bind(size.capacity, outputs.order); e.failure()) {
        return e;
    }

    if (auto e = bind(1, outputs.next_order); e.failure()) {
        return e;
    }

    outputs.commands = static_cast<int32_t*>((*commands)->untyped_data());

    if (lanes == 0) {
        return ffi::Error::Success();
    }

    if (slot == DataType::S16 && cell == DataType::S16) {
        return launch<int16_t, int16_t>(stream, operands, outputs, size, lanes);
    }

    if (slot == DataType::S16) {
        return launch<int16_t, int32_t>(stream, operands, outputs, size, lanes);
    }

    if (cell == DataType::S16) {
        return launch<int32_t, int16_t>(stream, operands, outputs, size, lanes);
    }

    return launch<int32_t, int32_t>(stream, operands, outputs, size, lanes);
}

const nanorts::Kernels nanorts_mayari_act_kernels("nanorts_mayari_act", {nanorts::kernel(act_kernel<int16_t, int16_t>), nanorts::kernel(act_kernel<int16_t, int32_t>), nanorts::kernel(act_kernel<int32_t, int16_t>), nanorts::kernel(act_kernel<int32_t, int32_t>)});

}

NANORTS_HANDLER(nanorts_mayari_act, Act,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .RemainingArgs()
                    .RemainingRets()
                    .Attr<std::string_view>("build_id"));
