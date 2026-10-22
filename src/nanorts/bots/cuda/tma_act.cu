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

constexpr int64_t kBytesPerCell = 10;
static_assert(kBytesPerCell == astar::kScratchBytesPerCell + 3);
constexpr int64_t kBytesPerSlot = 15;
constexpr int64_t kBytesPerEntity = 20;

constexpr int32_t kAbsent = -1;
constexpr int32_t kHighest = 2147483647;
constexpr int32_t kLowest = -kHighest - 1;
constexpr int64_t kNoKey = 9223372036854775807LL;
constexpr int64_t kKeyRadix = 4294967296LL;
constexpr int16_t kUnreached = 32767;
constexpr int kDirections = 4;

constexpr int32_t kKindAbsent = -1;
constexpr int32_t kKindNone = 0;
constexpr int32_t kKindMove = 1;
constexpr int32_t kKindHarvest = 2;
constexpr int32_t kKindReturn = 3;
constexpr int32_t kKindProduce = 4;
constexpr int32_t kKindAttack = 5;

constexpr int32_t kTypeBase = 1;
constexpr int32_t kTypeBarracks = 2;
constexpr int32_t kTypeWorker = 3;
constexpr int32_t kTypeLight = 4;
constexpr int32_t kTypeHeavy = 5;
constexpr int32_t kTypeRanged = 6;

constexpr int32_t kFar = 9999;
constexpr int32_t kFillerDuration = 10;
constexpr int32_t kWaitOne = 1;
constexpr int32_t kRecalibrationPeriod = 500;
constexpr int32_t kSpareResources = 5;
constexpr int32_t kBarracksHeld = 2;
constexpr int32_t kBarracksWeight = 10;
constexpr int32_t kUnreachedEnemy = 10;
constexpr int32_t kUnseenEnemyBase = 10;
constexpr int32_t kEnemyReach = 16;
constexpr int32_t kDefaultHarvesters = 2;
constexpr int32_t kHarvestSpan = 5;
constexpr int32_t kLargeHarvesters = 4;
constexpr int32_t kSmallMap = 12;
constexpr int32_t kTinyMap = 10;
constexpr int32_t kLargeMap = 64;
constexpr int32_t kLargeThreshold = 10;
constexpr int32_t kTinyPerMissing = 5;
constexpr int32_t kPerMissing = 8;
constexpr int32_t kTinyPerHeld = 3;
constexpr int32_t kPerHeld = 4;
constexpr int32_t kCloseBases = 4;
constexpr int32_t kRangedPreference = 3;
constexpr int32_t kBarracksWanted = 1;
constexpr int32_t kNear = 2;
constexpr int32_t kCloseCombat = 4;
constexpr int32_t kPredominanceReach = 5;
constexpr int32_t kRangedModifier = 4;
constexpr int32_t kAttackerModifier = 1;

constexpr int kCounted = 6;
constexpr int kWorkers = 0;
constexpr int kLights = 1;
constexpr int kHeavies = 2;
constexpr int kRangedUnits = 3;
constexpr int kBases = 4;
constexpr int kBarracks = 5;
constexpr int kTrained = 3;

constexpr int32_t kNothing = 0;
constexpr int32_t kTrain = 1;
constexpr int32_t kEngage = 2;
constexpr int32_t kHarvest = 3;
constexpr int32_t kBuild = 4;
constexpr int32_t kIdle = 5;
constexpr int32_t kMove = 6;

constexpr int32_t kBaseRole = 0;
constexpr int32_t kBarracksRole = 1;
constexpr int32_t kFighterRole = 2;
constexpr int32_t kBuilderRole = 3;
constexpr int32_t kHarvesterRole = 4;
constexpr int32_t kFreeRole = 5;

constexpr int8_t kFirstPass = 0;
constexpr int8_t kBaseBuilder = 1;
constexpr int8_t kBarracksBuilder = 2;
constexpr int8_t kHarvesters = 3;
constexpr int8_t kFreeWorkers = 4;
constexpr int8_t kIdleHarvesters = 5;
constexpr int8_t kSkipped = 6;

constexpr int kCanHarvest = 0;
constexpr int kIsResource = 1;
constexpr int kIsStockpile = 2;
constexpr int kCanAttack = 3;
constexpr int kAttackRange = 4;
constexpr int kCost = 5;
constexpr int kTableColumns = 6;

constexpr int kAttackDamage = 0;
constexpr int kAttackTime = 1;
constexpr int kActColumns = 2;

constexpr int kCommandKind = 0;
constexpr int kCommandDirection = 1;
constexpr int kCommandNoneDuration = 2;
constexpr int kCommandTargetCell = 3;
constexpr int kCommandProducedType = 4;
constexpr int kCommandRequestOrder = 5;

constexpr uint8_t kPresent = 1u << 0;
constexpr uint8_t kMine = 1u << 1;
constexpr uint8_t kEnemy = 1u << 2;
constexpr uint8_t kResource = 1u << 3;
constexpr uint8_t kStockpile = 1u << 4;
constexpr uint8_t kAvailable = 1u << 5;
constexpr uint8_t kCarrying = 1u << 6;

constexpr uint8_t kWall = 1u << 0;
constexpr uint8_t kOccupied = 1u << 1;
constexpr uint8_t kClaimed = 1u << 2;
constexpr uint8_t kFree = 1u << 3;
constexpr uint8_t kResourceAt = 1u << 4;
constexpr uint8_t kTaken = 1u << 5;
constexpr uint8_t kSearched = 1u << 6;
constexpr uint8_t kBlocks = 1u << 7;

constexpr int kOpenGrid = 0;
constexpr int kAstarGrid = 1;
constexpr int kScoutGrid = 2;

constexpr int kRing = 8;
constexpr int kRadiusOne = 4;
constexpr int kRadiusTwo = 12;
constexpr int kSites = 8;
constexpr int kEdges = 5;

__constant__ int32_t kCountedTypes[kCounted] = {kTypeWorker, kTypeLight, kTypeHeavy, kTypeRanged, kTypeBase, kTypeBarracks};
__constant__ int32_t kTrainedTypes[kTrained] = {kTypeLight, kTypeHeavy, kTypeRanged};
__constant__ int32_t kStepX[kDirections] = {0, 1, 0, -1};
__constant__ int32_t kStepY[kDirections] = {-1, 0, 1, 0};
__constant__ int32_t kRingX[kRadiusTwo] = {-1, 0, 1, 0, -1, -1, 1, 1, -2, 0, 2, 0};
__constant__ int32_t kRingY[kRadiusTwo] = {0, -1, 0, 1, -1, 1, -1, 1, 0, -2, 0, 2};
__constant__ int32_t kSiteX[kEdges][kSites] = {
    {-1, 0, 1, -1, 1, -1, 0, 1},
    {0, 1, 2, 1, 2, 0, 1, 2},
    {0, -1, -2, -1, -2, 0, -1, -2},
    {-1, -1, -1, 0, 0, 1, 1, 1},
    {-1, -1, -1, 0, 0, 1, 1, 1},
};
__constant__ int32_t kSiteY[kEdges][kSites] = {
    {-1, -1, -1, 0, 0, 1, 1, 1},
    {-1, -1, -1, 0, 0, 1, 1, 1},
    {-1, -1, -1, 0, 0, 1, 1, 1},
    {0, 1, 2, 1, 2, 0, 1, 2},
    {0, -1, -2, -1, -2, 0, -1, -2},
};

constexpr int64_t shared_bytes(int64_t cells, int64_t capacity, int64_t entities) {
    return kBytesPerCell * cells + kBytesPerSlot * capacity + kBytesPerEntity * entities;
}

__device__ int32_t clip(int32_t value, int32_t low, int32_t high) {
    return value < low ? low : value > high ? high : value;
}

__device__ int32_t wrapped_sub(int32_t left, int32_t right) {
    return static_cast<int32_t>(static_cast<uint32_t>(left) - static_cast<uint32_t>(right));
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

__device__ Pick least_of(Pick best) {
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        const int64_t key = __shfl_xor_sync(kEveryThread, best.key, offset);
        const int32_t index = __shfl_xor_sync(kEveryThread, best.index, offset);
        offer(best, key, index);
    }

    return best;
}

__device__ int64_t keyed(int32_t score, int32_t birth) {
    return static_cast<int64_t>(score) * kKeyRadix + static_cast<int64_t>(static_cast<uint32_t>(birth));
}

__device__ void or_cell(uint8_t* bits, int32_t cell, uint8_t bit) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(bits + cell);
    atomicOr(reinterpret_cast<uint32_t*>(address & ~uintptr_t{3}), static_cast<uint32_t>(bit) << (8 * (address & 3)));
}

__device__ int32_t scattered(int32_t index, int32_t size) {
    const int32_t wrapped = index < 0 ? index + size : index;
    return wrapped >= 0 && wrapped < size ? wrapped : kAbsent;
}

template <typename Slot, typename Cell>
struct Inputs {
    StateRows<Slot, Cell> state;
    Lanes<int32_t> table;
    Lanes<int32_t> player;
    Lanes<int16_t> hit_points;
    Lanes<int32_t> pending_completes_at;
    Lanes<int32_t> time;
    Lanes<int32_t> act_table;
    Lanes<bool> initialized;
    Lanes<int32_t> main_base;
    Lanes<int32_t> enemy_base;
    Lanes<int32_t> base_to_enemy_base;
    Lanes<int32_t> real_base_to_enemy;
    Lanes<bool> weighted;
};

struct Outputs {
    bool* initialized;
    int32_t* main_base;
    int32_t* enemy_base;
    int32_t* base_to_enemy_base;
    int32_t* real_base_to_enemy;
    bool* weighted;
    int32_t* commands;
};

struct Shared {
    astar::Scratch scratch;
    int32_t* birth;
    int32_t* remaining;
    int16_t* occupant;
    int16_t* cell;
    int16_t* future;
    int16_t* x;
    int16_t* y;
    int16_t* hit_points;
    int16_t* target;
    int16_t* rank;
    int16_t* walk;
    int16_t* bound;
    uint8_t* cells;
    uint8_t* type;
    uint8_t* bits;
    int8_t* kind;
    int8_t* direction;
    int8_t* produced;
    int8_t* none_duration;
    uint8_t* listed;
    int8_t* role;
    int8_t* group;

    __device__ static Shared carve(uint64_t* memory, const Extents& size) {
        const int32_t cells = size.cells;
        const int32_t capacity = size.capacity;
        const int32_t entities = size.entities;
        Shared carved;
        int32_t* words = reinterpret_cast<int32_t*>(memory);
        carved.birth = words;
        carved.remaining = words + entities;
        int16_t* halves = reinterpret_cast<int16_t*>(words + 2 * entities);
        carved.occupant = halves;
        halves += cells;
        carved.cell = halves;
        carved.future = halves + entities;
        carved.x = halves + 2 * entities;
        carved.y = halves + 3 * entities;
        carved.hit_points = halves + 4 * entities;
        halves += 5 * entities;
        carved.target = halves;
        carved.rank = halves + capacity;
        carved.walk = halves + 2 * capacity;
        carved.bound = halves + 3 * capacity;
        halves += 4 * capacity;
        carved.scratch = astar::Scratch::carve(halves, cells);
        uint8_t* bytes = carved.scratch.seen + cells;
        carved.cells = bytes;
        bytes += cells;
        carved.type = bytes;
        carved.bits = bytes + entities;
        bytes += 2 * entities;
        carved.kind = reinterpret_cast<int8_t*>(bytes);
        carved.direction = reinterpret_cast<int8_t*>(bytes + capacity);
        carved.produced = reinterpret_cast<int8_t*>(bytes + 2 * capacity);
        carved.none_duration = reinterpret_cast<int8_t*>(bytes + 3 * capacity);
        carved.listed = bytes + 4 * capacity;
        carved.role = reinterpret_cast<int8_t*>(bytes + 5 * capacity);
        carved.group = reinterpret_cast<int8_t*>(bytes + 6 * capacity);
        return carved;
    }
};

static_assert(2 * 4 + 5 * 2 + 2 == kBytesPerEntity, "the per-entity arrays and their bytes disagree");
static_assert(4 * 2 + 7 == kBytesPerSlot, "the per-slot arrays and their bytes disagree");
static_assert(nanorts::whole_act::kSharedOptInLimit / kBytesPerCell < kUnreached && nanorts::whole_act::kSharedOptInLimit / kBytesPerEntity <= INT16_MAX, "a staged cell or slot outgrew int16");

struct Desire {
    int32_t kind;
    int32_t direction;
    int32_t target_cell;
    int32_t produced_type;
    int32_t none_duration;
};

__device__ Desire nothing() {
    return Desire{kKindAbsent, kAbsent, kAbsent, kAbsent, 0};
}

__device__ Desire directed(int32_t kind, int32_t direction) {
    Desire desire = nothing();
    desire.kind = kind;
    desire.direction = direction;
    return desire;
}

__device__ Desire produced(int32_t direction, int32_t unit_type) {
    Desire desire = directed(kKindProduce, direction);
    desire.produced_type = unit_type;
    return desire;
}

__device__ Desire attack(int32_t cell) {
    Desire desire = nothing();
    desire.kind = kKindAttack;
    desire.target_cell = cell;
    return desire;
}

__device__ Desire wait_one() {
    Desire desire = nothing();
    desire.kind = kKindNone;
    desire.none_duration = kWaitOne;
    return desire;
}

struct Plan {
    int32_t mode;
    int32_t target;
    int32_t base;
    int32_t cell;
    int32_t unit_type;
    bool scouting;
};

__device__ Plan plan(int32_t mode) {
    return Plan{mode, kAbsent, kAbsent, kAbsent, kAbsent, false};
}

__device__ Plan engage(int32_t target, bool scouting = false) {
    Plan engaged = plan(target != kAbsent ? kEngage : kNothing);
    engaged.target = target;
    engaged.scouting = scouting;
    return engaged;
}

struct RolePlan {
    Plan plan;
    bool scouts;
    bool checks_conflicts;
};

struct Least {
    int32_t slot;
    int32_t score;
};

struct Errand {
    int32_t resource;
    int32_t base;
    int32_t enemy;
    bool attacks;
    bool harvests;
    bool returns;
};

struct Memory {
    bool initialized;
    int32_t main_base;
    int32_t enemy_base;
    int32_t base_to_enemy_base;
    int32_t real_base_to_enemy;
    bool weighted;
};

struct Strategy {
    bool defense;
    bool military;
    int32_t harvesters;
    int32_t production[kTrained];
    int32_t threshold;
};

struct Lane {
    Shared s;
    const int32_t* table;
    const int32_t* act_table;
    int32_t lane;
    int32_t width;
    int32_t height;
    int32_t cells;
    int32_t entities;
    int32_t capacity;
    int32_t types;
    int32_t player;
    int32_t start;
    int32_t other_start;
    int32_t time;
    int32_t balance;
    int32_t reserved;
    Memory memory;
    bool own_based;
    int32_t own_base_cell;
    bool enemy_based;
    int32_t enemy_base_cell;
    int32_t own[kCounted];
    int32_t theirs[kCounted];
    Strategy strategy;
    int32_t requests;
    int32_t used;

    __device__ int32_t column(int32_t type, int at) const {
        return table[clip(type, 0, types - 1) * kTableColumns + at];
    }

    __device__ int32_t act_column(int32_t type, int at) const {
        return act_table[clip(type, 0, types - 1) * kActColumns + at];
    }

    __device__ bool can_harvest(int32_t type) const {
        return column(type, kCanHarvest) != 0;
    }

    __device__ bool can_attack(int32_t type) const {
        return column(type, kCanAttack) != 0;
    }

    __device__ int32_t cost(int32_t type) const {
        return column(type, kCost);
    }

    __device__ int32_t reach(int32_t type) const {
        return column(type, kAttackRange);
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
        return abs(x - s.x[e]) + abs(y - s.y[e]);
    }

    __device__ bool on_board(int32_t x, int32_t y) const {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    __device__ int32_t offset_cell(int32_t cell, int32_t dx, int32_t dy) const {
        const int32_t x = x_of(cell) + dx;
        const int32_t y = y_of(cell) + dy;
        return on_board(x, y) && cell != kAbsent ? y * width + x : kAbsent;
    }

    __device__ int32_t step(int32_t cell, int32_t direction) const {
        if (direction < 0 || direction >= kDirections) {
            return kAbsent;
        }

        return offset_cell(cell, kStepX[direction], kStepY[direction]);
    }

    __device__ bool at(uint8_t bit, int32_t cell) const {
        return cell != kAbsent && (s.cells[clip(cell, 0, cells - 1)] & bit) != 0;
    }

    __device__ bool has(int32_t e, uint8_t bit) const {
        return (s.bits[e] & bit) != 0;
    }

    __device__ bool wall_at(int32_t x, int32_t y) const {
        return on_board(x, y) && (s.cells[y * width + x] & kWall) != 0;
    }

    __device__ bool resource_at(int32_t x, int32_t y) const {
        return on_board(x, y) && (s.cells[y * width + x] & kResourceAt) != 0;
    }

    __device__ bool beside_resource(int32_t cell) const {
        const int32_t x = x_of(cell);
        const int32_t y = y_of(cell);
        return resource_at(x, y - 1) || resource_at(x, y + 1) || resource_at(x - 1, y) || resource_at(x + 1, y);
    }

    __device__ bool around_resource(int32_t cell) const {
        const int32_t x = x_of(cell);
        const int32_t y = y_of(cell);
        bool found = false;

        for (int k = 0; k < kRing; ++k) {
            found = found || resource_at(x + kRingX[k], y + kRingY[k]);
        }

        return found;
    }

    __device__ bool chokepoint(int32_t cell) const {
        const int32_t x = x_of(cell);
        const int32_t y = y_of(cell);
        bool walls[kRing];
        int32_t skews[kRing];

        for (int k = 0; k < kRing; ++k) {
            const int32_t nx = x + kRingX[k];
            const int32_t ny = y + kRingY[k];
            walls[k] = wall_at(nx, ny);
            skews[k] = abs(nx - ny);
        }

        bool found = false;
        int32_t count = 0;

        for (int k = 0; k < kRing; ++k) {
            count += walls[k] ? 1 : 0;
            bool apart = false;

            for (int j = 0; j <= k; ++j) {
                apart = apart || (walls[j] && skews[k] + skews[j] > 1);
            }

            found = found || (walls[k] && count > 1 && apart);
        }

        return found;
    }

    __device__ bool open(int32_t cell) const {
        return cell != kAbsent && !at(kOccupied, cell) && !at(kWall, cell) && !beside_resource(clip(cell, 0, cells - 1));
    }

    __device__ bool move_allowed(int32_t cell) const {
        return cell != kAbsent && !at(kWall, cell) && !at(kOccupied, cell) && !at(kClaimed, cell);
    }

    __device__ bool affordable(int32_t price, int32_t spent) const {
        return price == 0 || reserved + spent + price <= balance;
    }

    __device__ int32_t widest() const {
        return max(height, width);
    }

    __device__ bool within_half(int32_t score, int32_t to_base) const {
        const int32_t half = widest() / 2;
        return score < half || to_base < half;
    }

    template <typename Score, typename Among>
    __device__ Least least(Score score, Among among) const {
        Pick best = nothing_picked();

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (among(e)) {
                offer(best, keyed(score(e), s.birth[e]), e);
            }
        }

        best = least_of(best);

        if (best.index == kHighest) {
            return Least{kAbsent, 0};
        }

        return Least{best.index, static_cast<int32_t>(best.key >> 32)};
    }

    template <typename Among>
    __device__ Least nearest(int32_t origin, Among among) const {
        const int32_t ox = x_of(origin);
        const int32_t oy = y_of(origin);
        return least([&](int32_t e) { return distance_to(e, ox, oy); }, among);
    }

    __device__ Least nearest_with(int32_t origin, uint8_t bit) const {
        return nearest(origin, [&](int32_t e) { return has(e, bit); });
    }

    __device__ int32_t score_from(int32_t e, int32_t ox, int32_t oy, int32_t range, int32_t damage) const {
        const int32_t distance = distance_to(e, ox, oy);
        return distance <= range && s.hit_points[e] <= damage ? 0 : distance;
    }

    template <typename Among>
    __device__ Least best_score(int32_t slot, Among among) const {
        const int32_t origin = s.cell[slot];
        const int32_t ox = x_of(origin);
        const int32_t oy = y_of(origin);
        const int32_t type = s.type[slot];
        const int32_t range = reach(type);
        const int32_t damage = act_column(type, kAttackDamage);
        return least([&](int32_t e) { return score_from(e, ox, oy, range, damage); }, among);
    }

    __device__ Least best_enemy(int32_t slot) const {
        return best_score(slot, [&](int32_t e) { return has(e, kEnemy); });
    }

    __device__ bool passable(int32_t cell, int grid) const {
        const uint8_t bits = s.cells[cell];

        if (grid == kOpenGrid) {
            return (bits & kFree) != 0;
        }

        if ((bits & kSearched) != 0) {
            return false;
        }

        return (bits & (grid == kScoutGrid ? kFree | kResourceAt : kFree)) != 0;
    }

    __device__ astar::Path search(int32_t origin, int32_t goal, int32_t range, int grid) {
        for (int32_t c = lane; c < cells; c += kThreads) {
            s.scratch.clear(c, passable(c, grid));
        }

        __syncwarp();
        astar::Path path{0, kAbsent, kAbsent, 0};

        if (lane == 0) {
            path = astar::search(s.scratch, origin, goal, range, width, height);
        }

        __syncwarp();
        path.found = __shfl_sync(kEveryThread, path.found, 0);
        path.step = __shfl_sync(kEveryThread, path.step, 0);
        path.distance = __shfl_sync(kEveryThread, path.distance, 0);
        return path;
    }

    __device__ int32_t remembered(int32_t kept, bool based, int32_t first) const {
        const int32_t chosen = kept == kAbsent ? first : kept;
        return based ? chosen : kAbsent;
    }

    __device__ void remember_bases() {
        memory.main_base = remembered(memory.main_base, own_based, own_base_cell);
        memory.enemy_base = remembered(memory.enemy_base, enemy_based, enemy_base_cell);
    }

    __device__ int16_t level(int32_t x, int32_t y) const {
        return on_board(x, y) ? s.scratch.parent[y * width + x] : kUnreached;
    }

    __device__ int32_t real_distance(int32_t main) {
        bool any_enemy = false;

        for (int32_t local = lane; local < capacity; local += kThreads) {
            any_enemy = any_enemy || has(other_start + local, kEnemy);
        }

        if (!warp_any(any_enemy)) {
            return kAbsent;
        }

        int16_t* levels = s.scratch.parent;

        for (int32_t c = lane; c < cells; c += kThreads) {
            levels[c] = c == main ? 0 : kUnreached;
        }

        __syncwarp();

        for (int32_t depth = 0;; ++depth) {
            bool grew = false;

            for (int32_t c = lane; c < cells; c += kThreads) {
                if (levels[c] != kUnreached || (s.cells[c] & kFree) == 0) {
                    continue;
                }

                const int32_t x = c % width;
                const int32_t y = c / width;

                if (level(x, y - 1) == depth || level(x, y + 1) == depth || level(x - 1, y) == depth || level(x + 1, y) == depth) {
                    levels[c] = static_cast<int16_t>(depth + 1);
                    grew = true;
                }
            }

            __syncwarp();

            if (!warp_any(grew)) {
                break;
            }
        }

        for (int32_t local = lane; local < capacity; local += kThreads) {
            const int32_t slot = other_start + local;
            int16_t bound = kUnreached;

            if (has(slot, kEnemy)) {
                const int32_t c = clip(s.cell[slot], 0, cells - 1);
                const int32_t x = c % width;
                const int32_t y = c / width;
                bound = min(min(level(x, y), min(level(x, y - 1), level(x, y + 1))), min(level(x - 1, y), level(x + 1, y)));
            }

            s.bound[local] = bound;
        }

        __syncwarp();
        int32_t best = kHighest;

        while (true) {
            Pick next = nothing_picked();

            for (int32_t local = lane; local < capacity; local += kThreads) {
                if (s.bound[local] != kUnreached) {
                    offer(next, s.bound[local], local);
                }
            }

            next = least_of(next);

            if (next.index == kHighest || next.key >= best) {
                break;
            }

            const astar::Path path = search(main, s.cell[other_start + next.index], 1, kOpenGrid);

            if (path.found != 0) {
                best = min(best, path.distance);
            }

            if (lane == 0) {
                s.bound[next.index] = kUnreached;
            }

            __syncwarp();
        }

        return best == kHighest ? kAbsent : best;
    }

    __device__ int32_t harvesters(bool based, int32_t base_to_enemy_base, int32_t to_resources, int32_t to_enemy) const {
        if (!based) {
            return 0;
        }

        if (widest() < kSmallMap || base_to_enemy_base < kEnemyReach) {
            return own[kBases];
        }

        if (to_resources > to_enemy) {
            return max(kDefaultHarvesters, own[kBases]);
        }

        if (to_resources / kHarvestSpan > kDefaultHarvesters) {
            return max(to_resources / kHarvestSpan, own[kBases]);
        }

        if (widest() >= kLargeMap) {
            return min(to_resources - 1, kLargeHarvesters);
        }

        return kDefaultHarvesters;
    }

    __device__ void production(bool prefers_ranged) {
        int32_t highest = 0;

        for (int k = 0; k < kTrained; ++k) {
            highest = max(highest, own[kLights + k]);
        }

        for (int k = 0; k < kTrained; ++k) {
            strategy.production[k] = highest == 0 ? (k == 1 ? 1 : 0) : (own[kLights + k] < highest ? 1 : 0);
        }

        if (prefers_ranged) {
            strategy.production[2] += kRangedPreference;
        }

        int32_t countered = kWorkers;
        int32_t most = 0;

        for (int index = kLights; index <= kRangedUnits; ++index) {
            if (theirs[index] > most && theirs[index] >= own[index]) {
                countered = index;
                most = theirs[index];
            }
        }

        if (countered == kLights || countered == kHeavies) {
            strategy.production[1] += 1;
        }

        if (countered == kRangedUnits) {
            strategy.production[0] += 1;
        }
    }

    __device__ int32_t threshold(int32_t real) const {
        const int32_t short_by = kBarracksWanted - own[kBarracks];
        const int32_t per_missing = widest() <= kTinyMap ? kTinyPerMissing : kPerMissing;
        const int32_t per_held = widest() <= kTinyMap ? kTinyPerHeld : kPerHeld;

        if (widest() >= kLargeMap && real != kAbsent) {
            return kLargeThreshold;
        }

        return short_by > 0 ? per_missing * short_by : per_held * own[kBarracks];
    }

    __device__ void macro() {
        remember_bases();
        const bool recalibrates = time % kRecalibrationPeriod == 0 && time != 0;
        const bool weighted = recalibrates ? own[kWorkers] != 0 || own[kBases] != 0 : memory.weighted;
        const int32_t main = memory.main_base;
        const int32_t enemy_base = memory.enemy_base;
        const bool based = main != kAbsent;
        const bool measures = based && memory.real_base_to_enemy <= 0;
        const int32_t real = measures ? real_distance(max(main, 0)) : memory.real_base_to_enemy;
        const int32_t base_to_enemy_base = based && enemy_base != kAbsent ? manhattan(main, enemy_base) : memory.base_to_enemy_base;
        const int32_t mx = x_of(main);
        const int32_t my = y_of(main);
        int32_t to_resources = kHighest;
        int32_t to_enemy_base = kHighest;
        bool resources = false;

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (has(e, kResource)) {
                resources = true;
                to_resources = min(to_resources, distance_to(e, mx, my));
            }

            if (has(e, kMine) && enemy_base != kAbsent) {
                to_enemy_base = min(to_enemy_base, manhattan(s.cell[e], enemy_base));
            }
        }

        to_resources = based && warp_any(resources) ? warp_min(to_resources) : kFar;
        to_enemy_base = warp_min(to_enemy_base);
        to_enemy_base = enemy_base != kAbsent && to_enemy_base != kHighest ? to_enemy_base : kFar;
        const Least nearest_enemy = nearest_with(main, kEnemy);
        const int32_t to_enemy = based && nearest_enemy.slot != kAbsent ? nearest_enemy.score : kFar;
        const int32_t weight = weighted ? 1 : 0;
        int32_t lead = 0;

        for (int k = kWorkers; k <= kRangedUnits; ++k) {
            lead += own[k] - theirs[k];
        }

        const int32_t approach = to_enemy == kFar ? 0 : (to_enemy_base == kFar ? kUnseenEnemyBase : to_enemy - to_enemy_base);
        const bool attacks = weight * (lead + approach) >= 0;
        const int32_t spare = balance - kSpareResources;
        const int32_t holds_barracks = own[kBarracks] > 0 ? kBarracksHeld : 0;
        const int32_t nearness = real == kAbsent ? kUnreachedEnemy : min(to_enemy, height + width) - kEnemyReach;
        const bool military = weight * (spare + kBarracksWeight * holds_barracks + nearness) >= 0;
        bool idle_barracks = false;

        for (int32_t e = lane; e < entities; e += kThreads) {
            idle_barracks = idle_barracks || ((s.bits[e] & (kMine | kAvailable)) == (kMine | kAvailable) && s.type[e] == kTypeBarracks);
        }

        const bool consulted = military && warp_any(idle_barracks);
        bool cut_off = false;

        if (based && nearest_enemy.slot != kAbsent) {
            cut_off = !consulted || search(max(main, 0), s.cell[nearest_enemy.slot], 1, kOpenGrid).found == 0;
        }

        const bool prefers_ranged = real == kAbsent || (base_to_enemy_base <= kCloseBases && own[kRangedUnits] == 0) || cut_off;
        strategy.defense = !attacks;
        strategy.military = military;
        strategy.harvesters = harvesters(based, base_to_enemy_base, to_resources, to_enemy);
        production(prefers_ranged);
        strategy.threshold = threshold(real);
        memory.base_to_enemy_base = base_to_enemy_base;
        memory.real_base_to_enemy = real;
        memory.weighted = weighted;
    }

    __device__ Errand errand(int32_t slot) const {
        const int32_t origin = s.cell[slot];
        const int32_t ox = x_of(origin);
        const int32_t oy = y_of(origin);
        Pick resource = nothing_picked();
        Pick base = nothing_picked();
        Pick enemy = nothing_picked();

        for (int32_t e = lane; e < entities; e += kThreads) {
            const uint8_t bits = s.bits[e];

            if ((bits & (kResource | kStockpile | kEnemy)) == 0) {
                continue;
            }

            const int64_t key = keyed(distance_to(e, ox, oy), s.birth[e]);

            if ((bits & kResource) != 0) {
                offer(resource, key, e);
            }

            if ((bits & kStockpile) != 0) {
                offer(base, key, e);
            }

            if ((bits & kEnemy) != 0) {
                offer(enemy, key, e);
            }
        }

        resource = least_of(resource);
        base = least_of(base);
        enemy = least_of(enemy);
        Errand found;
        found.resource = resource.index == kHighest ? kAbsent : resource.index;
        found.base = base.index == kHighest ? kAbsent : base.index;
        found.enemy = enemy.index == kHighest ? kAbsent : enemy.index;
        const int32_t to_base = found.base == kAbsent ? 0 : static_cast<int32_t>(base.key >> 32);
        const int32_t to_enemy = found.enemy == kAbsent ? 0 : static_cast<int32_t>(enemy.key >> 32);
        found.attacks = found.enemy != kAbsent && to_enemy <= kNear && to_base <= kNear;
        found.harvests = !found.attacks && found.resource != kAbsent && found.base != kAbsent;
        found.returns = !found.attacks && !found.harvests && found.base != kAbsent && has(slot, kCarrying);
        return found;
    }

    __device__ int32_t barracks_type() const {
        int32_t preferred = kTypeHeavy;
        int32_t most = 0;

        for (int k = 0; k < kTrained; ++k) {
            if (strategy.production[k] > most) {
                preferred = kTrainedTypes[k];
                most = strategy.production[k];
            }
        }

        return strategy.military ? preferred : kTypeLight;
    }

    __device__ void entries(int32_t& count, int32_t& first_pass_end) {
        int32_t workers = 0;

        for (int32_t local = lane; local < capacity; local += kThreads) {
            const int32_t slot = start + local;
            const bool worker = has(slot, kMine) && can_harvest(s.type[slot]);
            s.group[local] = worker ? 1 : 0;
            workers += worker ? 1 : 0;
        }

        workers = warp_sum(workers);
        __syncwarp();
        const bool builds_base = own[kBases] == 0 && workers > 0 && balance >= cost(kTypeBase);
        const int32_t first_harvester = builds_base ? 1 : 0;
        const int32_t harvesting = max(min(strategy.harvesters, workers - first_harvester), 0);
        const bool builds_barracks = strategy.military && harvesting >= 1 && own[kBarracks] < kBarracksWanted && balance > cost(kTypeBarracks);
        const int32_t first_free = first_harvester + harvesting;
        const bool from_free = builds_barracks && harvesting == 1 && workers > first_free;
        const int32_t barracks_rank = from_free ? first_free : first_harvester;
        int8_t groups[(256 + kThreads - 1) / kThreads];
        int8_t roles[(256 + kThreads - 1) / kThreads];
        bool harvesting_here[(256 + kThreads - 1) / kThreads];

        for (int32_t local = lane, k = 0; local < capacity; local += kThreads, ++k) {
            const int32_t slot = start + local;
            const bool owned = has(slot, kMine);
            const int32_t type = s.type[slot];
            const bool worker = s.group[local] != 0;
            const bool fighter = owned && can_attack(type) && !can_harvest(type);
            int32_t rank = 0;

            if (worker) {
                const int32_t born = s.birth[slot];

                for (int32_t other = 0; other < capacity; ++other) {
                    rank += s.group[other] != 0 && s.birth[start + other] < born ? 1 : 0;
                }
            }

            const bool base_builder = worker && builds_base && rank == 0;
            const bool barracks_builder = worker && builds_barracks && rank == barracks_rank;
            const bool harvester = worker && rank >= first_harvester && rank < first_free && !barracks_builder;
            const bool free = worker && rank >= first_free && !barracks_builder;
            int8_t group = kSkipped;

            if (fighter || type == kTypeBase || type == kTypeBarracks) {
                group = kFirstPass;
            } else if (base_builder) {
                group = kBaseBuilder;
            } else if (barracks_builder) {
                group = kBarracksBuilder;
            } else if (harvester) {
                group = kHarvesters;
            } else if (free) {
                group = kFreeWorkers;
            }

            int8_t role = kFreeRole;

            if (type == kTypeBase) {
                role = kBaseRole;
            } else if (type == kTypeBarracks) {
                role = kBarracksRole;
            } else if (fighter) {
                role = kFighterRole;
            } else if (base_builder || barracks_builder) {
                role = kBuilderRole;
            } else if (harvester) {
                role = kHarvesterRole;
            }

            groups[k] = group;
            roles[k] = role;
            harvesting_here[k] = harvester && group == kHarvesters;
        }

        __syncwarp();

        for (int32_t local = lane, k = 0; local < capacity; local += kThreads, ++k) {
            s.group[local] = groups[k];
            s.role[local] = roles[k];
        }

        __syncwarp();

        for (int32_t base = 0, k = 0; base < capacity; base += kThreads, ++k) {
            const int32_t local = base + lane;
            unsigned pending = __ballot_sync(kEveryThread, local < capacity && harvesting_here[k]);

            while (pending != 0) {
                const int32_t holder = __ffs(pending) - 1;
                pending &= pending - 1;
                const int32_t chosen = base + holder;
                const Errand found = errand(start + chosen);

                if (!(found.attacks || found.harvests || found.returns)) {
                    if (lane == 0) {
                        s.group[chosen] = kIdleHarvesters;
                        s.role[chosen] = kFreeRole;
                    }

                    __syncwarp();
                }
            }
        }

        __syncwarp();

        for (int32_t local = lane; local < capacity; local += kThreads) {
            const int32_t slot = start + local;

            if (!has(slot, kMine) || !has(slot, kAvailable)) {
                s.group[local] = kSkipped;
            }
        }

        __syncwarp();
        int32_t visiting = 0;
        int32_t first = 0;

        for (int32_t local = lane; local < capacity; local += kThreads) {
            const int32_t group = s.group[local];

            if (group == kSkipped) {
                continue;
            }

            const int32_t born = s.birth[start + local];
            int32_t place = 0;

            for (int32_t other = 0; other < capacity; ++other) {
                const int32_t other_group = s.group[other];
                place += other_group != kSkipped && (other_group < group || (other_group == group && s.birth[start + other] < born)) ? 1 : 0;
            }

            s.walk[place] = static_cast<int16_t>(local);
            ++visiting;
            first += group == kFirstPass ? 1 : 0;
        }

        count = warp_sum(visiting);
        first_pass_end = warp_sum(first);
        __syncwarp();
    }

    __device__ bool predominant(int32_t slot) const {
        const int32_t sx = s.x[slot];
        const int32_t sy = s.y[slot];
        int32_t allies = 0;
        int32_t enemies = 0;

        for (int32_t e = lane; e < entities; e += kThreads) {
            const uint8_t bits = s.bits[e];

            if ((bits & kPresent) == 0 || abs(s.x[e] - sx) > kPredominanceReach || abs(s.y[e] - sy) > kPredominanceReach) {
                continue;
            }

            allies += (bits & kMine) != 0 ? 1 : 0;
            enemies += (bits & kEnemy) != 0 ? 1 : 0;
        }

        return warp_sum(enemies) <= warp_sum(allies);
    }

    __device__ bool stalls(int32_t slot) const {
        return chokepoint(s.cell[slot]) && !predominant(slot);
    }

    __device__ Desire walk(int32_t start_cell, int32_t direction) const {
        return move_allowed(step(start_cell, direction)) ? directed(kKindMove, direction) : nothing();
    }

    __device__ int32_t adjacent_direction(int32_t cell, int32_t goal) const {
        if (goal == kAbsent) {
            return kAbsent;
        }

        const int32_t dx = x_of(goal) - x_of(cell);
        const int32_t dy = y_of(goal) - y_of(cell);

        for (int d = 0; d < kDirections; ++d) {
            if (dx == kStepX[d] && dy == kStepY[d]) {
                return d;
            }
        }

        return kAbsent;
    }

    __device__ Desire idle(int32_t slot) const {
        const int32_t type = s.type[slot];
        const int32_t range = reach(type);
        const int32_t origin = s.cell[slot];
        const int32_t ox = x_of(origin);
        const int32_t oy = y_of(origin);
        const Least target = least([](int32_t) { return 0; }, [&](int32_t e) {
            const int32_t dx = ox - s.x[e];
            const int32_t dy = oy - s.y[e];
            return has(e, kEnemy) && dx * dx + dy * dy <= range * range;
        });
        return can_attack(type) && target.slot != kAbsent ? attack(s.cell[target.slot]) : nothing();
    }

    __device__ Desire train(int32_t slot, int32_t unit_type) const {
        const int32_t origin = s.cell[slot];
        const int32_t known = clip(unit_type, 0, types - 1);
        const bool to_resources = can_harvest(known) && own[kWorkers] < strategy.harvesters;
        const uint8_t toward = to_resources ? kResource : kEnemy;
        int32_t cell[kDirections];
        int32_t cx[kDirections];
        int32_t cy[kDirections];
        int32_t nearest[kDirections];

        for (int d = 0; d < kDirections; ++d) {
            cell[d] = step(origin, d);
            cx[d] = x_of(cell[d]);
            cy[d] = y_of(cell[d]);
            nearest[d] = kHighest;
        }

        bool any_toward = false;

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (!has(e, toward)) {
                continue;
            }

            any_toward = true;

            for (int d = 0; d < kDirections; ++d) {
                nearest[d] = min(nearest[d], distance_to(e, cx[d], cy[d]));
            }
        }

        const bool found = warp_any(any_toward);
        int32_t best = 0;
        int32_t best_value = kLowest;
        bool any_open = false;

        for (int d = 0; d < kDirections; ++d) {
            const int32_t distance = found ? warp_min(nearest[d]) : 0;
            const bool open_cell = at(kFree, cell[d]);
            const int32_t value = open_cell ? -distance : kLowest;
            any_open = any_open || open_cell;

            if (d == 0 || value > best_value) {
                best = d;
                best_value = value;
            }
        }

        return any_open && affordable(cost(known), 0) ? produced(best, unit_type) : nothing();
    }

    __device__ Desire approach(int32_t slot, int32_t target) {
        const int32_t origin = s.cell[slot];
        const int32_t target_cell = s.cell[target];
        const int32_t target_type = s.type[target];
        const int32_t modifier = can_attack(target_type) ? (target_type == kTypeRanged ? kRangedModifier : kAttackerModifier) : 0;
        const int32_t current = manhattan(target_cell, origin) - modifier;
        int32_t candidate = kAbsent;
        int32_t key = kHighest;

        if (lane < kRadiusTwo) {
            candidate = offset_cell(origin, kRingX[lane], kRingY[lane]);
            const int32_t closer = manhattan(target_cell, candidate) - modifier;
            const bool qualifies = open(candidate) && closer < current && closer > 0;
            key = qualifies ? closer : kHighest;
        }

        int32_t order = 0;

        for (int other = 0; other < kRadiusTwo; ++other) {
            const int32_t other_key = __shfl_sync(kEveryThread, key, other);
            order += other_key < key || (other_key == key && other < lane) ? 1 : 0;
        }

        const int32_t count = __popc(__ballot_sync(kEveryThread, lane < kRadiusTwo && key != kHighest));
        int32_t direction = kAbsent;

        for (int32_t index = 0; index < count && direction == kAbsent; ++index) {
            const int32_t holder = __ffs(__ballot_sync(kEveryThread, lane < kRadiusTwo && order == index)) - 1;
            const int32_t goal = __shfl_sync(kEveryThread, candidate, holder);
            direction = search(origin, goal, 0, kAstarGrid).step;
        }

        return walk(origin, direction);
    }

    __device__ Desire carry_out(int32_t slot, const Plan& p, bool first_phase) {
        if (!first_phase && p.mode == kIdle) {
            return idle(slot);
        }

        const int32_t origin = s.cell[slot];
        const int32_t type = s.type[slot];
        const int32_t range = reach(type);
        const int32_t target = max(p.target, 0);
        const int32_t target_cell = s.cell[target];
        const bool engages = p.mode == kEngage;
        const bool escapes = s.remaining[target] <= act_column(type, kAttackTime);
        const bool strikes = engages && squared(origin, target_cell) <= range * range && !escapes;
        const bool waits = engages && !strikes && squared(origin, s.future[target]) <= range * range;
        const bool chases = engages && !strikes && !waits;

        if (strikes) {
            return attack(target_cell);
        }

        if (waits) {
            return wait_one();
        }

        if (chases) {
            const int grid = !first_phase && p.scouting ? kScoutGrid : kAstarGrid;
            const astar::Path path = search(origin, target_cell, range, grid);

            if (path.step == kAbsent) {
                return approach(slot, target);
            }

            if (!first_phase && p.scouting && at(kResourceAt, step(origin, path.step))) {
                return stalls(slot) ? nothing() : directed(kKindHarvest, path.step);
            }

            return walk(origin, path.step);
        }

        if (first_phase) {
            if (p.mode == kMove) {
                const astar::Path path = search(origin, p.cell, 0, kAstarGrid);
                return path.step != kAbsent ? walk(origin, path.step) : nothing();
            }

            return p.mode == kTrain ? train(slot, p.unit_type) : nothing();
        }

        const bool carrying = has(slot, kCarrying);
        const int32_t errand_slot = carrying ? p.base : p.target;
        const bool harvests = p.mode == kHarvest && errand_slot != kAbsent;
        const bool builds = p.mode == kBuild;

        if (!harvests && !builds) {
            return nothing();
        }

        const int32_t goal = harvests ? s.cell[errand_slot] : p.cell;
        const astar::Path path = search(origin, goal, 1, kAstarGrid);

        if (path.step != kAbsent) {
            return walk(origin, path.step);
        }

        const int32_t beside = adjacent_direction(origin, goal);

        if (harvests) {
            return beside != kAbsent ? directed(carrying ? kKindReturn : kKindHarvest, beside) : nothing();
        }

        const int32_t built = clip(p.unit_type, 0, types - 1);
        const bool allowed = beside != kAbsent && !at(kClaimed, step(origin, beside)) && affordable(cost(built), 0);
        return allowed ? produced(beside, p.unit_type) : nothing();
    }

    __device__ Plan defense_plan(int32_t slot) const {
        const Least target = best_enemy(slot);
        const Least to_base = nearest(s.cell[slot], [&](int32_t e) { return has(e, kMine) && s.type[e] == kTypeBase; });
        return engage(within_half(target.score, to_base.score) ? target.slot : kAbsent);
    }

    __device__ Plan ranged_plan(int32_t slot) const {
        const int32_t origin = s.cell[slot];
        const Least closest = nearest_with(origin, kEnemy);

        if (closest.slot == kAbsent) {
            return plan(kNothing);
        }

        const int32_t best = best_enemy(slot).slot;
        const bool one_shot = s.hit_points[max(best, 0)] <= act_column(s.type[slot], kAttackDamage);
        const bool near = closest.score <= kNear;
        const bool stepping_in = manhattan(origin, s.future[closest.slot]) == 1;
        const int32_t closest_cell = s.cell[closest.slot];
        int32_t first_away = kAbsent;

        for (int k = 0; k < kRadiusOne; ++k) {
            const int32_t cell = offset_cell(origin, kRingX[k], kRingY[k]);

            if (first_away == kAbsent && open(cell) && manhattan(closest_cell, cell) > 1) {
                first_away = cell;
            }
        }

        if (near && !one_shot && stepping_in && first_away != kAbsent) {
            Plan retreat = plan(kMove);
            retreat.cell = first_away;
            return retreat;
        }

        return engage(best);
    }

    __device__ Plan aggressive_plan(int32_t slot) const {
        const int32_t origin = s.cell[slot];
        const int32_t ox = x_of(origin);
        const int32_t oy = y_of(origin);
        const int32_t type = s.type[slot];
        const int32_t range = reach(type);
        const int32_t damage = act_column(type, kAttackDamage);
        Pick closest = nothing_picked();
        Pick favourite = nothing_picked();

        for (int32_t e = lane; e < entities; e += kThreads) {
            if (!has(e, kEnemy)) {
                continue;
            }

            const int64_t key = keyed(score_from(e, ox, oy, range, damage), s.birth[e]);
            offer(closest, key, e);
            const int32_t other = s.type[e];

            if ((type == kTypeHeavy && (other == kTypeBarracks || other == kTypeBase)) || (type == kTypeLight && other == kTypeRanged)) {
                offer(favourite, key, e);
            }
        }

        closest = least_of(closest);
        favourite = least_of(favourite);
        const int32_t closest_slot = closest.index == kHighest ? kAbsent : closest.index;
        const int32_t closest_score = closest.index == kHighest ? 0 : static_cast<int32_t>(closest.key >> 32);
        const int32_t favourite_slot = favourite.index == kHighest ? kAbsent : favourite.index;
        const bool keeps = favourite_slot == kAbsent || favourite_slot == closest_slot || closest_score < kCloseCombat;
        return engage(keeps ? closest_slot : favourite_slot);
    }

    __device__ Plan fighter_plan(int32_t slot) const {
        if (strategy.defense) {
            return defense_plan(slot);
        }

        if (!strategy.military) {
            return engage(best_enemy(slot).slot);
        }

        const int32_t type = s.type[slot];

        if (type == kTypeRanged) {
            return ranged_plan(slot);
        }

        if (type == kTypeHeavy || type == kTypeLight) {
            return aggressive_plan(slot);
        }

        return plan(kNothing);
    }

    __device__ int32_t site(int32_t anchor, int k) const {
        const int32_t x = x_of(anchor);
        const int32_t y = y_of(anchor);
        const int edge = x == 0 ? 1 : x == width - 1 ? 2 : y == 0 ? 3 : y == height - 1 ? 4 : 0;
        const int32_t cell = offset_cell(anchor, kSiteX[edge][k], kSiteY[edge][k]);
        return open(cell) ? cell : kAbsent;
    }

    __device__ int32_t chosen_site(const int32_t (&scores)[kSites], const int32_t (&sites)[kSites], bool zero_best_yields) const {
        int32_t best = 0;
        int32_t chosen = 0;

        for (int k = 0; k < kSites; ++k) {
            bool better = scores[k] > best;

            if (zero_best_yields) {
                better = better || best == 0;
            }

            better = better && sites[k] != kAbsent;

            if (better) {
                best = scores[k];
                chosen = sites[k];
            }
        }

        return chosen;
    }

    __device__ int32_t base_site(int32_t builder, const int32_t (&sites)[kSites]) const {
        int32_t to_resource[kSites];
        int32_t to_enemy[kSites];
        int32_t sx[kSites];
        int32_t sy[kSites];

        for (int k = 0; k < kSites; ++k) {
            to_resource[k] = kHighest;
            to_enemy[k] = 0;
            sx[k] = x_of(sites[k]);
            sy[k] = y_of(sites[k]);
        }

        for (int32_t e = lane; e < entities; e += kThreads) {
            const uint8_t bits = s.bits[e];

            if ((bits & (kResource | kEnemy)) == 0) {
                continue;
            }

            for (int k = 0; k < kSites; ++k) {
                const int32_t distance = distance_to(e, sx[k], sy[k]);

                if ((bits & kResource) != 0 && distance > 1) {
                    to_resource[k] = min(to_resource[k], distance);
                }

                if ((bits & kEnemy) != 0) {
                    to_enemy[k] = max(to_enemy[k], distance);
                }
            }
        }

        int32_t scores[kSites];

        for (int k = 0; k < kSites; ++k) {
            const int32_t nearest_resource = warp_min(to_resource[k]);
            const int32_t resource = nearest_resource == kHighest ? 0 : nearest_resource;
            scores[k] = warp_max(to_enemy[k]) - resource - manhattan(builder, sites[k]);
        }

        return chosen_site(scores, sites, false);
    }

    __device__ int32_t barracks_site(const int32_t (&sites)[kSites]) const {
        int32_t to_base[kSites];
        int32_t far_enemy[kSites];
        int32_t near_enemy[kSites];
        int32_t sx[kSites];
        int32_t sy[kSites];
        bool enemies = false;

        for (int k = 0; k < kSites; ++k) {
            to_base[k] = 0;
            far_enemy[k] = 0;
            near_enemy[k] = kHighest;
            sx[k] = x_of(sites[k]);
            sy[k] = y_of(sites[k]);
        }

        for (int32_t e = lane; e < entities; e += kThreads) {
            const uint8_t bits = s.bits[e];
            const bool own_base = (bits & kMine) != 0 && s.type[e] == kTypeBase;
            const bool enemy = (bits & kEnemy) != 0;

            if (!own_base && !enemy) {
                continue;
            }

            enemies = enemies || enemy;

            for (int k = 0; k < kSites; ++k) {
                const int32_t distance = distance_to(e, sx[k], sy[k]);

                if (own_base && distance > 1) {
                    to_base[k] = max(to_base[k], distance);
                }

                if (enemy) {
                    far_enemy[k] = max(far_enemy[k], distance);
                    near_enemy[k] = min(near_enemy[k], distance);
                }
            }
        }

        const bool any_enemy = warp_any(enemies);
        int32_t scores[kSites];

        for (int k = 0; k < kSites; ++k) {
            const int32_t base = warp_max(to_base[k]);
            int32_t score = 0;

            if (width < kLargeMap) {
                score = warp_max(far_enemy[k]) + base;
            } else {
                const int32_t nearest_enemy = warp_min(near_enemy[k]);
                score = base - (any_enemy ? nearest_enemy : 0);
            }

            const bool bad_spot = sites[k] != kAbsent && (chokepoint(clip(sites[k], 0, cells - 1)) || around_resource(clip(sites[k], 0, cells - 1)));
            scores[k] = bad_spot ? 0 : score;
        }

        return chosen_site(scores, sites, true);
    }

    __device__ RolePlan builder_plan(int32_t slot) const {
        const int32_t origin = s.cell[slot];
        const bool builds_base = own[kBases] == 0 && balance >= cost(kTypeBase);
        const bool builds_barracks = strategy.military && own[kBarracks] < kBarracksWanted && balance >= cost(kTypeBarracks);
        int32_t sites[kSites];
        bool any_site = false;

        for (int k = 0; k < kSites; ++k) {
            sites[k] = site(origin, k);
            any_site = any_site || sites[k] != kAbsent;
        }

        if (builds_base && any_site) {
            Plan build = plan(kBuild);
            build.cell = base_site(origin, sites);
            build.unit_type = kTypeBase;
            return RolePlan{build, false, true};
        }

        const Least closest_base = nearest(origin, [&](int32_t e) { return has(e, kMine) && s.type[e] == kTypeBase; });
        const int32_t anchor = closest_base.slot == kAbsent ? origin : s.cell[closest_base.slot];
        any_site = false;

        for (int k = 0; k < kSites; ++k) {
            sites[k] = site(anchor, k);
            any_site = any_site || sites[k] != kAbsent;
        }

        if (!builds_base && builds_barracks && any_site) {
            Plan build = plan(kBuild);
            build.cell = barracks_site(sites);
            build.unit_type = kTypeBarracks;
            return RolePlan{build, false, true};
        }

        const int32_t resource = nearest_with(origin, kResource).slot;
        const int32_t base = nearest_with(origin, kStockpile).slot;
        Plan worker = plan(kHarvest);
        worker.target = resource;
        worker.base = base;
        return RolePlan{worker, resource == kAbsent && base == kAbsent, true};
    }

    __device__ bool between(int32_t cell, int32_t first, int32_t last) const {
        const int32_t x = x_of(cell);
        const int32_t y = y_of(cell);
        const int32_t fx = x_of(first);
        const int32_t fy = y_of(first);
        const int32_t lx = x_of(last);
        const int32_t ly = y_of(last);
        return min(fx, lx) <= x && x <= max(fx, lx) && min(fy, ly) <= y && y <= max(fy, ly);
    }

    __device__ RolePlan harvester_plan(int32_t slot) const {
        const Errand found = errand(slot);

        if (found.attacks) {
            return RolePlan{engage(found.enemy), false, false};
        }

        Plan harvest = plan(found.harvests || found.returns ? kHarvest : kNothing);
        harvest.target = found.harvests ? found.resource : kAbsent;
        harvest.base = found.base;
        const bool blocked = found.harvests && found.enemy != kAbsent && between(s.cell[found.enemy], s.cell[slot], s.cell[found.resource]) && !has(slot, kCarrying);
        return RolePlan{harvest, blocked, true};
    }

    __device__ Plan scout_plan(int32_t slot) {
        const int32_t origin = s.cell[slot];
        const Least enemy = best_enemy(slot);
        const Least base = nearest_with(origin, kStockpile);
        const bool sighted = enemy.slot != kAbsent && (!strategy.defense || within_half(enemy.score, base.score));

        if (!sighted) {
            return plan(kNothing);
        }

        if (search(origin, s.cell[enemy.slot], 1, kAstarGrid).found != 0) {
            return stalls(slot) ? plan(kIdle) : engage(enemy.slot);
        }

        if (!has(slot, kCarrying)) {
            return engage(enemy.slot, true);
        }

        if (base.slot == kAbsent) {
            return plan(kNothing);
        }

        if (search(origin, s.cell[base.slot], 1, kAstarGrid).found != 0) {
            Plan returning = plan(kHarvest);
            returning.base = base.slot;
            return returning;
        }

        return engage(enemy.slot, true);
    }

    __device__ bool beside_listed_move(int32_t slot) const {
        const int32_t origin = s.cell[slot];
        bool listed = false;

        for (int k = 0; k < kRadiusOne; ++k) {
            const int32_t neighbour = offset_cell(origin, kRingX[k], kRingY[k]);
            const int32_t occupant = neighbour != kAbsent ? s.occupant[clip(neighbour, 0, cells - 1)] : kAbsent;
            const int32_t local = occupant - start;
            const bool own_unit = local >= 0 && local < capacity && occupant != kAbsent && has(clip(occupant, 0, entities - 1), kMine);
            listed = listed || (own_unit && s.listed[clip(local, 0, capacity - 1)] != 0);
        }

        return listed;
    }

    __device__ void accept(int32_t slot, const Desire& desire, bool first_phase) {
        const bool claims = desire.kind == kKindMove || desire.kind == kKindProduce;
        const int32_t cell = claims ? step(s.cell[slot], desire.direction) : kAbsent;
        const int32_t price = desire.kind == kKindProduce ? cost(clip(desire.produced_type, 0, types - 1)) : 0;
        const bool kept = desire.kind != kKindAbsent && !at(kClaimed | kTaken, cell) && affordable(price, used);

        if (!kept) {
            return;
        }

        if (lane == 0) {
            const int32_t local = slot - start;
            s.kind[local] = static_cast<int8_t>(desire.kind);
            s.direction[local] = static_cast<int8_t>(desire.direction);
            s.target[local] = static_cast<int16_t>(desire.target_cell);
            s.produced[local] = static_cast<int8_t>(desire.produced_type);
            s.none_duration[local] = static_cast<int8_t>(desire.none_duration);
            s.rank[local] = static_cast<int16_t>(requests);

            if (cell != kAbsent) {
                s.cells[cell] |= first_phase ? kTaken | kSearched : kTaken;
            }
        }

        __syncwarp();
        ++requests;
        used += price;
    }

    __device__ void visit_first(int32_t slot, int32_t role) {
        Plan chosen;
        const int32_t which = clip(role, kBaseRole, kFighterRole);

        if (which == kBaseRole) {
            const bool keeps_training = own[kWorkers] < strategy.harvesters || balance > strategy.threshold;
            const bool trains = balance >= cost(kTypeWorker) && (!strategy.military || keeps_training);
            chosen = plan(trains ? kTrain : kNothing);
            chosen.unit_type = kTypeWorker;
        } else if (which == kBarracksRole) {
            const int32_t trained = barracks_type();
            chosen = plan(balance >= cost(trained) ? kTrain : kNothing);
            chosen.unit_type = trained;
        } else {
            chosen = fighter_plan(slot);
        }

        accept(slot, carry_out(slot, chosen, true), true);
    }

    __device__ void visit_worker(int32_t slot, int32_t role) {
        const int32_t which = clip(role - kBuilderRole, 0, kFreeRole - kBuilderRole);
        RolePlan role_plan{plan(kNothing), true, true};

        if (which == 0) {
            role_plan = builder_plan(slot);
        } else if (which == 1) {
            role_plan = harvester_plan(slot);
        }

        const Plan chosen = role_plan.scouts ? scout_plan(slot) : role_plan.plan;
        Desire desire = carry_out(slot, chosen, false);

        if (role_plan.checks_conflicts && desire.kind == kKindMove && beside_listed_move(slot)) {
            desire = idle(slot);
        }

        if (lane == 0) {
            s.listed[slot - start] = desire.kind == kKindMove ? 1 : 0;
        }

        __syncwarp();
        accept(slot, desire, false);
    }

    __device__ void fill() {
        for (int32_t local = lane; local < capacity; local += kThreads) {
            const int32_t slot = start + local;
            const bool waits = (s.bits[slot] & (kMine | kAvailable)) == (kMine | kAvailable) && s.kind[local] == kKindAbsent;
            s.group[local] = waits ? 1 : 0;
        }

        __syncwarp();

        for (int32_t local = lane; local < capacity; local += kThreads) {
            if (s.group[local] == 0) {
                continue;
            }

            const int32_t born = s.birth[start + local];
            int32_t place = requests;

            for (int32_t other = 0; other < capacity; ++other) {
                place += s.group[other] != 0 && s.birth[start + other] < born ? 1 : 0;
            }

            s.kind[local] = static_cast<int8_t>(kKindNone);
            s.none_duration[local] = static_cast<int8_t>(kFillerDuration);
            s.rank[local] = static_cast<int16_t>(place);
        }

        __syncwarp();
    }

    __device__ void turn() {
        int32_t count = 0;
        int32_t first_pass_end = 0;
        entries(count, first_pass_end);

        for (int32_t index = 0; index < count; ++index) {
            const int32_t local = s.walk[index];
            const int32_t slot = start + local;
            const int32_t role = s.role[local];

            if (index < first_pass_end) {
                visit_first(slot, role);
            } else {
                visit_worker(slot, role);
            }
        }

        fill();
    }
};

template <typename Slot, typename Cell>
__device__ bool glance(Lane& a, const Inputs<Slot, Cell>& in, int64_t lane_index) {
    const int32_t* birth = in.state.birth.at(lane_index);
    const int8_t* unit_type = in.state.unit_type.at(lane_index);
    const int8_t* owner = in.state.owner.at(lane_index);
    const Cell* cell = in.state.cell.at(lane_index);
    const int8_t* pending_kind = in.state.pending_kind.at(lane_index);
    bool plays = false;
    int32_t workers = 0;
    int32_t bases = 0;
    Pick own_base = nothing_picked();
    Pick enemy_base = nothing_picked();

    for (int32_t e = a.lane; e < a.entities; e += kThreads) {
        const int32_t born = birth[e];

        if (born == kAbsent) {
            continue;
        }

        const int32_t holder = owner[e];
        const int32_t type = clip(unit_type[e], 0, a.types - 1);
        const bool mine = holder == a.player;
        const bool enemy = holder != kAbsent && holder != a.player;
        plays = plays || (mine && pending_kind[e] == kKindAbsent);
        workers += mine && type == kTypeWorker ? 1 : 0;
        bases += mine && type == kTypeBase ? 1 : 0;

        if (type == kTypeBase && mine) {
            offer(own_base, keyed(0, born), e);
        }

        if (type == kTypeBase && enemy) {
            offer(enemy_base, keyed(0, born), e);
        }
    }

    own_base = least_of(own_base);
    enemy_base = least_of(enemy_base);
    a.own_based = own_base.index != kHighest;
    a.enemy_based = enemy_base.index != kHighest;
    a.own_base_cell = a.own_based ? static_cast<int32_t>(cell[own_base.index]) : kAbsent;
    a.enemy_base_cell = a.enemy_based ? static_cast<int32_t>(cell[enemy_base.index]) : kAbsent;

    if (!a.memory.initialized) {
        a.remember_bases();
        a.memory.initialized = true;
        a.memory.weighted = warp_sum(workers) != 0 || warp_sum(bases) != 0;
    }

    return warp_any(plays);
}

template <typename Slot, typename Cell>
__device__ void load(Lane& a, const Inputs<Slot, Cell>& in, int64_t lane_index) {
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
    const int32_t* pending_completes_at = in.pending_completes_at.at(lane_index);
    const int32_t lane = a.lane;

    for (int32_t c = lane; c < a.cells; c += kThreads) {
        const int32_t occupant = entity_at[c];
        s.occupant[c] = static_cast<int16_t>(occupant);
        s.cells[c] = (terrain[c] ? kWall : 0) | (occupant != kAbsent ? kOccupied : 0);
    }

    for (int32_t local = lane; local < a.capacity; local += kThreads) {
        s.kind[local] = static_cast<int8_t>(kKindAbsent);
        s.direction[local] = static_cast<int8_t>(kAbsent);
        s.target[local] = static_cast<int16_t>(kAbsent);
        s.produced[local] = static_cast<int8_t>(kAbsent);
        s.none_duration[local] = 0;
        s.rank[local] = static_cast<int16_t>(kAbsent);
        s.listed[local] = 0;
    }

    __syncwarp();
    int32_t reserved = 0;
    int32_t own[kCounted] = {};
    int32_t theirs[kCounted] = {};
    const int32_t flat_steps[kDirections] = {-a.width, 1, a.width, -1};

    for (int32_t e = lane; e < a.entities; e += kThreads) {
        const int32_t born = birth[e];
        const bool present = born != kAbsent;
        const int32_t type = clip(unit_type[e], 0, a.types - 1);
        const int32_t holder = owner[e];
        const int32_t at = static_cast<int32_t>(cell[e]);
        const int32_t kind = present ? static_cast<int32_t>(pending_kind[e]) : kKindAbsent;
        const int32_t direction = pending_direction[e];
        const int32_t stepped = a.step(at, direction);
        const bool moving = present && kind == kKindMove;
        const bool claims = present && (kind == kKindMove || kind == kKindProduce);
        const bool resource = present && a.column(type, kIsResource) != 0;
        const bool mine = present && holder == a.player;
        const bool enemy = present && holder != kAbsent && holder != a.player;
        const bool stockpile = mine && a.column(type, kIsStockpile) != 0;
        const bool available = present && kind == kKindAbsent;
        const int32_t future = moving && stepped != kAbsent ? stepped : at;
        s.birth[e] = born;
        s.remaining[e] = moving ? wrapped_sub(pending_completes_at[e], a.time) : kHighest;
        s.cell[e] = static_cast<int16_t>(at);
        s.future[e] = static_cast<int16_t>(future);
        s.x[e] = static_cast<int16_t>(a.x_of(at));
        s.y[e] = static_cast<int16_t>(a.y_of(at));
        s.hit_points[e] = hit_points[e];
        s.type[e] = static_cast<uint8_t>(type);
        s.bits[e] = (present ? kPresent : 0) | (mine ? kMine : 0) | (enemy ? kEnemy : 0) | (resource ? kResource : 0) | (stockpile ? kStockpile : 0) | (available ? kAvailable : 0) | (carried[e] > 0 ? kCarrying : 0);

        if (claims && stepped != kAbsent) {
            or_cell(s.cells, stepped, kClaimed);
        }

        if (claims) {
            const int32_t blocked = scattered(at + flat_steps[clip(direction, 0, kDirections - 1)], a.cells);

            if (blocked != kAbsent) {
                or_cell(s.cells, blocked, kBlocks);
            }
        }

        if (resource && at != kAbsent) {
            const int32_t placed = scattered(at, a.cells);

            if (placed != kAbsent) {
                or_cell(s.cells, placed, kResourceAt);
            }
        }

        if (present && kind == kKindProduce && e >= a.start && e < a.start + a.capacity) {
            reserved += a.cost(clip(pending_produced_type[e], 0, a.types - 1));
        }

        for (int k = 0; k < kCounted; ++k) {
            own[k] += mine && type == kCountedTypes[k] ? 1 : 0;
            theirs[k] += enemy && type == kCountedTypes[k] ? 1 : 0;
        }
    }

    a.reserved = warp_sum(reserved);

    for (int k = 0; k < kCounted; ++k) {
        a.own[k] = warp_sum(own[k]);
        a.theirs[k] = warp_sum(theirs[k]);
    }

    __syncwarp();

    for (int32_t c = lane; c < a.cells; c += kThreads) {
        const uint8_t bits = s.cells[c];
        const bool free = (bits & (kWall | kOccupied | kBlocks)) == 0;
        s.cells[c] = static_cast<uint8_t>((bits & ~kBlocks) | (free ? kFree : 0));
    }

    __syncwarp();
}

template <typename Slot, typename Cell>
__global__ void __launch_bounds__(kThreads) act_kernel(Inputs<Slot, Cell> in, Outputs out, Extents size) {
    extern __shared__ uint64_t memory[];
    const int64_t lane_index = blockIdx.x;
    Lane a;
    a.s = Shared::carve(memory, size);
    a.table = in.table.at(lane_index);
    a.act_table = in.act_table.at(lane_index);
    a.lane = threadIdx.x;
    a.width = size.width;
    a.height = size.height;
    a.cells = size.cells;
    a.entities = size.entities;
    a.capacity = size.capacity;
    a.types = size.types;
    a.player = in.player.at(lane_index)[0];
    a.start = size.neutral + a.player * size.capacity;
    a.other_start = size.neutral + (1 - a.player) * size.capacity;
    a.time = in.time.at(lane_index)[0];
    a.balance = in.state.resources.at(lane_index)[clip(a.player, 0, nanorts::whole_act::kPlayers - 1)];
    a.memory.initialized = in.initialized.at(lane_index)[0];
    a.memory.main_base = in.main_base.at(lane_index)[0];
    a.memory.enemy_base = in.enemy_base.at(lane_index)[0];
    a.memory.base_to_enemy_base = in.base_to_enemy_base.at(lane_index)[0];
    a.memory.real_base_to_enemy = in.real_base_to_enemy.at(lane_index)[0];
    a.memory.weighted = in.weighted.at(lane_index)[0];
    a.requests = 0;
    a.used = 0;
    const bool plays = glance(a, in, lane_index);

    if (plays) {
        load(a, in, lane_index);
        a.macro();
        a.turn();
    }

    using nanorts::whole_act::kCommandFields;
    using nanorts::whole_act::kPlayers;
    const Shared& s = a.s;
    int32_t* commands = out.commands + lane_index * kCommandFields * kPlayers * size.capacity;

    for (int32_t local = a.lane; local < size.capacity; local += kThreads) {
        int32_t unissued[kCommandFields];
        unissued[kCommandKind] = kKindAbsent;
        unissued[kCommandDirection] = kAbsent;
        unissued[kCommandNoneDuration] = 0;
        unissued[kCommandTargetCell] = kAbsent;
        unissued[kCommandProducedType] = kAbsent;
        unissued[kCommandRequestOrder] = kAbsent;
        int32_t issued[kCommandFields];

        if (plays) {
            issued[kCommandKind] = s.kind[local];
            issued[kCommandDirection] = s.direction[local];
            issued[kCommandNoneDuration] = s.none_duration[local];
            issued[kCommandTargetCell] = s.target[local];
            issued[kCommandProducedType] = s.produced[local];
            issued[kCommandRequestOrder] = s.rank[local];
        }

        for (int field = 0; field < kCommandFields; ++field) {
            for (int32_t who = 0; who < kPlayers; ++who) {
                commands[(field * kPlayers + who) * size.capacity + local] = plays && who == a.player ? issued[field] : unissued[field];
            }
        }
    }

    if (a.lane == 0) {
        out.initialized[lane_index] = a.memory.initialized;
        out.main_base[lane_index] = a.memory.main_base;
        out.enemy_base[lane_index] = a.memory.enemy_base;
        out.base_to_enemy_base[lane_index] = a.memory.base_to_enemy_base;
        out.real_base_to_enemy[lane_index] = a.memory.real_base_to_enemy;
        out.weighted[lane_index] = a.memory.weighted;
    }
}

template <typename Slot, typename Cell>
ffi::Error launch(cudaStream_t stream, nanorts::whole_act::Operands& operands, const Outputs& outputs, const Extents& size, int64_t lanes) {
    Inputs<Slot, Cell> in{};
    const int64_t entities = size.entities;

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

    if (auto e = operands.next(entities, in.pending_completes_at); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.time); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{size.types} * kActColumns, in.act_table); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.initialized); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.main_base); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.enemy_base); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.base_to_enemy_base); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.real_base_to_enemy); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.weighted); e.failure()) {
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

constexpr int64_t kExtraOperands = 4;
constexpr int64_t kReadOperands = nanorts::whole_act::kStateOperands + 2 + kExtraOperands;
constexpr int64_t kCarryLeaves = 6;
constexpr int64_t kHitPointsOperand = nanorts::whole_act::kStateOperands + 2;
constexpr int64_t kActTableOperand = kHitPointsOperand + 3;

ffi::Error Act(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets results, std::string_view build_id) {
    using ffi::DataType;
    constexpr std::string_view kTarget = "nanorts_tma_act";

    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (args.size() != kReadOperands + kCarryLeaves || results.size() != kCarryLeaves + 1) {
        return ffi::Error::InvalidArgument(
            "nanorts_tma_act takes the State leaves, the unit tables, the player and the macro "
            "memory, and returns the macro memory and the Commands");
    }

    auto commands = results.get<ffi::AnyBuffer>(kCarryLeaves);

    if (!commands.has_value()) {
        return commands.error();
    }

    ffi::AnyBuffer table_rows = *args.get<ffi::AnyBuffer>(nanorts::whole_act::kStateOperands);
    ffi::AnyBuffer act_rows = *args.get<ffi::AnyBuffer>(kActTableOperand);
    const auto issued = (*commands)->dimensions();
    const auto table = table_rows.dimensions();
    const auto acted = act_rows.dimensions();

    if ((*commands)->element_type() != ffi::S32 || issued.size() != 4 || issued[1] != nanorts::whole_act::kCommandFields || issued[2] != nanorts::whole_act::kPlayers || table.size() != 3 || table[2] != kTableColumns || table_rows.element_type() != ffi::S32 || acted.size() != 3 || acted[1] != table[1] || acted[2] != kActColumns || act_rows.element_type() != ffi::S32) {
        return ffi::Error::InvalidArgument(
            "the Commands are `[lanes, 6, 2, C]` int32 and the unit tables `[1, T, 6]` and "
            "`[1, T, 2]` int32");
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

    if (size.cells <= 0 || size.capacity <= 0 || size.neutral < 0 || size.types <= kTypeRanged || shared > nanorts::whole_act::kSharedOptInLimit || nanorts::whole_act::threads_for(size.capacity) > nanorts::whole_act::kMaxThreads) {
        return ffi::Error::InvalidArgument(
            "a table holding every unit type tma names, over a lane whose act fits one block's "
            "shared memory");
    }

    auto hit_points = operands.at(kHitPointsOperand);

    if (!hit_points.has_value()) {
        return hit_points.error();
    }

    if (hit_points->element_type() != DataType::S16) {
        return ffi::Error::InvalidArgument("hit points are int16");
    }

    Outputs outputs{};
    int64_t result = 0;
    int64_t operand = kReadOperands;
    const auto bind = [&](int64_t elements, auto*& into) { return nanorts::whole_act::carry_result(operands, results, result++, operand++, lanes, elements, into); };

    if (auto e = bind(1, outputs.initialized); e.failure()) {
        return e;
    }

    if (auto e = bind(1, outputs.main_base); e.failure()) {
        return e;
    }

    if (auto e = bind(1, outputs.enemy_base); e.failure()) {
        return e;
    }

    if (auto e = bind(1, outputs.base_to_enemy_base); e.failure()) {
        return e;
    }

    if (auto e = bind(1, outputs.real_base_to_enemy); e.failure()) {
        return e;
    }

    if (auto e = bind(1, outputs.weighted); e.failure()) {
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

const nanorts::Kernels nanorts_tma_act_kernels("nanorts_tma_act", {nanorts::kernel(act_kernel<int16_t, int16_t>), nanorts::kernel(act_kernel<int16_t, int32_t>), nanorts::kernel(act_kernel<int32_t, int16_t>), nanorts::kernel(act_kernel<int32_t, int32_t>)});

}

NANORTS_HANDLER(nanorts_tma_act, Act,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .RemainingArgs()
                    .RemainingRets()
                    .Attr<std::string_view>("build_id"));
