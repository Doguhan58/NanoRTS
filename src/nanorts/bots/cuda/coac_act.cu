#include <cstdint>

#include "astar.cuh"
#include "nanorts_ffi.cuh"
#include "whole_act.cuh"

namespace {

namespace ffi = xla::ffi;
namespace astar = nanorts::astar;
using nanorts::whole_act::Extents;
using nanorts::whole_act::Lanes;
using nanorts::whole_act::StateRows;

constexpr int64_t kBytesPerCell = 8;
static_assert(kBytesPerCell == astar::kScratchBytesPerCell + 1);
constexpr int64_t kLaneBytes = 1024;
constexpr int64_t kBytesPerSlot = 84;
constexpr int64_t kBytesPerEntity = 14;

constexpr int32_t kAbsent = -1;
constexpr int32_t kLast = 2147483647;
constexpr int32_t kLowest = -kLast - 1;
constexpr int kDirections = 4;
constexpr int32_t kWaitOne = 1;
constexpr unsigned kEveryThread = 0xffffffffu;
constexpr int kMaxWarps = nanorts::whole_act::kMaxThreads / nanorts::whole_act::kWarp;
constexpr int kMaxQueries = 4;

constexpr int32_t kNoAction = -1;
constexpr int32_t kMove = 0;
constexpr int32_t kHarvest = 2;
constexpr int32_t kTrain = 3;
constexpr int32_t kTrainDirection = 6;
constexpr int32_t kBuildModified = 7;
constexpr int32_t kCoacAttack = 8;
constexpr int32_t kTrueIdle = 9;
constexpr int32_t kHarvestReturn = 10;

constexpr int32_t kKindAbsent = -1;
constexpr int32_t kKindNone = 0;
constexpr int32_t kKindMove = 1;
constexpr int32_t kKindHarvest = 2;
constexpr int32_t kKindReturn = 3;
constexpr int32_t kKindProduce = 4;
constexpr int32_t kKindAttack = 5;

constexpr int32_t kTypeBarracks = 2;
constexpr int32_t kTypeWorker = 3;
constexpr int32_t kTypeHeavy = 5;
constexpr int32_t kTypeRanged = 6;

constexpr int32_t kUp = 0;
constexpr int32_t kRight = 1;
constexpr int32_t kDown = 2;
constexpr int32_t kLeft = 3;

constexpr int kCanHarvest = 0;
constexpr int kIsResource = 1;
constexpr int kIsStockpile = 2;
constexpr int kCanAttack = 3;
constexpr int kAttackRange = 4;
constexpr int kCost = 5;
constexpr int kTableColumns = 6;

constexpr int kAttackDamage = 0;
constexpr int kAttackTime = 1;
constexpr int kMoveTime = 2;
constexpr int kProduceTime = 3;
constexpr int kTurnColumns = 4;

constexpr int kCommandKind = 0;
constexpr int kCommandDirection = 1;
constexpr int kCommandNoneDuration = 2;
constexpr int kCommandTargetCell = 3;
constexpr int kCommandProducedType = 4;
constexpr int kCommandRequestOrder = 5;

constexpr int kEntryKind = 0;
constexpr int kEntryUnitBirth = 1;
constexpr int kEntryOrder = 2;
constexpr int kEntryTarget = 3;
constexpr int kEntryTargetBirth = 4;
constexpr int kEntryBase = 5;
constexpr int kEntryBaseBirth = 6;
constexpr int kEntryCell = 7;
constexpr int kEntryUnitType = 8;
constexpr int kEntryRows = 9;

constexpr int kHarvester = 0;
constexpr int kHarvestBase = 1;
constexpr int kHarvestBaseBirth = 2;
constexpr int kHarvestRows = 3;

constexpr int32_t kUnseen = -1;
constexpr int32_t kDefaultOpening = 0;
constexpr int32_t kSixteenAOpening = 1;
constexpr int32_t kSixteenABalance = 5;
constexpr int kSixteenAUnits = 8;
constexpr int kSixteenAFields = 6;
constexpr int32_t kFloodWidth = 10;
constexpr int32_t kAlwaysTrainingWidth = 8;
constexpr int32_t kWorkersPerBase = 2;
constexpr int32_t kAbundant = 15;
constexpr int32_t kRich = 10;
constexpr int32_t kCombatLead = 4;
constexpr int32_t kChaseDistance = 5;
constexpr int32_t kCentroidDistance = 3;
constexpr int32_t kDangerDistance = 2;
constexpr int32_t kBuilderReach = 2;
constexpr int32_t kBarracksSpare = 1;
constexpr int32_t kFar = 9999999;
constexpr int32_t kHeaviesBeforeRanged = 3;

constexpr int32_t kFellBack = 0;
constexpr int32_t kBlocked = 1;
constexpr int32_t kMoved = 2;

constexpr uint8_t kOpen = 1;
constexpr uint8_t kTaken = 2;
constexpr uint8_t kEmpty = 4;
constexpr uint8_t kHolding = 8;

constexpr uint16_t kActive = 1u << 0;
constexpr uint16_t kMine = 1u << 1;
constexpr uint16_t kEnemy = 1u << 2;
constexpr uint16_t kResource = 1u << 3;
constexpr uint16_t kStockpile = 1u << 4;
constexpr uint16_t kCombat = 1u << 5;
constexpr uint16_t kMyWorker = 1u << 6;
constexpr uint16_t kMyBase = 1u << 7;
constexpr uint16_t kMyBarracks = 1u << 8;
constexpr uint16_t kAvailable = 1u << 9;
constexpr uint16_t kMoving = 1u << 10;
constexpr uint16_t kAlive = 1u << 11;
constexpr uint16_t kFreeWorker = 1u << 12;
constexpr uint16_t kClosestResource = 1u << 13;
constexpr uint16_t kSeparated = 1u << 14;

constexpr uint8_t kExecutes = 1;
constexpr uint8_t kStanding = 2;
constexpr uint8_t kFiller = 4;
constexpr uint8_t kEligible = 8;
constexpr uint8_t kDraws = 16;
constexpr uint8_t kBlockedEligible = 32;
constexpr uint8_t kDrew = 64;

__constant__ int32_t kSixteenA[kSixteenAUnits][kSixteenAFields] = {
    {0, -1, 0, 0, 1, 25},
    {0, -1, 0, 1, 1, 25},
    {0, -1, 15, 14, 1, 25},
    {0, -1, 15, 15, 1, 25},
    {1, 0, 2, 2, 10, 0},
    {1, 1, 13, 13, 10, 0},
    {3, 0, 1, 1, 1, 0},
    {3, 1, 14, 14, 1, 0}
};

constexpr int64_t shared_bytes(int64_t cells, int64_t capacity, int64_t entities) {
    return kBytesPerCell * cells + kLaneBytes + kBytesPerSlot * capacity + kBytesPerEntity * entities;
}

__device__ int32_t clip(int32_t value, int32_t low, int32_t high) {
    return value < low ? low : value > high ? high : value;
}

__device__ int32_t gathered(int32_t index, int32_t size) {
    return clip(index < 0 ? index + size : index, 0, size - 1);
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

__device__ int32_t step_x(int32_t direction) {
    const int32_t known = clip(direction, 0, kDirections - 1);
    return known == kRight ? 1 : known == kLeft ? -1 : 0;
}

__device__ int32_t step_y(int32_t direction) {
    const int32_t known = clip(direction, 0, kDirections - 1);
    return known == kUp ? -1 : known == kDown ? 1 : 0;
}

__device__ int32_t neighbour(int32_t cell, int32_t direction, int32_t width) {
    return wrapped_add(cell, step_y(direction) * width + step_x(direction));
}

__device__ int32_t squared_distance(int32_t cell, int32_t other, int32_t width) {
    const int32_t dx = astar::floor_mod(cell, width) - astar::floor_mod(other, width);
    const int32_t dy = astar::floor_div(cell, width) - astar::floor_div(other, width);
    return wrapped_add(wrapped_mul(dx, dx), wrapped_mul(dy, dy));
}

__device__ int32_t adjacent_direction(int32_t origin, int32_t goal, int32_t width) {
    if (goal == kAbsent) {
        return kAbsent;
    }

    const int32_t dx = astar::floor_mod(goal, width) - astar::floor_mod(origin, width);
    const int32_t dy = astar::floor_div(goal, width) - astar::floor_div(origin, width);

    for (int32_t direction = 0; direction < kDirections; ++direction) {
        if (dx == step_x(direction) && dy == step_y(direction)) {
            return direction;
        }
    }

    return kAbsent;
}

__constant__ int32_t kFleeOrder[kDirections] = {kUp, kDown, kRight, kLeft};
__constant__ int32_t kTrainOrder[kDirections] = {kUp, kDown, kLeft, kRight};
__constant__ int32_t kAdjacentOrder[kDirections] = {kDown, kUp, kRight, kLeft};

__device__ int32_t max_cycles(int32_t width) {
    return width <= 8 ? 3000 : width <= 16 ? 4000 : width <= 24 ? 5000 : width <= 32 ? 6000 : width <= 64 ? 8000 : 12000;
}

struct Least {
    int32_t key;
    int32_t order;
    int32_t slot;
};

__device__ Least nothing(int32_t width) {
    return Least{kLast, kLast, width};
}

__device__ bool before(const Least& given, const Least& other) {
    return given.key < other.key || (given.key == other.key && (given.order < other.order || (given.order == other.order && given.slot < other.slot)));
}

__device__ void offer(Least& best, bool candidate, int32_t key, int32_t order, int32_t slot) {
    const Least given{key, order, slot};

    if (candidate && before(given, best)) {
        best = given;
    }
}

__device__ int32_t found(const Least& best, int32_t width) {
    return best.slot < width ? best.slot : kAbsent;
}

__device__ int32_t nearest(const uint16_t* bits, const int16_t* xs, const int16_t* ys, const int32_t* births, uint16_t bit, int32_t x, int32_t y, int32_t begin, int32_t count) {
    Least best = nothing(count);

    for (int32_t index = 0; index < count; ++index) {
        const int32_t slot = begin + index;

        if ((bits[slot] & bit) == 0) {
            continue;
        }

        const int32_t born = births[slot];
        offer(best, true, astar::manhattan(xs[slot], ys[slot], x, y), born != kAbsent ? born : kLast, index);
    }

    return found(best, count);
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
    Lanes<int32_t> draws;
    Lanes<int32_t> entries[kEntryRows];
    Lanes<bool> trained;
    Lanes<int32_t> next_order;
    Lanes<int32_t> harvest[kHarvestRows];
    Lanes<bool> was_separated;
    Lanes<bool> attack_all;
    Lanes<int32_t> opening;
};

struct Outputs {
    int32_t* entries[kEntryRows];
    bool* trained;
    int32_t* next_order;
    int32_t* harvest[kHarvestRows];
    bool* was_separated;
    bool* attack_all;
    int32_t* opening;
    int32_t* commands;
    bool* drew;
};

struct Facts {
    uint32_t reserved[nanorts::whole_act::kPlayers];
    int32_t active_count;
    int32_t my_combat_cost;
    int32_t other_combat_cost;
    int32_t company;
    int32_t company_x_sum;
    int32_t company_y_sum;
    int32_t heavies;
    int32_t my_workers;
    int32_t bases;
    int32_t busy_bases;
    int32_t barracks;
    int32_t resources;
    int32_t separated;
    int32_t opening_differs;
    int32_t next_order;
    int32_t resources_used;
    int32_t stop;
    int32_t queued;
    int32_t kept_count;
    int32_t standing;
};

struct Shared {
    astar::Scratch scratch;
    uint8_t* flags;
    Facts* facts;
    Least* least;
    uint32_t* bits_scratch;
    int32_t* damage;
    int32_t* entries[kEntryRows];
    int32_t* harvest[kHarvestRows];
    int32_t* goal;
    int32_t* radius;
    int32_t* claim;
    int32_t* cost;
    int16_t* x;
    int16_t* y;
    int16_t* next_x;
    int16_t* next_y;
    int16_t* builder_base;
    int16_t* near_enemy;
    int16_t* per_base;
    int16_t* members;
    int16_t* start;
    int16_t* queue;
    int16_t* rank;
    uint16_t* bits;
    uint8_t* trained;
    uint8_t* listed;
    uint8_t* marks;
    uint8_t* selector;
    int8_t* direction;
    uint8_t* options;

    __device__ static Shared carve(uint64_t* memory, const Extents& size) {
        const int32_t cells = size.cells;
        const int32_t capacity = size.capacity;
        const int32_t entities = size.entities;
        Shared carved;
        carved.scratch = astar::Scratch::carve(memory, cells);
        carved.flags = carved.scratch.seen + cells;
        char* const lane = reinterpret_cast<char*>(memory) + kBytesPerCell * cells;
        carved.facts = reinterpret_cast<Facts*>(lane);
        carved.least = reinterpret_cast<Least*>(lane + 128);
        carved.bits_scratch = reinterpret_cast<uint32_t*>(carved.least + 2 * kMaxQueries * kMaxWarps);
        int32_t* words = reinterpret_cast<int32_t*>(lane + kLaneBytes);
        carved.damage = words;
        words += entities;

        for (int k = 0; k < kEntryRows; ++k, words += capacity) {
            carved.entries[k] = words;
        }

        for (int k = 0; k < kHarvestRows; ++k, words += capacity) {
            carved.harvest[k] = words;
        }

        carved.goal = words;
        carved.radius = words + capacity;
        carved.claim = words + 2 * capacity;
        carved.cost = words + 3 * capacity;
        int16_t* halves = reinterpret_cast<int16_t*>(words + 4 * capacity);
        carved.x = halves;
        carved.y = halves + entities;
        carved.next_x = halves + 2 * entities;
        carved.next_y = halves + 3 * entities;
        halves += 4 * entities;
        carved.builder_base = halves;
        carved.near_enemy = halves + capacity;
        carved.per_base = halves + 2 * capacity;
        carved.members = halves + 3 * capacity;
        carved.start = halves + 4 * capacity;
        carved.queue = halves + 5 * capacity;
        carved.rank = halves + 6 * capacity;
        carved.bits = reinterpret_cast<uint16_t*>(halves + 7 * capacity);
        uint8_t* bytes = reinterpret_cast<uint8_t*>(carved.bits + entities);
        carved.trained = bytes;
        carved.listed = bytes + capacity;
        carved.marks = bytes + 2 * capacity;
        carved.selector = bytes + 3 * capacity;
        carved.direction = reinterpret_cast<int8_t*>(bytes + 4 * capacity);
        carved.options = bytes + 5 * capacity;
        return carved;
    }
};

static_assert(sizeof(Facts) <= 128, "the lane's facts outgrew their bytes");
static_assert(128 + 2 * kMaxQueries * kMaxWarps * sizeof(Least) + 2 * kMaxWarps * sizeof(uint32_t) <= kLaneBytes, "the lane's reduction scratch outgrew its bytes");

template <typename Slot, typename Cell>
struct Lane {
    const int32_t* birth;
    const int8_t* unit_type;
    const int8_t* owner;
    const Cell* cell;
    const int32_t* carried;
    const int8_t* pending_kind;
    const int8_t* pending_direction;
    const int8_t* pending_produced_type;
    const Slot* entity_at;
    const int32_t* resources;
    const bool* terrain;
    const int32_t* table;
    const int16_t* hit_points;
    const Cell* pending_target_cell;
    const int32_t* pending_completes_at;
    const int32_t* turn_table;
    const int32_t* draws;
    int32_t player;
    int32_t time;
    int32_t first;
    Extents size;

    __device__ Lane(const Inputs<Slot, Cell>& in, int64_t lane, const Extents& extents) : birth(in.state.birth.at(lane)), unit_type(in.state.unit_type.at(lane)), owner(in.state.owner.at(lane)), cell(in.state.cell.at(lane)), carried(in.state.carried.at(lane)), pending_kind(in.state.pending_kind.at(lane)), pending_direction(in.state.pending_direction.at(lane)), pending_produced_type(in.state.pending_produced_type.at(lane)), entity_at(in.state.entity_at.at(lane)), resources(in.state.resources.at(lane)), terrain(in.state.terrain.at(lane)), table(in.table.at(lane)), hit_points(in.hit_points.at(lane)), pending_target_cell(in.pending_target_cell.at(lane)), pending_completes_at(in.pending_completes_at.at(lane)), turn_table(in.turn_table.at(lane)), draws(in.draws.at(lane)), player(in.player.at(lane)[0]), time(in.time.at(lane)[0]), first(extents.neutral + in.player.at(lane)[0] * extents.capacity), size(extents) {}

    __device__ int32_t own_row(int32_t member) const {
        return clip(first, 0, size.entities - size.capacity) + member;
    }

    __device__ bool active(int32_t slot) const {
        return birth[slot] != kAbsent;
    }

    __device__ int32_t order(int32_t slot) const {
        const int32_t born = birth[slot];
        return born != kAbsent ? born : kLast;
    }

    __device__ int32_t type_of(int32_t slot) const {
        return clip(unit_type[slot], 0, size.types - 1);
    }

    __device__ int32_t column(int32_t type, int at) const {
        return table[type * kTableColumns + at];
    }

    __device__ int32_t turn_column(int32_t type, int at) const {
        return turn_table[type * kTurnColumns + at];
    }

    __device__ int32_t cost(int32_t type) const {
        return column(type, kCost);
    }

    __device__ int32_t cell_at(int32_t slot) const {
        return static_cast<int32_t>(cell[slot]);
    }

    __device__ bool exists(int32_t slot, int32_t born) const {
        return slot != kAbsent && birth[clip(slot, 0, size.entities - 1)] == born;
    }

    __device__ int32_t cell_of(int32_t slot) const {
        return slot != kAbsent ? cell_at(clip(slot, 0, size.entities - 1)) : kAbsent;
    }

    __device__ int32_t balance(int32_t who) const {
        return resources[gathered(who, nanorts::whole_act::kPlayers)];
    }

    __device__ bool on_board(int32_t x, int32_t y) const {
        return x >= 0 && x < size.width && y >= 0 && y < size.height;
    }

    __device__ int32_t cell_from(int32_t x, int32_t y) const {
        return on_board(x, y) ? y * size.width + x : 0;
    }
};

template <typename Slot, typename Cell>
__device__ int32_t walk_order(const Lane<Slot, Cell>& lane, const Shared& shared, bool listed);

template <typename Slot, typename Cell>
struct Walk {
    const Lane<Slot, Cell>& lane;
    const Shared& shared;
    int parity;

    __device__ int32_t x(int32_t slot) const {
        return shared.x[slot];
    }

    __device__ int32_t y(int32_t slot) const {
        return shared.y[slot];
    }

    __device__ uint16_t bits(int32_t slot) const {
        return shared.bits[slot];
    }

    __device__ bool is(int32_t slot, uint16_t bit) const {
        return (shared.bits[slot] & bit) != 0;
    }

    __device__ int32_t distance(int32_t slot, int32_t ux, int32_t uy) const {
        return astar::manhattan(x(slot), y(slot), ux, uy);
    }

    __device__ int32_t members(uint16_t all_of, uint16_t none_of = 0, bool starts = true) const {
        const bool owns = static_cast<int32_t>(threadIdx.x) < lane.size.capacity;
        const uint16_t held = owns ? bits(lane.own_row(threadIdx.x)) : 0;
        const bool listed = owns && starts && (held & all_of) == all_of && (held & none_of) == 0;
        return walk_order(lane, shared, listed);
    }

    __device__ int32_t distance_to(int32_t slot, int32_t ux, int32_t uy) const {
        return distance(clip(slot, 0, lane.size.entities - 1), ux, uy);
    }

    __device__ int32_t birth_of(int32_t slot) const {
        return slot != kAbsent ? lane.order(clip(slot, 0, lane.size.entities - 1)) : kAbsent;
    }

    template <int K>
    __device__ void reduce(Least (&best)[K], uint32_t& any) {
        static_assert(K <= kMaxQueries);

        for (int k = 0; k < K; ++k) {
            for (int offset = nanorts::whole_act::kWarp / 2; offset > 0; offset /= 2) {
                const Least other{__shfl_xor_sync(kEveryThread, best[k].key, offset), __shfl_xor_sync(kEveryThread, best[k].order, offset), __shfl_xor_sync(kEveryThread, best[k].slot, offset)};

                if (before(other, best[k])) {
                    best[k] = other;
                }
            }
        }

        for (int offset = nanorts::whole_act::kWarp / 2; offset > 0; offset /= 2) {
            any |= __shfl_xor_sync(kEveryThread, any, offset);
        }

        Least* buffer = shared.least + parity * kMaxQueries * kMaxWarps;
        uint32_t* ored = shared.bits_scratch + parity * kMaxWarps;
        parity ^= 1;
        const int warp = threadIdx.x / nanorts::whole_act::kWarp;

        if (threadIdx.x % nanorts::whole_act::kWarp == 0) {
            for (int k = 0; k < K; ++k) {
                buffer[k * kMaxWarps + warp] = best[k];
            }

            ored[warp] = any;
        }

        __syncthreads();
        const int warps = blockDim.x / nanorts::whole_act::kWarp;

        for (int k = 0; k < K; ++k) {
            best[k] = buffer[k * kMaxWarps];

            for (int other = 1; other < warps; ++other) {
                if (before(buffer[k * kMaxWarps + other], best[k])) {
                    best[k] = buffer[k * kMaxWarps + other];
                }
            }
        }

        any = ored[0];

        for (int other = 1; other < warps; ++other) {
            any |= ored[other];
        }
    }

    template <int K>
    __device__ void reduce(Least (&best)[K]) {
        uint32_t any = 0;
        reduce(best, any);
    }
};

struct Fields {
    int32_t target = kAbsent;
    int32_t target_birth = kAbsent;
    int32_t base = kAbsent;
    int32_t base_birth = kAbsent;
    int32_t cell = kAbsent;
    int32_t unit_type = kAbsent;
};

template <typename Slot, typename Cell>
__device__ void put(const Lane<Slot, Cell>& lane, const Shared& shared, int32_t slot, int32_t kind, const Fields& fields) {
    const int32_t member = slot - lane.first;

    if (slot == kAbsent || member < 0 || member >= lane.size.capacity) {
        return;
    }

    const int32_t born = lane.birth[lane.own_row(member)];
    int32_t* const* entries = shared.entries;
    const bool known = entries[kEntryKind][member] != kNoAction && entries[kEntryUnitBirth][member] == born;

    if (!known) {
        entries[kEntryOrder][member] = shared.facts->next_order;
        shared.facts->next_order = wrapped_add(shared.facts->next_order, 1);
    }

    entries[kEntryKind][member] = kind;
    entries[kEntryUnitBirth][member] = born;
    entries[kEntryTarget][member] = fields.target;
    entries[kEntryTargetBirth][member] = fields.target_birth;
    entries[kEntryBase][member] = fields.base;
    entries[kEntryBaseBirth][member] = fields.base_birth;
    entries[kEntryCell][member] = fields.cell;
    entries[kEntryUnitType][member] = fields.unit_type;
    shared.trained[member] = 0;
}

template <typename Slot, typename Cell>
__device__ int32_t walk_order(const Lane<Slot, Cell>& lane, const Shared& shared, bool listed) {
    const int32_t thread = threadIdx.x;
    const int32_t capacity = lane.size.capacity;
    const bool owns = thread < capacity;

    if (owns) {
        shared.listed[thread] = listed ? 1 : 0;
    }

    const int32_t count = __syncthreads_count(owns && listed);
    bool irregular = false;

    if (owns && listed) {
        const int32_t born = lane.order(lane.own_row(thread));
        irregular = born == kLast || born == kLowest;
        int32_t rank = 0;

        for (int32_t member = 0; member < capacity; ++member) {
            if (member == thread || shared.listed[member] == 0) {
                continue;
            }

            const int32_t other = lane.order(lane.own_row(member));
            rank += other < born || (other == born && member < thread);
            irregular |= other == born;
        }

        shared.members[rank] = static_cast<int16_t>(thread);
    }

    if (__syncthreads_or(irregular)) {
        if (thread == 0) {
            int32_t after = kLowest;

            for (int32_t trip = 0; trip < count; ++trip) {
                int32_t member = 0;
                int32_t least = kLast;

                for (int32_t other = 0; other < capacity; ++other) {
                    const int32_t born = lane.order(lane.own_row(other));
                    const int32_t key = shared.listed[other] != 0 && born > after ? born : kLast;

                    if (key < least) {
                        least = key;
                        member = other;
                    }
                }

                shared.members[trip] = static_cast<int16_t>(member);
                after = lane.order(lane.own_row(member));
            }
        }

        __syncthreads();
    }

    return count;
}

template <typename Slot, typename Cell>
__device__ void clear_lane(const Lane<Slot, Cell>& lane, const Shared& shared, int32_t next_order) {
    const Extents& size = lane.size;

    if (threadIdx.x == 0) {
        *shared.facts = Facts{};
        shared.facts->next_order = next_order;
    }

    for (int32_t cell = threadIdx.x; cell < size.cells; cell += blockDim.x) {
        shared.scratch.top[cell] = 0;
        shared.scratch.below[cell] = 0;
        shared.scratch.parent[cell] = 0;
    }

    for (int32_t slot = threadIdx.x; slot < size.entities; slot += blockDim.x) {
        shared.damage[slot] = 0;
    }

    for (int32_t member = threadIdx.x; member < size.capacity; member += blockDim.x) {
        shared.per_base[member] = 0;
    }
}

template <typename Slot, typename Cell>
__device__ void read_slots(const Lane<Slot, Cell>& lane, const Shared& shared) {
    const Extents& size = lane.size;
    Facts* facts = shared.facts;

    for (int32_t slot = threadIdx.x; slot < size.entities; slot += blockDim.x) {
        const bool active = lane.active(slot);
        const int32_t type = lane.type_of(slot);
        const int32_t holder = lane.owner[slot];
        const int32_t at = lane.cell_at(slot);
        const int32_t x = astar::floor_mod(at, size.width);
        const int32_t y = astar::floor_div(at, size.width);
        const int32_t kind = lane.pending_kind[slot];
        const bool moving = active && kind == kKindMove;
        const int32_t direction = lane.pending_direction[slot];
        const bool mine = active && holder == lane.player;
        const bool enemy = active && holder != kAbsent && holder != lane.player;
        const bool resource = active && lane.column(type, kIsResource) != 0;
        const bool stockpile = active && !resource && lane.column(type, kIsStockpile) != 0;
        const bool others = active && !resource && !stockpile;
        const bool harvests = lane.column(type, kCanHarvest) != 0;
        const bool combat = others && !harvests && lane.column(type, kCanAttack) != 0;
        const bool my_worker = others && harvests && mine;
        const bool my_base = stockpile && mine;
        const bool my_barracks = others && !harvests && !combat && type == kTypeBarracks && mine;
        const bool available = active && kind == kKindAbsent;
        uint16_t bits = 0;
        bits |= active ? kActive : 0;
        bits |= mine ? kMine : 0;
        bits |= enemy ? kEnemy : 0;
        bits |= resource ? kResource : 0;
        bits |= stockpile ? kStockpile : 0;
        bits |= combat ? kCombat : 0;
        bits |= my_worker ? kMyWorker : 0;
        bits |= my_base ? kMyBase : 0;
        bits |= my_barracks ? kMyBarracks : 0;
        bits |= available ? kAvailable : 0;
        bits |= moving ? kMoving : 0;
        bits |= my_worker && available ? kFreeWorker : 0;
        shared.bits[slot] = bits;
        shared.x[slot] = static_cast<int16_t>(x);
        shared.y[slot] = static_cast<int16_t>(y);
        shared.next_x[slot] = static_cast<int16_t>(moving ? x + step_x(direction) : x);
        shared.next_y[slot] = static_cast<int16_t>(moving ? y + step_y(direction) : y);

        if (!active) {
            continue;
        }

        if (kind == kKindMove || kind == kKindProduce) {
            int32_t target = neighbour(at, direction, size.width);

            if (target < 0) {
                target += size.cells;
            }

            if (target >= 0 && target < size.cells) {
                shared.scratch.top[target] = 1;
            }
        }

        if (kind == kKindProduce && slot >= size.neutral) {
            const int32_t block = (slot - size.neutral) / size.capacity;
            const int32_t produced = clip(lane.pending_produced_type[slot], 0, size.types - 1);
            atomicAdd(&facts->reserved[block], static_cast<uint32_t>(lane.cost(produced)));
        }

        if ((my_worker || (combat && mine)) && kind == kKindAttack) {
            const int32_t target = static_cast<int32_t>(lane.pending_target_cell[slot]);

            if (target != kAbsent) {
                int32_t victim = static_cast<int32_t>(lane.entity_at[clip(target, 0, size.cells - 1)]);

                if (victim != kAbsent) {
                    if (victim < 0) {
                        victim += size.entities;
                    }

                    if (victim >= 0 && victim < size.entities) {
                        atomicAdd(&shared.damage[victim], lane.turn_column(type, kAttackDamage));
                    }
                }
            }
        }

        atomicAdd(&facts->active_count, 1);

        if (combat && mine) {
            atomicAdd(&facts->my_combat_cost, lane.cost(type));
            atomicAdd(&facts->company, 1);
            atomicAdd(&facts->company_x_sum, x);
            atomicAdd(&facts->company_y_sum, y);

            if (type == kTypeHeavy) {
                atomicAdd(&facts->heavies, 1);
            }
        }

        if (combat && !mine) {
            atomicAdd(&facts->other_combat_cost, lane.cost(type));
        }

        if (my_worker) {
            atomicAdd(&facts->my_workers, 1);
        }

        if (my_base) {
            atomicAdd(&facts->bases, 1);

            if (!available) {
                atomicAdd(&facts->busy_bases, 1);
            }
        }

        if (my_barracks) {
            atomicAdd(&facts->barracks, 1);
        }

        if (resource) {
            atomicAdd(&facts->resources, 1);
        }

        if (at >= 0 && at < size.cells) {
            if (resource) {
                shared.scratch.below[at] = 1;
            }

            if (enemy) {
                shared.scratch.parent[at] = 1;
            }
        }
    }
}

template <typename Slot, typename Cell>
__device__ void flag_cells(const Lane<Slot, Cell>& lane, const Shared& shared) {
    for (int32_t cell = threadIdx.x; cell < lane.size.cells; cell += blockDim.x) {
        const bool holding = static_cast<int32_t>(lane.entity_at[cell]) != kAbsent;
        const bool empty = !lane.terrain[cell] && !holding;
        const bool pending = shared.scratch.top[cell] != 0;
        shared.flags[cell] = (empty ? kEmpty : 0) | (pending ? kTaken : 0) | (empty && !pending ? kOpen : 0) | (holding ? kHolding : 0);
    }
}

template <typename Slot, typename Cell>
__device__ void read_targets(const Lane<Slot, Cell>& lane, const Shared& shared) {
    const Extents& size = lane.size;

    for (int32_t slot = threadIdx.x; slot < size.entities; slot += blockDim.x) {
        const uint16_t bits = shared.bits[slot];

        if ((bits & kEnemy) != 0 && shared.damage[slot] < lane.hit_points[slot]) {
            shared.bits[slot] = bits | kAlive;
        }

        if ((bits & kResource) == 0) {
            continue;
        }

        const int32_t base = nearest(shared.bits, shared.x, shared.y, lane.birth, kStockpile, shared.x[slot], shared.y[slot], 0, size.entities);

        if (base != kAbsent && (shared.bits[clip(base, 0, size.entities - 1)] & kMine) != 0) {
            shared.bits[slot] |= kClosestResource;
        }
    }
}

template <typename Slot, typename Cell>
__device__ void cut_off_bases(const Lane<Slot, Cell>& lane, const Shared& shared) {
    const Extents& size = lane.size;

    if (size.width >= kFloodWidth) {
        return;
    }

    int16_t* reached = shared.scratch.top;
    const int16_t* holds_resource = shared.scratch.below;
    const int16_t* seed = shared.scratch.parent;
    const auto spreads = [&](int32_t cell) { return !lane.terrain[cell] && holds_resource[cell] == 0; };
    const auto beside = [&](int32_t cell, const auto& test) {
        const int32_t x = cell % size.width;
        const int32_t y = cell / size.width;
        return (y > 0 && test(cell - size.width)) || (y < size.height - 1 && test(cell + size.width)) || (x > 0 && test(cell - 1)) || (x < size.width - 1 && test(cell + 1));
    };

    __syncthreads();

    for (int32_t cell = threadIdx.x; cell < size.cells; cell += blockDim.x) {
        const bool seeded = seed[cell] != 0 || beside(cell, [&](int32_t other) { return seed[other] != 0; });
        reached[cell] = seeded ? 1 : 0;
    }

    bool changed = true;

    while (__syncthreads_or(changed)) {
        changed = false;

        for (int32_t cell = threadIdx.x; cell < size.cells; cell += blockDim.x) {
            if (reached[cell] != 0) {
                continue;
            }

            if (beside(cell, [&](int32_t other) { return reached[other] != 0 && spreads(other); })) {
                reached[cell] = 1;
                changed = true;
            }
        }
    }

    for (int32_t slot = threadIdx.x; slot < size.entities; slot += blockDim.x) {
        if ((shared.bits[slot] & kMyBase) == 0) {
            continue;
        }

        const int32_t at = clip(lane.cell_at(slot), 0, size.cells - 1);

        if (!(spreads(at) && reached[at] != 0)) {
            shared.bits[slot] |= kSeparated;
            atomicAdd(&shared.facts->separated, 1);
        }
    }
}

template <typename Slot, typename Cell>
__device__ void recognise_opening(const Lane<Slot, Cell>& lane, const Shared& shared) {
    const Extents& size = lane.size;

    if (size.entities < kSixteenAUnits || shared.facts->active_count != kSixteenAUnits) {
        return;
    }

    for (int32_t slot = threadIdx.x; slot < size.entities; slot += blockDim.x) {
        const int32_t key = lane.order(slot);
        int32_t rank = 0;

        for (int32_t other = 0; other < size.entities; ++other) {
            const int32_t known = lane.order(other);
            rank += known < key || (known == key && other < slot);
        }

        if (rank >= kSixteenAUnits) {
            continue;
        }

        const int32_t at = lane.cell_at(slot);
        const int32_t shown[kSixteenAFields] = {
            lane.unit_type[slot],
            lane.owner[slot],
            astar::floor_mod(at, size.width),
            astar::floor_div(at, size.width),
            lane.hit_points[slot],
            lane.carried[slot]
        };

        for (int field = 0; field < kSixteenAFields; ++field) {
            if (shown[field] != kSixteenA[rank][field]) {
                atomicOr(&shared.facts->opening_differs, 1);
            }
        }
    }
}

template <typename Slot, typename Cell>
__device__ void read_own(const Inputs<Slot, Cell>& in, const Lane<Slot, Cell>& lane, const Shared& shared, int64_t lane_index) {
    const Extents& size = lane.size;
    const int32_t member = threadIdx.x;

    if (member >= size.capacity) {
        return;
    }

    const int32_t slot = lane.own_row(member);

    for (int k = 0; k < kEntryRows; ++k) {
        shared.entries[k][member] = in.entries[k].at(lane_index)[member];
    }

    shared.trained[member] = in.trained.at(lane_index)[member] ? 1 : 0;
    const int32_t born = lane.birth[slot];
    const int32_t harvester = in.harvest[kHarvester].at(lane_index)[member];
    const int32_t base = in.harvest[kHarvestBase].at(lane_index)[member];
    const int32_t base_birth = in.harvest[kHarvestBaseBirth].at(lane_index)[member];
    const int32_t known = clip(base, 0, size.entities - 1);
    const bool kept = harvester != kAbsent && harvester == born && lane.birth[known] == base_birth && (shared.bits[known] & kMyBase) != 0;
    shared.harvest[kHarvester][member] = kept ? harvester : kAbsent;
    shared.harvest[kHarvestBase][member] = kept ? base : kAbsent;
    shared.harvest[kHarvestBaseBirth][member] = kept ? base_birth : kAbsent;
    const int32_t x = shared.x[slot];
    const int32_t y = shared.y[slot];
    const bool builds_barracks = (shared.bits[slot] & kMyWorker) != 0 && lane.pending_kind[slot] == kKindProduce && lane.pending_produced_type[slot] == kTypeBarracks;
    shared.builder_base[member] = static_cast<int16_t>(builds_barracks ? nearest(shared.bits, shared.x, shared.y, lane.birth, kMyBase, x, y, lane.own_row(0), size.capacity) : kAbsent);
    shared.near_enemy[member] = static_cast<int16_t>(nearest(shared.bits, shared.x, shared.y, lane.birth, kEnemy, x, y, 0, size.entities));
}

template <typename Slot, typename Cell>
__device__ bool held(const Lane<Slot, Cell>& lane, const Shared& shared, int32_t member) {
    const int32_t harvester = shared.harvest[kHarvester][member];
    return harvester != kAbsent && harvester == lane.birth[lane.own_row(member)];
}

struct Turn {
    int32_t balance;
    int32_t enemy_balance;
    bool was_separated;
    bool attack_all;
    int32_t opening;
    bool attack_with_combat;
};

template <typename Slot, typename Cell>
__device__ void fight(Walk<Slot, Cell>& walk, const Turn& turn, int32_t slot) {
    const Lane<Slot, Cell>& lane = walk.lane;
    const Shared& shared = walk.shared;
    const Extents& size = lane.size;
    const int32_t entities = size.entities;
    const int32_t ux = walk.x(slot);
    const int32_t uy = walk.y(slot);
    const int32_t type = lane.type_of(slot);
    const int32_t reach = lane.column(type, kAttackRange);
    const int32_t reach_squared = wrapped_mul(reach, reach);
    const int32_t attack_time = lane.turn_column(type, kAttackTime);
    const int32_t dealt = lane.turn_column(type, kAttackDamage);
    Least best[4] = {nothing(entities), nothing(entities), nothing(entities), nothing(entities)};

    for (int32_t other = threadIdx.x; other < entities; other += blockDim.x) {
        if (!walk.is(other, kAlive)) {
            continue;
        }

        const int32_t dx = walk.x(other) - ux;
        const int32_t dy = walk.y(other) - uy;
        const int32_t order = lane.order(other);
        const int32_t after = astar::manhattan(shared.next_x[other], shared.next_y[other], ux, uy);
        offer(best[3], true, after, order, other);

        if (wrapped_add(wrapped_mul(dx, dx), wrapped_mul(dy, dy)) > reach_squared) {
            continue;
        }

        offer(best[2], true, after, order, other);
        const bool leaves = walk.is(other, kMoving) && wrapped_sub(lane.pending_completes_at[other], lane.time) <= attack_time;

        if (leaves) {
            continue;
        }

        offer(best[1], true, walk.distance(other, ux, uy), order, other);
        const int32_t remaining = wrapped_sub(wrapped_sub(lane.hit_points[other], shared.damage[other]), dealt);

        if (remaining <= 0) {
            offer(best[0], true, wrapped_sub(0, remaining), order, other);
        }
    }

    walk.reduce(best);
    const bool strikes = best[1].slot < entities;
    const int32_t target_struck = best[0].slot < entities ? best[0].slot : found(best[1], entities);
    const int32_t leaving = found(best[2], entities);
    const int32_t known = clip(leaving, 0, entities - 1);
    const int32_t ex = shared.next_x[known];
    const int32_t ey = shared.next_y[known];
    const int32_t leaving_reach = lane.column(lane.type_of(known), kAttackRange);
    const bool lands_out_of_reach = wrapped_add(wrapped_mul(ex - ux, ex - ux), wrapped_mul(ey - uy, ey - uy)) > wrapped_mul(leaving_reach, leaving_reach);
    bool fits[kDirections];
    int32_t first_fit = kDirections;

    for (int k = 0; k < kDirections; ++k) {
        const int32_t direction = kFleeOrder[k];
        const int32_t tx = ux + step_x(direction);
        const int32_t ty = uy + step_y(direction);
        const bool farther = astar::manhattan(ex, ey, tx, ty) > astar::manhattan(ux, uy, ex, ey);
        fits[k] = lane.on_board(tx, ty) && (shared.flags[lane.cell_from(tx, ty)] & kEmpty) != 0 && farther;

        if (fits[k] && first_fit == kDirections) {
            first_fit = k;
        }
    }

    const bool flees = first_fit < kDirections;
    const int32_t first = flees ? first_fit : 0;
    const int32_t chosen = ex == ux ? (fits[0] ? 0 : fits[1] ? 1 : first) : ey == uy ? (fits[2] ? 2 : fits[3] ? 3 : first) : first;
    const int32_t refuge = lane.cell_from(ux + step_x(kFleeOrder[chosen]), uy + step_y(kFleeOrder[chosen]));
    const bool ranged = reach > 1;
    const bool retreats = !strikes && leaving != kAbsent && ranged && lands_out_of_reach && flees;
    const int32_t closest = found(best[3], entities);
    const bool roams = best[3].slot < entities && best[2].slot >= entities;
    Facts* facts = shared.facts;
    const int32_t company = facts->company;
    const int32_t divisor = company > 1 ? company : 1;
    const int32_t centre_x = astar::floor_div(facts->company_x_sum, divisor);
    const int32_t centre_y = astar::floor_div(facts->company_y_sum, divisor);
    const bool late = lane.time > max_cycles(size.width) / 2;
    const bool near_centre = astar::manhattan(centre_x, centre_y, ux, uy) <= kCentroidDistance;
    const bool pursues = turn.attack_with_combat || walk.distance_to(closest, ux, uy) <= kChaseDistance || late || (company > 0 && near_centre);
    const bool chases = roams && pursues;
    const bool advances = roams && !pursues && company > 0;
    Fields fields;
    const int32_t kind = strikes || chases ? kCoacAttack : retreats || advances ? kMove : kTrueIdle;
    fields.target = strikes ? target_struck : chases ? closest : kAbsent;
    fields.target_birth = walk.birth_of(fields.target);
    fields.cell = retreats ? refuge : advances ? lane.cell_from(centre_x, centre_y) : kAbsent;

    if (threadIdx.x == 0) {
        put(lane, shared, slot, kind, fields);

        if (strikes) {
            const int32_t damage = wrapped_add(shared.damage[target_struck], dealt);
            shared.damage[target_struck] = damage;

            if (damage >= lane.hit_points[target_struck]) {
                shared.bits[target_struck] &= static_cast<uint16_t>(~kAlive);
            }
        }
    }
}

template <typename Slot, typename Cell>
__device__ void build_barracks(Walk<Slot, Cell>& walk, const Turn& turn) {
    const Lane<Slot, Cell>& lane = walk.lane;
    const Shared& shared = walk.shared;
    const Extents& size = lane.size;
    const int32_t entities = size.entities;
    Facts* facts = shared.facts;
    const int32_t barracks_cost = lane.cost(kTypeBarracks);
    const bool starts = !turn.attack_all && facts->barracks == 0 && wrapped_sub(turn.balance, facts->resources_used) >= wrapped_add(barracks_cost, kBarracksSpare);
    const bool owns = static_cast<int32_t>(threadIdx.x) < size.capacity;
    const int32_t count = walk.members(kMyBase, 0, starts);

    for (int32_t trip = 0; trip < count; ++trip) {
        __syncthreads();

        if (facts->stop != 0) {
            break;
        }

        const int32_t member = shared.members[trip];
        const int32_t base = lane.first + member;
        const int32_t bx = walk.x(base);
        const int32_t by = walk.y(base);
        const bool constructing = __syncthreads_or(owns && shared.builder_base[threadIdx.x] == member);
        Least near[2] = {nothing(entities), nothing(entities)};

        for (int32_t other = threadIdx.x; other < entities; other += blockDim.x) {
            const int32_t distance = walk.distance(other, bx, by);
            offer(near[0], walk.is(other, kAlive), distance, lane.order(other), other);
            offer(near[1], walk.is(other, kFreeWorker), distance, lane.order(other), other);
        }

        walk.reduce(near);
        const int32_t threat = found(near[0], entities);
        const int32_t builder = found(near[1], entities);
        const bool rich = turn.balance > kRich && turn.balance >= turn.enemy_balance;
        const bool takes_longer = wrapped_mul(walk.distance_to(threat, bx, by), lane.turn_column(kTypeWorker, kMoveTime)) > lane.turn_column(kTypeBarracks, kProduceTime);
        const bool safe = walk.is(base, kSeparated) || threat == kAbsent || takes_longer || rich;
        const bool considered = !constructing && safe;
        const int32_t known = clip(builder, 0, entities - 1);
        const int32_t wx = walk.x(known);
        const int32_t wy = walk.y(known);
        Least danger[1] = {nothing(entities)};
        uint32_t resource_near = 0;

        for (int32_t other = threadIdx.x; other < entities; other += blockDim.x) {
            offer(danger[0], walk.is(other, kAlive), walk.distance(other, wx, wy), lane.order(other), other);

            if (!walk.is(other, kResource)) {
                continue;
            }

            for (int k = 0; k < kDirections; ++k) {
                const int32_t direction = kAdjacentOrder[k];

                if (abs(walk.x(other) - (wx + step_x(direction))) <= 1 && abs(walk.y(other) - (wy + step_y(direction))) <= 1) {
                    resource_near |= 1u << k;
                }
            }
        }

        walk.reduce(danger, resource_near);
        const int32_t enemy = found(danger[0], entities);
        const int32_t ek = clip(enemy, 0, entities - 1);
        int32_t site = 0;
        int32_t picked = turn.was_separated ? kFar : 0;
        bool any_fits = false;

        for (int k = 0; k < kDirections; ++k) {
            const int32_t direction = kAdjacentOrder[k];
            const int32_t tx = wx + step_x(direction);
            const int32_t ty = wy + step_y(direction);
            const int32_t at = lane.cell_from(tx, ty);
            const bool fits = lane.on_board(tx, ty) && (resource_near & (1u << k)) == 0 && (shared.flags[at] & kHolding) == 0;
            const int32_t value = fits ? walk.distance(ek, tx, ty) : turn.was_separated ? kFar : 0;
            any_fits |= fits;

            if (k == 0 || (turn.was_separated ? value < picked : value > picked)) {
                picked = value;
                site = at;
            }
        }

        const bool site_found = any_fits && enemy != kAbsent;
        const bool builds = considered && builder != kAbsent && (enemy == kAbsent || walk.distance_to(enemy, wx, wy) > kDangerDistance) && walk.distance(base, wx, wy) <= kBuilderReach && site_found;

        if (threadIdx.x == 0) {
            if (builds) {
                Fields fields;
                fields.cell = site;
                fields.unit_type = kTypeBarracks;
                put(lane, shared, builder, kBuildModified, fields);
                shared.bits[builder] &= static_cast<uint16_t>(~kFreeWorker);
                facts->resources_used = wrapped_add(facts->resources_used, barracks_cost);
            }

            if (considered && builder == kAbsent) {
                facts->stop = 1;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        facts->stop = 0;
    }
}

template <typename Slot, typename Cell>
__device__ void assign_harvesters(Walk<Slot, Cell>& walk, const Turn& turn) {
    const Lane<Slot, Cell>& lane = walk.lane;
    const Shared& shared = walk.shared;
    const Extents& size = lane.size;
    const int32_t entities = size.entities;
    const int32_t capacity = size.capacity;
    Facts* facts = shared.facts;
    bool free_worker = false;

    for (int32_t other = threadIdx.x; other < entities; other += blockDim.x) {
        free_worker |= walk.is(other, kFreeWorker);
    }

    const bool any_free = __syncthreads_or(free_worker);
    const bool starts = !turn.attack_all && any_free && facts->bases > 0 && facts->resources > 0;
    const int32_t member = threadIdx.x;
    const bool owns = member < capacity;

    if (owns) {
        int32_t harvesters = 0;

        for (int32_t other = 0; other < capacity; ++other) {
            harvesters += held(lane, shared, other) && shared.harvest[kHarvestBase][other] - lane.first == member;
        }

        shared.per_base[member] = static_cast<int16_t>(harvesters);
    }

    const int32_t count = walk.members(kMyBase, 0, starts);

    for (int32_t trip = 0; trip < count; ++trip) {
        __syncthreads();

        if (facts->stop != 0) {
            break;
        }

        const int32_t row = owns ? lane.own_row(member) : 0;
        const bool candidate = owns && walk.is(row, kFreeWorker) && !held(lane, shared, member);
        Least resource = nothing(entities);
        Least base = nothing(capacity);

        if (candidate) {
            const int32_t ux = walk.x(row);
            const int32_t uy = walk.y(row);

            for (int32_t other = 0; other < entities; ++other) {
                if (!walk.is(other, kClosestResource)) {
                    continue;
                }

                offer(resource, true, walk.distance(other, ux, uy), lane.order(other), other);
            }

            for (int32_t other = 0; other < capacity; ++other) {
                const int32_t at = lane.own_row(other);

                if (!walk.is(at, kMyBase) || shared.per_base[other] >= kWorkersPerBase) {
                    continue;
                }

                offer(base, true, walk.distance(at, ux, uy), lane.order(at), other);
            }
        }

        const int32_t target = found(resource, entities);
        const int32_t harvested = found(base, capacity);
        const int32_t score = target != kAbsent && harvested != kAbsent ? wrapped_add(base.key, resource.key) : kLast;
        Least worker[1] = {nothing(capacity)};
        offer(worker[0], candidate, score, owns ? lane.order(row) : kLast, member);
        walk.reduce(worker);
        const int32_t chosen = found(worker[0], capacity);
        const bool assigns = chosen != kAbsent && worker[0].key != kLast;

        if (assigns && member == chosen) {
            const int32_t base_slot = lane.first + harvested;
            Fields fields;
            fields.target = target;
            fields.target_birth = walk.birth_of(target);
            fields.base = base_slot;
            fields.base_birth = walk.birth_of(base_slot);
            put(lane, shared, lane.first + member, kHarvest, fields);
            shared.harvest[kHarvester][member] = lane.birth[row];
            shared.harvest[kHarvestBase][member] = base_slot;
            shared.harvest[kHarvestBaseBirth][member] = fields.base_birth;
            shared.per_base[harvested] = static_cast<int16_t>(shared.per_base[harvested] + 1);
        }

        if (threadIdx.x == 0 && !assigns) {
            facts->stop = 1;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        facts->stop = 0;
    }
}

template <typename Slot, typename Cell>
__device__ void send_free_workers(Walk<Slot, Cell>& walk, const Turn& turn) {
    const Lane<Slot, Cell>& lane = walk.lane;
    const Shared& shared = walk.shared;
    const int32_t entities = lane.size.entities;
    Facts* facts = shared.facts;
    const int32_t count = walk.members(kFreeWorker);

    for (int32_t trip = 0; trip < count; ++trip) {
        __syncthreads();
        const int32_t member = shared.members[trip];
        const int32_t slot = lane.first + member;
        const int32_t ux = walk.x(slot);
        const int32_t uy = walk.y(slot);
        Least near[2] = {nothing(entities), nothing(entities)};

        for (int32_t other = threadIdx.x; other < entities; other += blockDim.x) {
            const int32_t distance = walk.distance(other, ux, uy);
            offer(near[0], walk.is(other, kAlive), distance, lane.order(other), other);
            offer(near[1], walk.is(other, kResource), distance, lane.order(other), other);
        }

        walk.reduce(near);
        const int32_t enemy = found(near[0], entities);
        const int32_t resource = found(near[1], entities);
        const bool threatened = enemy != kAbsent && walk.distance_to(enemy, ux, uy) <= kDangerDistance;
        const bool fights = turn.attack_all || !held(lane, shared, member) || threatened;

        if (!fights) {
            if (threadIdx.x == 0) {
                Fields fields;
                fields.target = resource;
                fields.target_birth = walk.birth_of(resource);
                fields.base = shared.harvest[kHarvestBase][member];
                fields.base_birth = shared.harvest[kHarvestBaseBirth][member];
                put(lane, shared, slot, kHarvestReturn, fields);
            }

            continue;
        }

        if (!turn.attack_all) {
            if (threadIdx.x == 0) {
                facts->company = wrapped_add(facts->company, 1);
                facts->company_x_sum = wrapped_add(facts->company_x_sum, ux);
                facts->company_y_sum = wrapped_add(facts->company_y_sum, uy);

                for (int k = 0; k < kHarvestRows; ++k) {
                    shared.harvest[k][member] = kAbsent;
                }
            }

            __syncthreads();
        }

        fight(walk, turn, slot);
    }

    __syncthreads();
}

template <typename Slot, typename Cell>
__device__ void train(const Walk<Slot, Cell>& walk, const Turn& turn, int32_t member, int32_t unit_type) {
    const Lane<Slot, Cell>& lane = walk.lane;
    const Shared& shared = walk.shared;
    Facts* facts = shared.facts;
    const int32_t slot = lane.first + member;
    const int32_t cost = lane.cost(unit_type);
    const bool affords = walk.is(slot, kAvailable) && wrapped_sub(turn.balance, facts->resources_used) >= cost;
    const int32_t ux = walk.x(slot);
    const int32_t uy = walk.y(slot);
    const int32_t enemy = shared.near_enemy[member];
    const int32_t known = clip(enemy, 0, lane.size.entities - 1);
    bool any_fits = false;
    int32_t best = kLowest;
    int32_t site = 0;

    for (int k = 0; k < kDirections; ++k) {
        const int32_t direction = kTrainOrder[k];
        const int32_t tx = ux + step_x(direction);
        const int32_t ty = uy + step_y(direction);
        const int32_t at = lane.cell_from(tx, ty);
        const bool fits = lane.on_board(tx, ty) && (shared.flags[at] & kOpen) != 0;
        const int32_t score = enemy != kAbsent ? -walk.distance(known, tx, ty) : 0;
        const int32_t value = fits ? score : kLowest;

        if (k == 0 || value > best) {
            best = value;
            site = at;
        }

        any_fits |= fits;
    }

    if (affords) {
        facts->resources_used = wrapped_add(facts->resources_used, cost);
    }

    if (affords && any_fits) {
        Fields fields;
        fields.cell = site;
        fields.unit_type = unit_type;
        put(lane, shared, slot, kTrainDirection, fields);
    }
}

template <typename Slot, typename Cell>
__device__ void train_at_barracks(Walk<Slot, Cell>& walk, const Turn& turn) {
    const Lane<Slot, Cell>& lane = walk.lane;
    const Shared& shared = walk.shared;
    const int32_t enough = turn.opening == kSixteenAOpening ? 0 : kHeaviesBeforeRanged;
    const int32_t unit_type = shared.facts->heavies > enough || turn.was_separated ? kTypeRanged : kTypeHeavy;
    const int32_t count = walk.members(kMyBarracks);

    if (threadIdx.x == 0) {
        for (int32_t trip = 0; trip < count; ++trip) {
            train(walk, turn, shared.members[trip], unit_type);
        }
    }

    __syncthreads();
}

template <typename Slot, typename Cell>
__device__ void train_at_bases(Walk<Slot, Cell>& walk, const Turn& turn) {
    const Lane<Slot, Cell>& lane = walk.lane;
    const Shared& shared = walk.shared;
    Facts* facts = shared.facts;

    if (lane.size.width <= kAlwaysTrainingWidth) {
        const int32_t count = walk.members(kMyBase, kSeparated);

        if (threadIdx.x == 0) {
            for (int32_t trip = 0; trip < count; ++trip) {
                train(walk, turn, shared.members[trip], kTypeWorker);
            }
        }

        __syncthreads();
    }

    const int32_t enough = wrapped_mul(kWorkersPerBase, facts->bases);
    const int32_t workers = wrapped_add(facts->my_workers, facts->busy_bases);
    const int32_t count = walk.members(kMyBase);

    if (threadIdx.x == 0) {
        int32_t trained = 0;

        for (int32_t trip = 0; trip < count; ++trip) {
            if (wrapped_add(workers, trained) >= enough && wrapped_sub(turn.balance, facts->resources_used) < kAbundant) {
                break;
            }

            train(walk, turn, shared.members[trip], kTypeWorker);
            ++trained;
        }
    }

    __syncthreads();
}

template <typename Slot, typename Cell>
__device__ void fight_with_combat_units(Walk<Slot, Cell>& walk, const Turn& turn) {
    const Lane<Slot, Cell>& lane = walk.lane;
    const Shared& shared = walk.shared;
    const int32_t count = walk.members(kCombat | kMine | kAvailable);

    for (int32_t trip = 0; trip < count; ++trip) {
        __syncthreads();
        fight(walk, turn, lane.first + shared.members[trip]);
    }

    __syncthreads();
}

struct Unit {
    int32_t birth;
    int32_t cell;
    int32_t type;
    int32_t carried;
    bool present;
    bool available;
};

struct Entry {
    int32_t kind;
    int32_t unit_birth;
    int32_t order;
    int32_t target;
    int32_t target_birth;
    int32_t base;
    int32_t base_birth;
    int32_t cell;
    int32_t unit_type;
    bool trained;
};

struct Desire {
    int32_t kind = kKindAbsent;
    int32_t direction = kAbsent;
    int32_t target_cell = kAbsent;
    int32_t produced_type = kAbsent;
    int32_t claimed = kAbsent;
    int32_t cost = 0;
    int32_t none_duration = 0;
};

__device__ Desire waiting() {
    Desire desire;
    desire.kind = kKindNone;
    desire.none_duration = kWaitOne;
    return desire;
}

struct Usage {
    uint32_t reserved;
    int32_t balance;
    bool other_within;
    bool moves_allowed;
};

struct Row {
    int32_t goal;
    int32_t radius;
    bool draws;
    bool blocked_eligible;
    int32_t options;
    Desire fallback;
};

template <typename Slot, typename Cell>
__device__ Desire production(const Lane<Slot, Cell>& lane, const uint8_t* flags, const Usage& usage, int32_t cell, int32_t direction, int32_t produced_type) {
    const Extents& size = lane.size;
    const int32_t claimed = direction != kAbsent ? neighbour(cell, direction, size.width) : kAbsent;
    const int32_t cost = lane.cost(clip(produced_type, 0, size.types - 1));
    const int32_t reserved = static_cast<int32_t>(usage.reserved);
    const bool allowed = direction != kAbsent && (flags[clip(claimed, 0, size.cells - 1)] & kTaken) == 0 && (reserved == 0 || wrapped_add(cost, reserved) <= usage.balance) && usage.other_within;
    Desire desire;

    if (allowed) {
        desire.kind = kKindProduce;
        desire.direction = direction;
        desire.produced_type = produced_type;
        desire.claimed = claimed;
        desire.cost = cost;
    }

    return desire;
}

template <typename Slot, typename Cell>
__device__ bool completed(const Lane<Slot, Cell>& lane, const Entry& entry, const Unit& unit) {
    const bool carrying = unit.carried > 0;

    if (entry.kind == kMove) {
        return unit.cell == entry.cell;
    }

    if (entry.kind == kCoacAttack) {
        return !lane.exists(entry.target, entry.target_birth);
    }

    if (entry.kind == kHarvest) {
        return carrying ? !lane.exists(entry.base, entry.base_birth) : !lane.exists(entry.target, entry.target_birth);
    }

    if (entry.kind == kTrainDirection) {
        return entry.trained;
    }

    if (entry.kind == kBuildModified) {
        return entry.cell != kAbsent && static_cast<int32_t>(lane.entity_at[clip(entry.cell, 0, lane.size.cells - 1)]) != kAbsent;
    }

    if (entry.kind == kHarvestReturn) {
        return !lane.exists(entry.target, entry.target_birth) && !carrying;
    }

    return false;
}

template <typename Slot, typename Cell>
__device__ int32_t next_cell(const Lane<Slot, Cell>& lane, int32_t slot) {
    const int32_t at = lane.cell_at(slot);
    const bool moving = lane.active(slot) && lane.pending_kind[slot] == kKindMove;
    return moving ? neighbour(at, lane.pending_direction[slot], lane.size.width) : at;
}

template <typename Slot, typename Cell>
__device__ int32_t move_options(const Lane<Slot, Cell>& lane, const uint8_t* flags, const Usage& usage, int32_t cell) {
    const Extents& size = lane.size;
    const int32_t x0 = astar::floor_mod(cell, size.width);
    const int32_t y0 = astar::floor_div(cell, size.width);
    int32_t options = 0;

    for (int32_t direction = 0; direction < kDirections; ++direction) {
        const int32_t x = x0 + step_x(direction);
        const int32_t y = y0 + step_y(direction);
        const bool passes = lane.on_board(x, y) && (flags[y * size.width + x] & kOpen) != 0 && usage.moves_allowed;
        options |= passes ? 1 << direction : 0;
    }

    return options;
}

template <typename Slot, typename Cell>
__device__ Row plan(const Lane<Slot, Cell>& lane, const uint8_t* flags, const Usage& usage, const Entry& entry, const Unit& unit) {
    const Extents& size = lane.size;
    const bool carrying = unit.carried > 0;
    const int32_t target_cell = lane.cell_of(entry.target);
    const int32_t range = lane.column(unit.type, kAttackRange);
    const int32_t range_squared = wrapped_mul(range, range);
    const bool in_range = squared_distance(unit.cell, target_cell, size.width) <= range_squared;
    const int32_t target_next = entry.target != kAbsent ? next_cell(lane, clip(entry.target, 0, size.entities - 1)) : kAbsent;
    const bool coming = squared_distance(unit.cell, target_next, size.width) <= range_squared;
    const int32_t kind = entry.kind;
    const bool attacks = kind == kCoacAttack;
    const bool harvests = kind == kHarvest || kind == kHarvestReturn;
    const bool builds = kind == kBuildModified;
    const bool waits = attacks && !in_range && coming;
    const int32_t harvest_goal = carrying ? lane.cell_of(entry.base) : target_cell;
    const int32_t goal = kind == kMove ? entry.cell : attacks ? target_cell : harvests ? harvest_goal : builds ? entry.cell : kAbsent;
    const int32_t radius = attacks ? range : harvests || builds ? 1 : 0;
    const bool searches = goal != kAbsent && !(attacks && in_range) && !waits;
    Desire fallback;

    if (harvests) {
        const int32_t direction = adjacent_direction(unit.cell, harvest_goal, size.width);

        if (direction != kAbsent) {
            fallback.kind = carrying ? kKindReturn : kKindHarvest;
            fallback.direction = direction;
        }
    } else if (kind == kTrainDirection || builds) {
        fallback = production(lane, flags, usage, unit.cell, adjacent_direction(unit.cell, entry.cell, size.width), entry.unit_type);

        if (fallback.kind == kKindAbsent) {
            fallback = waiting();
        }
    } else if (attacks) {
        if (!in_range) {
            fallback = waiting();
        } else if (target_cell != kAbsent) {
            fallback.kind = kKindAttack;
            fallback.target_cell = target_cell;
        }
    } else if (kind == kTrueIdle) {
        fallback = waiting();
    }

    return Row{searches ? goal : unit.cell, searches ? radius : 0, attacks && searches, builds || attacks, move_options(lane, flags, usage, unit.cell), fallback};
}

__device__ void record_row(const Shared& shared, const Unit& unit, const Row& row, bool standing, bool executes) {
    const int32_t member = threadIdx.x;
    shared.goal[member] = row.goal;
    shared.radius[member] = row.radius;
    shared.claim[member] = row.fallback.claimed;
    shared.cost[member] = row.fallback.cost;
    shared.start[member] = static_cast<int16_t>(unit.cell);
    shared.rank[member] = static_cast<int16_t>(kAbsent);
    shared.selector[member] = kFellBack;
    shared.direction[member] = static_cast<int8_t>(kAbsent);
    shared.options[member] = static_cast<uint8_t>(row.options);
    shared.marks[member] = (executes ? kExecutes : 0) | (standing ? kStanding : 0) | (row.fallback.kind != kKindAbsent ? kEligible : 0) | (row.draws ? kDraws : 0) | (row.blocked_eligible ? kBlockedEligible : 0);

    if (executes) {
        atomicAdd(&shared.facts->queued, 1);
    }

    if (standing) {
        atomicAdd(&shared.facts->standing, 1);
    }
}

__device__ int32_t queue_slot_and_settle(const Shared& shared, int32_t capacity, const Entry& entry, bool executes) {
    const int32_t member = threadIdx.x;
    const int32_t key = executes ? entry.order : kLast;
    int32_t position = 0;
    int32_t settled = 0;

    for (int32_t other = 0; other < capacity; ++other) {
        const uint8_t marks = shared.marks[other];
        const int32_t order = shared.entries[kEntryOrder][other];
        const int32_t other_key = (marks & kExecutes) != 0 ? order : kLast;
        position += other_key < key || (other_key == key && other < member);
        settled += (marks & kStanding) != 0 && order < entry.order;
    }

    shared.queue[position] = static_cast<int16_t>(member);
    return settled;
}

__device__ int32_t drawn_direction(int32_t first, int32_t options) {
    for (int turn = 0; turn < kDirections; ++turn) {
        const int32_t direction = static_cast<int32_t>((static_cast<uint32_t>(first) + static_cast<uint32_t>(turn)) & 3u);

        if (((options >> direction) & 1) == 1) {
            return direction;
        }
    }

    return kAbsent;
}

template <typename Slot, typename Cell>
__device__ void run_ordered_pass(const Lane<Slot, Cell>& lane, const Shared& shared, const Usage& usage) {
    const Extents& size = lane.size;
    astar::Scratch scratch = shared.scratch;
    uint8_t* flags = shared.flags;
    const int32_t queued = shared.facts->queued;
    uint32_t kept_cost = 0;
    int32_t kept_count = 0;
    int32_t calls = 0;

    for (int32_t position = 0; position < queued; ++position) {
        __syncthreads();

        for (int32_t cell = threadIdx.x; cell < size.cells; cell += blockDim.x) {
            scratch.clear(cell, (flags[cell] & kOpen) != 0);
        }

        __syncthreads();

        if (threadIdx.x != 0) {
            continue;
        }

        const int32_t member = shared.queue[position];
        const int32_t start = shared.start[member];
        const uint8_t marks = shared.marks[member];
        const astar::Path path = astar::search(scratch, start, shared.goal[member], shared.radius[member], size.width, size.height);
        const bool stepped = path.step != astar::kAbsent;
        int32_t selector = kFellBack;
        int32_t direction = kAbsent;
        bool eligible = (marks & kEligible) != 0;
        int32_t claim = shared.claim[member];
        int32_t cost = shared.cost[member];

        if (stepped && usage.moves_allowed) {
            selector = kMoved;
            direction = path.step;
            eligible = true;
            claim = neighbour(start, direction, size.width);
            cost = 0;
        } else if (stepped) {
            selector = kBlocked;
            eligible = (marks & kBlockedEligible) != 0;
            claim = kAbsent;
            cost = 0;
        }

        const bool drew = (marks & kDraws) != 0 && !(stepped && usage.moves_allowed);

        if (drew) {
            const int32_t first = lane.draws[clip(calls, 0, size.capacity - 1)];
            selector = kMoved;
            direction = drawn_direction(first, shared.options[member]);
            eligible = true;
            claim = direction != kAbsent ? neighbour(start, direction, size.width) : kAbsent;
            cost = 0;
            ++calls;
        }

        const bool on_board = claim >= 0 && claim < size.cells;
        const bool conflicting = on_board && (flags[claim] & kTaken) != 0;
        const bool overdrawn = cost > 0 && static_cast<int32_t>(usage.reserved + kept_cost + static_cast<uint32_t>(cost)) > usage.balance;
        const bool kept = eligible && !conflicting && !overdrawn;

        if (on_board) {
            flags[claim] = (flags[claim] & ~kOpen) | (kept ? kTaken : 0);
        }

        shared.rank[member] = static_cast<int16_t>(kept ? kept_count : kAbsent);
        shared.selector[member] = static_cast<uint8_t>(selector);
        shared.direction[member] = static_cast<int8_t>(direction);

        if (drew) {
            shared.marks[member] = marks | kDrew;
        }

        if (kept) {
            kept_cost += static_cast<uint32_t>(cost);
            ++kept_count;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        shared.facts->kept_count = kept_count;
    }
}

__device__ Desire decided_desire(const Shared& shared, const Row& row, int32_t& rank) {
    const int32_t member = threadIdx.x;
    rank = shared.rank[member];
    Desire decided;

    if (rank == kAbsent) {
        return decided;
    }

    const int32_t selector = shared.selector[member];

    if (selector == kMoved) {
        const int32_t direction = shared.direction[member];

        if (direction == kAbsent) {
            return waiting();
        }

        decided.kind = kKindMove;
        decided.direction = direction;
        return decided;
    }

    if (selector == kBlocked) {
        return waiting();
    }

    return row.fallback;
}

__device__ void write_commands(const Outputs& out, int64_t lane_index, int32_t capacity, int32_t player, const Desire& decided, int32_t rank, bool filler, int32_t filler_order) {
    using nanorts::whole_act::kCommandFields;
    using nanorts::whole_act::kPlayers;
    const bool kept = rank != kAbsent;
    int32_t command[kCommandFields];
    command[kCommandKind] = kept ? decided.kind : filler ? kKindNone : kKindAbsent;
    command[kCommandDirection] = kept ? decided.direction : kAbsent;
    command[kCommandNoneDuration] = kept && decided.kind == kKindNone ? decided.none_duration : filler ? kWaitOne : 0;
    command[kCommandTargetCell] = kept ? decided.target_cell : kAbsent;
    command[kCommandProducedType] = kept ? decided.produced_type : kAbsent;
    command[kCommandRequestOrder] = kept ? rank : filler ? filler_order : kAbsent;
    int32_t unissued[kCommandFields];
    unissued[kCommandKind] = kKindAbsent;
    unissued[kCommandDirection] = kAbsent;
    unissued[kCommandNoneDuration] = 0;
    unissued[kCommandTargetCell] = kAbsent;
    unissued[kCommandProducedType] = kAbsent;
    unissued[kCommandRequestOrder] = kAbsent;
    int32_t* commands = out.commands + lane_index * kCommandFields * kPlayers * capacity;

    for (int field = 0; field < kCommandFields; ++field) {
        for (int32_t who = 0; who < kPlayers; ++who) {
            commands[(field * kPlayers + who) * capacity + threadIdx.x] = who == player ? command[field] : unissued[field];
        }
    }
}

__device__ void write_carry(const Outputs& out, const Shared& shared, int64_t lane_index, int32_t capacity, const Entry& entry, int32_t settled, bool standing, bool executes) {
    const int32_t member = threadIdx.x;
    const int64_t at = lane_index * capacity + member;
    const int32_t held_fields[kEntryRows] = {entry.kind, entry.unit_birth, settled, entry.target, entry.target_birth, entry.base, entry.base_birth, entry.cell, entry.unit_type};

    for (int k = 0; k < kEntryRows; ++k) {
        out.entries[k][at] = standing ? held_fields[k] : kAbsent;
    }

    out.trained[at] = standing && (entry.trained || (executes && (entry.kind == kTrain || entry.kind == kTrainDirection)));

    for (int k = 0; k < kHarvestRows; ++k) {
        out.harvest[k][at] = shared.harvest[k][member];
    }

    out.drew[at] = (shared.marks[member] & kDrew) != 0;
}

template <typename Slot, typename Cell>
__device__ void translate(const Lane<Slot, Cell>& lane, const Shared& shared, const Outputs& out, int64_t lane_index) {
    const Extents& size = lane.size;
    const int32_t capacity = size.capacity;
    const int32_t member = threadIdx.x;
    const bool owns = member < capacity;
    Facts* facts = shared.facts;
    const int32_t own = gathered(lane.player, nanorts::whole_act::kPlayers);
    const int32_t other = gathered(1 - lane.player, nanorts::whole_act::kPlayers);
    const auto within = [&](int32_t who) { return facts->reserved[who] == 0u || static_cast<int32_t>(facts->reserved[who]) <= lane.balance(who); };
    const Usage usage{facts->reserved[own], lane.balance(own), within(other), within(own) && within(other)};
    Unit unit{kAbsent, kAbsent, 0, 0, false, false};
    Entry entry{kNoAction, kAbsent, kAbsent, kAbsent, kAbsent, kAbsent, kAbsent, kAbsent, kAbsent, false};
    bool standing = false;
    bool executes = false;
    Row row{};

    if (owns) {
        const int32_t slot = lane.own_row(member);
        unit.birth = lane.birth[slot];
        unit.present = unit.birth != kAbsent;
        unit.cell = lane.cell_at(slot);
        unit.type = lane.type_of(slot);
        unit.carried = lane.carried[slot];
        unit.available = unit.present && lane.pending_kind[slot] == kKindAbsent;
        entry = Entry{shared.entries[kEntryKind][member], shared.entries[kEntryUnitBirth][member], shared.entries[kEntryOrder][member], shared.entries[kEntryTarget][member], shared.entries[kEntryTargetBirth][member], shared.entries[kEntryBase][member], shared.entries[kEntryBaseBirth][member], shared.entries[kEntryCell][member], shared.entries[kEntryUnitType][member], shared.trained[member] != 0};
        const bool alive = entry.kind != kNoAction && entry.unit_birth == unit.birth;
        standing = alive && !completed(lane, entry, unit);
        executes = standing && unit.available;
        row = plan(lane, shared.flags, usage, entry, unit);
        record_row(shared, unit, row, standing, executes);
    }

    __syncthreads();
    const int32_t settled = owns ? queue_slot_and_settle(shared, capacity, entry, executes) : 0;

    run_ordered_pass(lane, shared, usage);

    Desire decided;
    int32_t rank = kAbsent;
    bool filler = false;

    if (owns) {
        decided = decided_desire(shared, row, rank);
        filler = unit.available && rank == kAbsent;

        if (filler) {
            shared.marks[member] |= kFiller;
        }
    }

    __syncthreads();

    if (!owns) {
        return;
    }

    int32_t filler_rank = 0;

    for (int32_t slot = 0; slot < capacity; ++slot) {
        filler_rank += (shared.marks[slot] & kFiller) != 0 && lane.birth[lane.own_row(slot)] < unit.birth;
    }

    write_commands(out, lane_index, capacity, lane.player, decided, rank, filler, wrapped_add(facts->kept_count, filler_rank));
    write_carry(out, shared, lane_index, capacity, entry, settled, standing, executes);

    if (member == 0) {
        out.next_order[lane_index] = facts->standing;
    }
}

template <typename Slot, typename Cell>
__global__ void __launch_bounds__(nanorts::whole_act::kMaxThreads) act_kernel(Inputs<Slot, Cell> in, Outputs out, Extents size) {
    extern __shared__ uint64_t memory[];
    const int64_t lane_index = blockIdx.x;
    const Lane<Slot, Cell> lane(in, lane_index, size);
    const Shared shared = Shared::carve(memory, size);
    Facts* facts = shared.facts;

    clear_lane(lane, shared, in.next_order.at(lane_index)[0]);
    __syncthreads();
    read_slots(lane, shared);
    __syncthreads();
    flag_cells(lane, shared);
    read_targets(lane, shared);
    __syncthreads();
    read_own(in, lane, shared, lane_index);
    cut_off_bases(lane, shared);
    const int32_t carried_opening = in.opening.at(lane_index)[0];

    if (carried_opening == kUnseen) {
        recognise_opening(lane, shared);
    }

    __syncthreads();

    Turn turn;
    turn.balance = lane.balance(lane.player);
    turn.enemy_balance = lane.balance(1 - lane.player);
    turn.was_separated = in.was_separated.at(lane_index)[0] || facts->separated > 0;
    turn.attack_all = in.attack_all.at(lane_index)[0] || (turn.balance == 0 && facts->resources == 0);
    const bool sixteen_a = size.entities >= kSixteenAUnits && facts->active_count == kSixteenAUnits && facts->opening_differs == 0 && lane.resources[0] == kSixteenABalance && lane.resources[1] == kSixteenABalance;
    turn.opening = carried_opening != kUnseen ? carried_opening : sixteen_a ? kSixteenAOpening : kDefaultOpening;
    turn.attack_with_combat = wrapped_sub(facts->my_combat_cost, kCombatLead) > facts->other_combat_cost;

    Walk<Slot, Cell> walk{lane, shared, 0};
    build_barracks(walk, turn);
    assign_harvesters(walk, turn);
    send_free_workers(walk, turn);
    train_at_barracks(walk, turn);
    train_at_bases(walk, turn);
    fight_with_combat_units(walk, turn);

    translate(lane, shared, out, lane_index);

    if (threadIdx.x == 0) {
        out.was_separated[lane_index] = turn.was_separated;
        out.attack_all[lane_index] = turn.attack_all;
        out.opening[lane_index] = turn.opening;
    }
}

template <typename Slot, typename Cell>
ffi::Error launch(cudaStream_t stream, nanorts::whole_act::Operands& operands, const Outputs& outputs, const Extents& size, int64_t lanes) {
    Inputs<Slot, Cell> in{};

    if (auto e = nanorts::whole_act::bind_state(operands, size, in.state); e.failure()) {
        return e;
    }

    const int64_t entities = size.entities;
    const int64_t capacity = size.capacity;

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

    if (auto e = operands.next(capacity, in.draws); e.failure()) {
        return e;
    }

    for (int k = 0; k < kEntryRows; ++k) {
        if (auto e = operands.next(capacity, in.entries[k]); e.failure()) {
            return e;
        }
    }

    if (auto e = operands.next(capacity, in.trained); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.next_order); e.failure()) {
        return e;
    }

    for (int k = 0; k < kHarvestRows; ++k) {
        if (auto e = operands.next(capacity, in.harvest[k]); e.failure()) {
            return e;
        }
    }

    if (auto e = operands.next(int64_t{1}, in.was_separated); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.attack_all); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.opening); e.failure()) {
        return e;
    }

    const int threads = nanorts::whole_act::threads_for(size.capacity);
    const int64_t shared = shared_bytes(size.cells, size.capacity, size.entities);
    act_kernel<Slot, Cell><<<static_cast<unsigned>(lanes), threads, static_cast<size_t>(shared), stream>>>(in, outputs, size);
    return nanorts::launched();
}

constexpr int64_t kExtraOperands = 6;
constexpr int64_t kReadOperands = nanorts::whole_act::kStateOperands + 2 + kExtraOperands;
constexpr int64_t kCarryLeaves = kEntryRows + 2 + kHarvestRows + 3;
constexpr int64_t kHitPointsOperand = nanorts::whole_act::kStateOperands + 2;
constexpr int64_t kTargetCellOperand = kHitPointsOperand + 1;
constexpr int64_t kTurnTableOperand = kHitPointsOperand + 4;

ffi::Error Act(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets results, std::string_view build_id) {
    using ffi::DataType;
    constexpr std::string_view kTarget = "nanorts_coac_act";

    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (args.size() != kReadOperands + kCarryLeaves || results.size() != kCarryLeaves + 2) {
        return ffi::Error::InvalidArgument(
            "nanorts_coac_act takes the State leaves, the unit tables, the player, the draws "
            "and the carry, and returns the carry, the Commands and the slots that drew");
    }

    auto commands = results.get<ffi::AnyBuffer>(kCarryLeaves);

    if (!commands.has_value()) {
        return commands.error();
    }

    auto drew = results.get<ffi::AnyBuffer>(kCarryLeaves + 1);

    if (!drew.has_value()) {
        return drew.error();
    }

    ffi::AnyBuffer table_rows = *args.get<ffi::AnyBuffer>(nanorts::whole_act::kStateOperands);
    ffi::AnyBuffer turn_rows = *args.get<ffi::AnyBuffer>(kTurnTableOperand);
    const auto issued = (*commands)->dimensions();
    const auto table = table_rows.dimensions();
    const auto turned = turn_rows.dimensions();

    if ((*commands)->element_type() != ffi::S32 || issued.size() != 4 || issued[1] != nanorts::whole_act::kCommandFields || issued[2] != nanorts::whole_act::kPlayers || table.size() != 3 || table[2] != kTableColumns || table_rows.element_type() != ffi::S32 || turned.size() != 3 || turned[1] != table[1] || turned[2] != kTurnColumns || turn_rows.element_type() != ffi::S32) {
        return ffi::Error::InvalidArgument(
            "the Commands are `[lanes, 6, 2, C]` int32 and the unit tables `[1, T, 6]` and "
            "`[1, T, 4]` int32");
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

    if (size.cells <= 0 || size.capacity <= 0 || size.neutral < 0 || size.types <= kTypeRanged || shared > nanorts::whole_act::kSharedLimit || nanorts::whole_act::threads_for(size.capacity) > nanorts::whole_act::kMaxThreads) {
        return ffi::Error::InvalidArgument(
            "a table holding every unit type coacAI names, over a lane whose act fits one "
            "block's threads and shared memory");
    }

    auto hit_points = operands.at(kHitPointsOperand);
    auto target_cell = operands.at(kTargetCellOperand);

    if (!hit_points.has_value()) {
        return hit_points.error();
    }

    if (!target_cell.has_value()) {
        return target_cell.error();
    }

    if (hit_points->element_type() != DataType::S16 || target_cell->element_type() != cell || (*drew)->element_type() != DataType::PRED || static_cast<int64_t>((*drew)->size_bytes()) != lanes * size.capacity) {
        return ffi::Error::InvalidArgument(
            "hit points are int16, pending targets share the cells' dtype, and the slots that "
            "drew are one bool row per lane");
    }

    Outputs outputs{};
    int64_t result = 0;
    int64_t operand = kReadOperands;
    const auto bind = [&](int64_t elements, auto*& into) { return nanorts::whole_act::carry_result(operands, results, result++, operand++, lanes, elements, into); };

    for (int k = 0; k < kEntryRows; ++k) {
        if (auto e = bind(size.capacity, outputs.entries[k]); e.failure()) {
            return e;
        }
    }

    if (auto e = bind(size.capacity, outputs.trained); e.failure()) {
        return e;
    }

    if (auto e = bind(1, outputs.next_order); e.failure()) {
        return e;
    }

    for (int k = 0; k < kHarvestRows; ++k) {
        if (auto e = bind(size.capacity, outputs.harvest[k]); e.failure()) {
            return e;
        }
    }

    if (auto e = bind(1, outputs.was_separated); e.failure()) {
        return e;
    }

    if (auto e = bind(1, outputs.attack_all); e.failure()) {
        return e;
    }

    if (auto e = bind(1, outputs.opening); e.failure()) {
        return e;
    }

    outputs.commands = static_cast<int32_t*>((*commands)->untyped_data());
    outputs.drew = static_cast<bool*>((*drew)->untyped_data());

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

const nanorts::Kernels nanorts_coac_act_kernels("nanorts_coac_act", {nanorts::kernel(act_kernel<int16_t, int16_t>), nanorts::kernel(act_kernel<int16_t, int32_t>), nanorts::kernel(act_kernel<int32_t, int16_t>), nanorts::kernel(act_kernel<int32_t, int32_t>)});

}

NANORTS_HANDLER(nanorts_coac_act, Act,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .RemainingArgs()
                    .RemainingRets()
                    .Attr<std::string_view>("build_id"));
