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
constexpr int64_t kLaneBytes = 96;
constexpr int64_t kBytesPerSlot = 33;

constexpr int32_t kAbsent = -1;
constexpr int32_t kLast = 2147483647;
constexpr int kDirections = 4;
constexpr int32_t kFillerDuration = 10;

constexpr int64_t kWorkerRush = 0;
constexpr int64_t kBarracksRush = 1;
constexpr int64_t kWorkerDefense = 2;
constexpr int64_t kBarracksDefense = 3;

constexpr int32_t kNoAction = -1;
constexpr int32_t kAttack = 1;
constexpr int32_t kHarvest = 2;
constexpr int32_t kTrain = 3;
constexpr int32_t kBuild = 4;
constexpr int32_t kTrainDirection = 6;

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

constexpr int kCanHarvest = 0;
constexpr int kIsResource = 1;
constexpr int kIsStockpile = 2;
constexpr int kCanAttack = 3;
constexpr int kAttackRange = 4;
constexpr int kCost = 5;
constexpr int kTableColumns = 6;

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

constexpr int32_t kFellBack = 0;
constexpr int32_t kBlocked = 1;
constexpr int32_t kMoved = 2;

constexpr uint8_t kOpen = 1;
constexpr uint8_t kTaken = 2;
constexpr uint8_t kEmpty = 4;

constexpr unsigned long long kNoUnit = ~0ull;

__host__ __device__ constexpr bool trains_at_barracks(int64_t code) {
    return code == kBarracksRush || code == kBarracksDefense;
}

constexpr int64_t shared_bytes(int64_t cells, int64_t capacity) {
    return kBytesPerCell * cells + kLaneBytes + kBytesPerSlot * capacity;
}

constexpr uint8_t kExecutes = 1;
constexpr uint8_t kStanding = 2;
constexpr uint8_t kFiller = 4;
constexpr uint8_t kEligible = 8;

template <typename Slot, typename Cell>
struct Inputs {
    StateRows<Slot, Cell> state;
    Lanes<int32_t> table;
    Lanes<int32_t> player;
    Lanes<int32_t> entries[kEntryRows];
    Lanes<bool> trained;
    Lanes<int32_t> next_order;
};

struct Outputs {
    int32_t* entries[kEntryRows];
    bool* trained;
    int32_t* next_order;
    int32_t* commands;
};

struct Strategy {
    int32_t code;
    int32_t trained;
};

struct Facts {
    unsigned long long first_worker = kNoUnit;
    unsigned long long second_worker = kNoUnit;
    unsigned long long last_base = 0ull;
    int32_t any_base = 0;
    int32_t any_barracks = 0;
    int32_t workers_held = 0;
    int32_t base_cell = kAbsent;
    uint32_t reserved[nanorts::whole_act::kPlayers] = {0u, 0u};
    int32_t builds = 0;
    int32_t desired = 0;
    int32_t visit = kLast;
    int32_t site = kAbsent;
    int32_t queued = 0;
    int32_t kept_count = 0;
    int32_t standing = 0;
};

static_assert(sizeof(Facts) <= kLaneBytes, "the lane's facts outgrew their bytes");

struct Slots {
    int32_t* birth;
    int32_t* order;
    int32_t* goal;
    int32_t* radius;
    int32_t* claim;
    int32_t* cost;
    int16_t* start;
    int16_t* queue;
    int16_t* rank;
    uint8_t* marks;
    uint8_t* selector;
    int8_t* direction;

    __device__ static Slots carve(char* base, int capacity) {
        Slots slots;
        int32_t* words = reinterpret_cast<int32_t*>(base);
        slots.birth = words;
        slots.order = words + capacity;
        slots.goal = words + 2 * capacity;
        slots.radius = words + 3 * capacity;
        slots.claim = words + 4 * capacity;
        slots.cost = words + 5 * capacity;
        int16_t* halves = reinterpret_cast<int16_t*>(words + 6 * capacity);
        slots.start = halves;
        slots.queue = halves + capacity;
        slots.rank = halves + 2 * capacity;
        uint8_t* bytes = reinterpret_cast<uint8_t*>(halves + 3 * capacity);
        slots.marks = bytes;
        slots.selector = bytes + capacity;
        slots.direction = reinterpret_cast<int8_t*>(bytes + 2 * capacity);
        return slots;
    }
};

struct Unit {
    int32_t birth;
    int32_t cell;
    int32_t type;
    int32_t carried;
    int32_t rank;
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

struct Fields {
    int32_t target = kAbsent;
    int32_t target_birth = kAbsent;
    int32_t base = kAbsent;
    int32_t base_birth = kAbsent;
    int32_t cell = kAbsent;
    int32_t unit_type = kAbsent;
};

struct Closest {
    int32_t slot = kAbsent;
    int32_t birth = kAbsent;
    int32_t distance = kLast;
    bool found = false;
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

__device__ unsigned long long earliest(int32_t birth, int32_t slot) {
    return static_cast<unsigned long long>(static_cast<uint32_t>(birth)) << 32 | static_cast<uint32_t>(slot);
}

__device__ unsigned long long latest(int32_t birth, int32_t slot) {
    return static_cast<unsigned long long>(static_cast<uint32_t>(birth)) << 32 | (0xffffffffu - static_cast<uint32_t>(slot));
}

__device__ int32_t clip(int32_t value, int32_t low, int32_t high) {
    return value < low ? low : value > high ? high : value;
}

__device__ int32_t gathered(int32_t index, int32_t size) {
    return clip(index < 0 ? index + size : index, 0, size - 1);
}

__device__ int32_t step_x(int32_t direction) {
    const int32_t known = clip(direction, 0, kDirections - 1);
    return known == 1 ? 1 : known == 3 ? -1 : 0;
}

__device__ int32_t step_y(int32_t direction) {
    const int32_t known = clip(direction, 0, kDirections - 1);
    return known == 0 ? -1 : known == 2 ? 1 : 0;
}

__device__ int32_t wrapped_add(int32_t left, int32_t right) {
    return static_cast<int32_t>(static_cast<uint32_t>(left) + static_cast<uint32_t>(right));
}

__device__ int32_t neighbour(int32_t cell, int32_t direction, int32_t width) {
    return wrapped_add(cell, step_y(direction) * width + step_x(direction));
}

__device__ int32_t manhattan(int32_t cell, int32_t other, int32_t width) {
    return abs(astar::floor_mod(cell, width) - astar::floor_mod(other, width)) + abs(astar::floor_div(cell, width) - astar::floor_div(other, width));
}

__device__ int32_t squared_distance(int32_t cell, int32_t other, int32_t width) {
    const int32_t dx = astar::floor_mod(cell, width) - astar::floor_mod(other, width);
    const int32_t dy = astar::floor_div(cell, width) - astar::floor_div(other, width);
    return dx * dx + dy * dy;
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
    int32_t player;
    Extents size;

    __device__ Lane(const Inputs<Slot, Cell>& in, int64_t lane, const Extents& extents) : birth(in.state.birth.at(lane)), unit_type(in.state.unit_type.at(lane)), owner(in.state.owner.at(lane)), cell(in.state.cell.at(lane)), carried(in.state.carried.at(lane)), pending_kind(in.state.pending_kind.at(lane)), pending_direction(in.state.pending_direction.at(lane)), pending_produced_type(in.state.pending_produced_type.at(lane)), entity_at(in.state.entity_at.at(lane)), resources(in.state.resources.at(lane)), terrain(in.state.terrain.at(lane)), table(in.table.at(lane)), player(in.player.at(lane)[0]), size(extents) {}

    __device__ bool active(int32_t slot) const {
        return birth[slot] != kAbsent;
    }

    __device__ int32_t type_of(int32_t slot) const {
        return clip(unit_type[slot], 0, size.types - 1);
    }

    __device__ int32_t at(int32_t type, int column) const {
        return table[type * kTableColumns + column];
    }

    __device__ int32_t cost(int32_t type) const {
        return at(type, kCost);
    }

    __device__ int32_t cell_at(int32_t slot) const {
        return static_cast<int32_t>(cell[slot]);
    }

    __device__ bool enemy(int32_t slot) const {
        const int32_t holder = owner[slot];
        return holder != kAbsent && holder != player;
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
};

__device__ void consider(Closest& best, bool candidate, int32_t slot, int32_t born, int32_t distance) {
    if (candidate && (!best.found || distance < best.distance || (distance == best.distance && born < best.birth))) {
        best = Closest{slot, born, distance, true};
    }
}

template <typename Slot, typename Cell>
__device__ void closest_units(const Lane<Slot, Cell>& lane, const Unit& unit, bool worker, Closest& enemy, Closest& resource, Closest& stockpile) {
    if (!unit.present) {
        return;
    }

    for (int32_t slot = 0; slot < lane.size.entities; ++slot) {
        const int32_t born = lane.birth[slot];

        if (born == kAbsent) {
            continue;
        }

        const int32_t type = lane.type_of(slot);
        const bool is_enemy = lane.enemy(slot);
        const bool is_resource = worker && lane.at(type, kIsResource) != 0;
        const bool is_stockpile = worker && lane.at(type, kIsStockpile) != 0 && lane.owner[slot] == lane.player;

        if (!is_enemy && !is_resource && !is_stockpile) {
            continue;
        }

        const int32_t distance = manhattan(unit.cell, lane.cell_at(slot), lane.size.width);
        consider(enemy, is_enemy, slot, born, distance);
        consider(resource, is_resource, slot, born, distance);
        consider(stockpile, is_stockpile, slot, born, distance);
    }
}

__device__ void put(Entry& entry, const Unit& unit, bool chosen, bool last, int32_t kind, const Fields& fields, uint32_t next_order, int32_t capacity) {
    if (!chosen) {
        return;
    }

    const bool known = entry.kind != kNoAction && entry.unit_birth == unit.birth;

    if (!known) {
        entry.order = static_cast<int32_t>(next_order + static_cast<uint32_t>(last ? capacity : unit.rank));
    }

    entry.kind = kind;
    entry.unit_birth = unit.birth;
    entry.target = fields.target;
    entry.target_birth = fields.target_birth;
    entry.base = fields.base;
    entry.base_birth = fields.base_birth;
    entry.cell = fields.cell;
    entry.unit_type = fields.unit_type;
    entry.trained = false;
}

template <typename Slot, typename Cell>
__device__ int32_t building_site(const Lane<Slot, Cell>& lane, const uint8_t* flags, Facts* facts, int32_t reserved) {
    const Extents& size = lane.size;
    const int32_t longest = size.height > size.width ? size.height : size.width;
    __syncthreads();

    if (threadIdx.x == 0) {
        facts->visit = kLast;
        facts->site = kAbsent;
    }

    __syncthreads();
    const int32_t x0 = astar::floor_mod(facts->desired, size.width);
    const int32_t y0 = astar::floor_div(facts->desired, size.width);
    const auto visit = [&](int32_t cell) {
        const int32_t dx = cell % size.width - x0;
        const int32_t dy = cell / size.width - y0;
        const int32_t ring = abs(dx) > abs(dy) ? abs(dx) : abs(dy);

        if (ring == 0) {
            return kLast;
        }

        const int32_t side = dy == -ring ? 0 : dx == ring ? 1 : dy == ring ? 2 : 3;
        const int32_t along = (side == 0 || side == 2 ? dx : dy) + ring;
        return (ring * 4 + side) * (2 * longest + 1) + along;
    };

    const auto open = [&](int32_t cell, int32_t value) { return (flags[cell] & kEmpty) != 0 && cell != reserved && value != kLast; };

    for (int32_t cell = threadIdx.x; cell < size.cells; cell += blockDim.x) {
        const int32_t value = visit(cell);

        if (open(cell, value)) {
            atomicMin(&facts->visit, value);
        }
    }

    __syncthreads();

    for (int32_t cell = threadIdx.x; cell < size.cells; cell += blockDim.x) {
        const int32_t value = visit(cell);

        if (open(cell, value) && value == facts->visit) {
            facts->site = cell;
        }
    }

    __syncthreads();
    return facts->visit == kLast ? kAbsent : facts->site;
}

template <typename Slot, typename Cell>
__device__ int32_t build(const Lane<Slot, Cell>& lane, const uint8_t* flags, Facts* facts, Entry& entry, const Unit& unit, bool builder, int32_t type, int32_t reserved, uint32_t next_order) {
    const bool building = entry.kind == kBuild && entry.unit_birth == unit.birth && entry.unit_type == type;
    const bool builds = builder && !building;
    __syncthreads();

    if (threadIdx.x == 0) {
        facts->builds = 0;
    }

    __syncthreads();

    if (builds) {
        facts->builds = 1;
        facts->desired = unit.cell;
    }

    __syncthreads();
    const bool any = facts->builds != 0;
    const int32_t site = any ? building_site(lane, flags, facts, reserved) : kAbsent;
    Fields fields;
    fields.cell = site;
    fields.unit_type = type;
    put(entry, unit, builds, false, kBuild, fields, next_order, lane.size.capacity);
    return any && site != kAbsent ? site : reserved;
}

__device__ bool rush_harvest(Entry& entry, const Unit& unit, bool harvester, const Closest& resource, const Closest& stockpile, uint32_t next_order, int32_t capacity) {
    const bool carrying = unit.carried > 0;
    const bool harvesting = entry.kind == kHarvest && entry.unit_birth == unit.birth;
    const bool same_base = entry.base == stockpile.slot && entry.base_birth == stockpile.birth;
    const bool same_target = entry.target == resource.slot && entry.target_birth == resource.birth;
    const bool occupied = carrying ? stockpile.found : stockpile.found && resource.found;
    const bool unchanged = harvesting && same_base && (carrying || same_target);
    Fields fields;
    fields.target = carrying ? kAbsent : resource.slot;
    fields.target_birth = carrying ? kAbsent : resource.birth;
    fields.base = stockpile.slot;
    fields.base_birth = stockpile.birth;
    put(entry, unit, harvester && occupied && !unchanged, false, kHarvest, fields, next_order, capacity);
    return harvester && !occupied;
}

__device__ void defense_harvest(Entry& entry, const Unit& unit, bool harvester, const Closest& resource, const Closest& stockpile, uint32_t next_order, int32_t capacity) {
    const bool unchanged = entry.kind == kHarvest && entry.unit_birth == unit.birth && entry.target == resource.slot && entry.target_birth == resource.birth && entry.base == stockpile.slot && entry.base_birth == stockpile.birth;
    Fields fields;
    fields.target = resource.slot;
    fields.target_birth = resource.birth;
    fields.base = stockpile.slot;
    fields.base_birth = stockpile.birth;
    put(entry, unit, harvester && resource.found && stockpile.found && !unchanged, false, kHarvest, fields, next_order, capacity);
}

__device__ Fields attacking(const Closest& enemy) {
    Fields fields;
    fields.target = enemy.slot;
    fields.target_birth = enemy.birth;
    return fields;
}

template <typename Slot, typename Cell>
__device__ void defend(const Lane<Slot, Cell>& lane, const Facts* facts, Entry& entry, const Unit& unit, bool defender, const Closest& enemy, uint32_t next_order) {
    const int32_t reach = lane.size.height / 2;
    const int32_t base_distance = facts->any_base != 0 ? manhattan(unit.cell, facts->base_cell, lane.size.width) : 0;
    const bool engages = enemy.found && (enemy.distance < reach || base_distance < reach);
    Fields fields;
    fields.target = engages ? enemy.slot : kAbsent;
    fields.target_birth = engages ? enemy.birth : kAbsent;
    put(entry, unit, defender, false, kAttack, fields, next_order, lane.size.capacity);
}

template <typename Slot, typename Cell>
__device__ void decide(const Lane<Slot, Cell>& lane, const uint8_t* flags, Facts* facts, const Strategy& strategy, Entry& entry, const Unit& unit, const Closest& enemy, const Closest& resource, const Closest& stockpile, uint32_t next_order) {
    const int32_t capacity = lane.size.capacity;
    const uint32_t step = static_cast<uint32_t>(capacity) + 1u;
    uint32_t next = next_order;
    const int32_t resources = lane.balance(lane.player);
    const bool worker = unit.present && lane.at(unit.type, kCanHarvest) != 0;
    const bool melee = unit.present && lane.at(unit.type, kCanAttack) != 0 && lane.at(unit.type, kCanHarvest) == 0 && unit.available;
    const unsigned long long key = earliest(unit.birth, threadIdx.x);
    const bool has_worker = facts->first_worker != kNoUnit;
    const bool builds_base = facts->any_base == 0 && resources >= lane.cost(kTypeBase);
    const bool base_builder = worker && key == facts->first_worker && builds_base;
    const unsigned long long after_base = builds_base && has_worker ? facts->second_worker : facts->first_worker;
    const bool rush = strategy.code == kWorkerRush || strategy.code == kBarracksRush;
    const bool barracks = trains_at_barracks(strategy.code);

    if (barracks) {
        Fields worker_type;
        worker_type.unit_type = kTypeWorker;
        put(entry, unit, unit.present && unit.type == kTypeBase && unit.available && facts->workers_held < 1 && resources >= lane.cost(kTypeWorker), false, kTrain, worker_type, next, capacity);
        next += step;
        Fields trained_type;
        trained_type.unit_type = strategy.trained;
        put(entry, unit, unit.present && unit.type == kTypeBarracks && unit.available && resources >= lane.cost(strategy.trained), false, kTrain, trained_type, next, capacity);
        next += step;
    } else {
        Fields worker_type;
        worker_type.unit_type = kTypeWorker;
        put(entry, unit, unit.present && unit.type == kTypeBase && unit.available && resources >= lane.cost(kTypeWorker), false, kTrain, worker_type, next, capacity);
        next += step;
    }

    if (rush) {
        put(entry, unit, melee && enemy.found, false, kAttack, attacking(enemy), next, capacity);
    } else {
        defend(lane, facts, entry, unit, melee, enemy, next);
    }

    next += step;
    const int32_t reserved = build(lane, flags, facts, entry, unit, base_builder, kTypeBase, kAbsent, next);
    next += step;

    if (!barracks) {
        const bool harvester = worker && !base_builder && key == after_base;

        if (rush) {
            const bool idle = rush_harvest(entry, unit, harvester, resource, stockpile, next, capacity);
            next += step;
            const bool free = (worker && !base_builder && !harvester) || idle;
            put(entry, unit, free && enemy.found, idle, kAttack, attacking(enemy), next, capacity);
        } else {
            defense_harvest(entry, unit, harvester, resource, stockpile, next, capacity);
            next += step;
            defend(lane, facts, entry, unit, worker && !base_builder && !harvester, enemy, next);
        }

        return;
    }

    const int32_t used = builds_base && has_worker ? lane.cost(kTypeBase) : 0;
    const bool builds_barracks = facts->any_barracks == 0 && resources >= wrapped_add(lane.cost(kTypeBarracks), used);
    const bool barracks_builder = worker && !base_builder && key == after_base && builds_barracks;
    build(lane, flags, facts, entry, unit, barracks_builder, kTypeBarracks, reserved, next);
    next += step;
    const bool free = worker && !base_builder && !barracks_builder;

    if (rush) {
        const bool idle = rush_harvest(entry, unit, free, resource, stockpile, next, capacity);
        next += step;
        put(entry, unit, idle && enemy.found, false, kAttack, attacking(enemy), next, capacity);
    } else {
        defense_harvest(entry, unit, free, resource, stockpile, next, capacity);
    }
}

struct Usage {
    uint32_t reserved;
    int32_t balance;
    bool other_within;
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
__device__ int32_t training_direction(const Lane<Slot, Cell>& lane, const uint8_t* flags, int32_t cell, int32_t trained) {
    const Extents& size = lane.size;
    const int32_t x0 = astar::floor_mod(cell, size.width);
    const int32_t y0 = astar::floor_div(cell, size.width);
    const bool harvests = lane.at(clip(trained, 0, size.types - 1), kCanHarvest) != 0;
    int32_t closest[kDirections] = {kLast, kLast, kLast, kLast};
    bool any_near = false;

    for (int32_t slot = 0; slot < size.entities; ++slot) {
        if (!lane.active(slot)) {
            continue;
        }

        const bool near = harvests ? lane.at(lane.type_of(slot), kIsResource) != 0 : lane.enemy(slot);

        if (!near) {
            continue;
        }

        any_near = true;
        const int32_t other = lane.cell_at(slot);
        const int32_t ex = astar::floor_mod(other, size.width);
        const int32_t ey = astar::floor_div(other, size.width);

        for (int32_t direction = 0; direction < kDirections; ++direction) {
            const int32_t distance = abs(x0 + step_x(direction) - ex) + abs(y0 + step_y(direction) - ey);

            if (distance < closest[direction]) {
                closest[direction] = distance;
            }
        }
    }

    int32_t best = 0;
    int32_t best_score = 0;
    bool any_open = false;

    for (int32_t direction = 0; direction < kDirections; ++direction) {
        const int32_t x = x0 + step_x(direction);
        const int32_t y = y0 + step_y(direction);
        const bool open = x >= 0 && x < size.width && y >= 0 && y < size.height && (flags[y * size.width + x] & kOpen) != 0;
        const int32_t score = open ? (any_near ? -closest[direction] : 0) : -kLast;

        if (direction == 0 || score > best_score) {
            best = direction;
            best_score = score;
        }

        any_open |= open;
    }

    return any_open ? best : kAbsent;
}

struct Row {
    int32_t goal;
    int32_t radius;
    Desire fallback;
};

template <typename Slot, typename Cell>
__device__ Row plan(const Lane<Slot, Cell>& lane, const uint8_t* flags, const Usage& usage, const Entry& entry, const Unit& unit) {
    const Extents& size = lane.size;
    const bool carrying = unit.carried > 0;
    const int32_t target_cell = lane.cell_of(entry.target);
    const int32_t range = lane.at(unit.type, kAttackRange);
    const bool in_range = squared_distance(unit.cell, target_cell, size.width) <= range * range;
    const int32_t harvest_goal = carrying ? lane.cell_of(entry.base) : target_cell;
    int32_t goal = kAbsent;
    int32_t radius = 0;
    Desire fallback;

    if (entry.kind == kAttack) {
        goal = target_cell;
        radius = range;

        if (in_range && target_cell != kAbsent) {
            fallback.kind = kKindAttack;
            fallback.target_cell = target_cell;
        }
    } else if (entry.kind == kHarvest) {
        goal = harvest_goal;
        radius = 1;
        const int32_t direction = adjacent_direction(unit.cell, harvest_goal, size.width);

        if (direction != kAbsent) {
            fallback.kind = carrying ? kKindReturn : kKindHarvest;
            fallback.direction = direction;
        }
    } else if (entry.kind == kTrain) {
        fallback = production(lane, flags, usage, unit.cell, training_direction(lane, flags, unit.cell, entry.unit_type), entry.unit_type);
    } else if (entry.kind == kBuild) {
        goal = entry.cell;
        radius = 1;
        fallback = production(lane, flags, usage, unit.cell, adjacent_direction(unit.cell, entry.cell, size.width), entry.unit_type);
    }

    const bool searches = goal != kAbsent && !(entry.kind == kAttack && in_range);
    return Row{searches ? goal : unit.cell, searches ? radius : 0, fallback};
}

struct Shared {
    astar::Scratch scratch;
    uint8_t* flags;
    Facts* facts;
    Slots slots;

    __device__ static Shared carve(uint64_t* shared, const Extents& size) {
        Shared carved;
        carved.scratch = astar::Scratch::carve(shared, size.cells);
        carved.flags = carved.scratch.seen + size.cells;
        char* const after_cells = reinterpret_cast<char*>(shared) + kBytesPerCell * size.cells;
        carved.facts = reinterpret_cast<Facts*>(after_cells);
        carved.slots = Slots::carve(after_cells + kLaneBytes, size.capacity);
        return carved;
    }
};

template <typename Slot, typename Cell>
__device__ void mark_pending_claims(const Lane<Slot, Cell>& lane, const Shared& shared) {
    const Extents& size = lane.size;

    if (threadIdx.x == 0) {
        *shared.facts = Facts{};
    }

    for (int32_t cell = threadIdx.x; cell < size.cells; cell += blockDim.x) {
        shared.scratch.top[cell] = 0;
    }

    __syncthreads();

    for (int32_t slot = threadIdx.x; slot < size.entities; slot += blockDim.x) {
        if (!lane.active(slot)) {
            continue;
        }

        const int32_t kind = lane.pending_kind[slot];

        if (kind == kKindMove || kind == kKindProduce) {
            int32_t target = neighbour(lane.cell_at(slot), lane.pending_direction[slot], size.width);

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
            atomicAdd(&shared.facts->reserved[block], static_cast<uint32_t>(lane.cost(produced)));
        }
    }
}

template <typename Slot, typename Cell>
__device__ void load_own_slot(const Inputs<Slot, Cell>& in, const Lane<Slot, Cell>& lane, const Shared& shared, int64_t lane_index, Unit& unit, Entry& entry) {
    const Extents& size = lane.size;
    const int32_t thread = threadIdx.x;
    const int32_t first = clip(size.neutral + lane.player * size.capacity, 0, size.entities - size.capacity);
    const int32_t slot = first + thread;
    unit.birth = lane.birth[slot];
    unit.present = unit.birth != kAbsent;
    unit.cell = lane.cell_at(slot);
    unit.type = lane.type_of(slot);
    unit.carried = lane.carried[slot];
    unit.available = unit.present && lane.pending_kind[slot] == kKindAbsent;
    shared.slots.birth[thread] = unit.birth;
    const int32_t* row[kEntryRows];

    for (int k = 0; k < kEntryRows; ++k) {
        row[k] = in.entries[k].at(lane_index);
    }

    entry = Entry{row[kEntryKind][thread], row[kEntryUnitBirth][thread], row[kEntryOrder][thread], row[kEntryTarget][thread], row[kEntryTargetBirth][thread], row[kEntryBase][thread], row[kEntryBaseBirth][thread], row[kEntryCell][thread], row[kEntryUnitType][thread], in.trained.at(lane_index)[thread]};

    if (!unit.present) {
        return;
    }

    Facts* facts = shared.facts;

    if (unit.type == kTypeBase) {
        atomicOr(&facts->any_base, 1);
        atomicMax(&facts->last_base, latest(unit.birth, thread));
    }

    if (unit.type == kTypeBarracks) {
        atomicOr(&facts->any_barracks, 1);
    }

    if (unit.type == kTypeWorker) {
        atomicAdd(&facts->workers_held, 1);
    }

    if (lane.at(unit.type, kCanHarvest) != 0) {
        atomicMin(&facts->first_worker, earliest(unit.birth, thread));
    }
}

template <typename Slot, typename Cell>
__device__ void flag_cells(const Lane<Slot, Cell>& lane, const Shared& shared) {
    for (int32_t cell = threadIdx.x; cell < lane.size.cells; cell += blockDim.x) {
        const bool empty = !lane.terrain[cell] && static_cast<int32_t>(lane.entity_at[cell]) == kAbsent;
        const bool pending = shared.scratch.top[cell] != 0;
        shared.flags[cell] = (empty ? kEmpty : 0) | (pending ? kTaken : 0) | (empty && !pending ? kOpen : 0);
    }
}

__device__ void find_second_worker_and_base(const Shared& shared, bool worker, Unit& unit) {
    Facts* facts = shared.facts;
    const int32_t thread = threadIdx.x;

    if (worker && earliest(unit.birth, thread) != facts->first_worker) {
        atomicMin(&facts->second_worker, earliest(unit.birth, thread));
    }

    if (unit.present && unit.type == kTypeBase && latest(unit.birth, thread) == facts->last_base) {
        facts->base_cell = unit.cell;
    }
}

__device__ int32_t rank_among_present(const Slots& slots, int32_t capacity, int32_t birth) {
    int32_t rank = 0;

    for (int32_t slot = 0; slot < capacity; ++slot) {
        const int32_t born = slots.birth[slot];
        rank += born != kAbsent && born < birth;
    }

    return rank;
}

template <typename Slot, typename Cell>
__device__ bool completed(const Lane<Slot, Cell>& lane, const Entry& entry, const Unit& unit) {
    if (entry.kind == kAttack) {
        return !lane.exists(entry.target, entry.target_birth);
    }

    if (entry.kind == kHarvest) {
        return unit.carried > 0 ? !lane.exists(entry.base, entry.base_birth) : !lane.exists(entry.target, entry.target_birth);
    }

    if (entry.kind == kTrain) {
        return entry.trained;
    }

    if (entry.kind == kBuild) {
        return entry.cell != kAbsent && static_cast<int32_t>(lane.entity_at[clip(entry.cell, 0, lane.size.cells - 1)]) != kAbsent;
    }

    return false;
}

__device__ void record_row(const Shared& shared, const Entry& entry, const Unit& unit, const Row& row, bool standing, bool executes) {
    const Slots& slots = shared.slots;
    const int32_t thread = threadIdx.x;
    slots.order[thread] = entry.order;
    slots.goal[thread] = row.goal;
    slots.radius[thread] = row.radius;
    slots.claim[thread] = row.fallback.claimed;
    slots.cost[thread] = row.fallback.cost;
    slots.start[thread] = static_cast<int16_t>(unit.cell);
    slots.rank[thread] = static_cast<int16_t>(kAbsent);
    slots.selector[thread] = kFellBack;
    slots.direction[thread] = static_cast<int8_t>(kAbsent);
    slots.marks[thread] = (executes ? kExecutes : 0) | (standing ? kStanding : 0) | (row.fallback.kind != kKindAbsent ? kEligible : 0);

    if (executes) {
        atomicAdd(&shared.facts->queued, 1);
    }

    if (standing) {
        atomicAdd(&shared.facts->standing, 1);
    }
}

__device__ int32_t queue_slot_and_settle(const Slots& slots, int32_t capacity, const Entry& entry, bool executes) {
    const int32_t thread = threadIdx.x;
    const int32_t key = executes ? entry.order : kLast;
    int32_t position = 0;
    int32_t settled = 0;

    for (int32_t slot = 0; slot < capacity; ++slot) {
        const uint8_t marks = slots.marks[slot];
        const int32_t order = slots.order[slot];
        const int32_t other_key = (marks & kExecutes) != 0 ? order : kLast;
        position += other_key < key || (other_key == key && slot < thread);
        settled += (marks & kStanding) != 0 && order < entry.order;
    }

    slots.queue[position] = static_cast<int16_t>(thread);
    return settled;
}

__device__ void run_ordered_pass(const Shared& shared, const Extents& size, const Usage& usage) {
    const Slots& slots = shared.slots;
    astar::Scratch scratch = shared.scratch;
    uint8_t* flags = shared.flags;
    const int32_t queued = shared.facts->queued;
    const bool moves_allowed = usage.other_within && (usage.reserved == 0u || static_cast<int32_t>(usage.reserved) <= usage.balance);
    uint32_t kept_cost = 0;
    int32_t kept_count = 0;

    for (int32_t position = 0; position < queued; ++position) {
        __syncthreads();

        for (int32_t cell = threadIdx.x; cell < size.cells; cell += blockDim.x) {
            scratch.clear(cell, (flags[cell] & kOpen) != 0);
        }

        __syncthreads();

        if (threadIdx.x != 0) {
            continue;
        }

        const int32_t slot = slots.queue[position];
        const int32_t start = slots.start[slot];
        const astar::Path path = astar::search(scratch, start, slots.goal[slot], slots.radius[slot], size.width, size.height);
        const bool stepped = path.step != astar::kAbsent;
        int32_t selector = kFellBack;
        int32_t direction = kAbsent;
        bool eligible = (slots.marks[slot] & kEligible) != 0;
        int32_t claim = slots.claim[slot];
        int32_t cost = slots.cost[slot];

        if (stepped && moves_allowed) {
            selector = kMoved;
            direction = path.step;
            eligible = true;
            claim = neighbour(start, direction, size.width);
            cost = 0;
        } else if (stepped) {
            selector = kBlocked;
            eligible = false;
            claim = kAbsent;
            cost = 0;
        }

        const bool on_board = claim >= 0 && claim < size.cells;
        const bool conflicting = on_board && (flags[claim] & kTaken) != 0;
        const bool overdrawn = cost > 0 && static_cast<int32_t>(usage.reserved + kept_cost + static_cast<uint32_t>(cost)) > usage.balance;
        const bool kept = eligible && !conflicting && !overdrawn;

        if (on_board) {
            flags[claim] = (flags[claim] & ~kOpen) | (kept ? kTaken : 0);
        }

        slots.rank[slot] = static_cast<int16_t>(kept ? kept_count : kAbsent);
        slots.selector[slot] = static_cast<uint8_t>(selector);
        slots.direction[slot] = static_cast<int8_t>(direction);

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

__device__ void decided_desire(const Slots& slots, const Row& row, Desire& decided, int32_t& rank) {
    const int32_t thread = threadIdx.x;
    rank = slots.rank[thread];

    if (rank == kAbsent) {
        return;
    }

    const int32_t selector = slots.selector[thread];

    if (selector == kMoved) {
        decided.kind = kKindMove;
        decided.direction = slots.direction[thread];
    } else if (selector == kFellBack) {
        decided = row.fallback;
    }
}

__device__ void write_commands(const Outputs& out, int64_t lane_index, int32_t capacity, int32_t player, const Desire& decided, int32_t rank, bool filler, int32_t filler_order) {
    using nanorts::whole_act::kCommandFields;
    using nanorts::whole_act::kPlayers;
    const bool kept = rank != kAbsent;
    int32_t command[kCommandFields];
    command[kCommandKind] = kept ? decided.kind : filler ? kKindNone : kKindAbsent;
    command[kCommandDirection] = kept ? decided.direction : kAbsent;
    command[kCommandNoneDuration] = kept && decided.kind == kKindNone ? decided.none_duration : filler ? kFillerDuration : 0;
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

__device__ void write_carry(const Outputs& out, int64_t lane_index, int32_t capacity, const Entry& entry, int32_t settled, bool standing, bool executes) {
    const int64_t at = lane_index * capacity + threadIdx.x;
    const int32_t held[kEntryRows] = {entry.kind, entry.unit_birth, settled, entry.target, entry.target_birth, entry.base, entry.base_birth, entry.cell, entry.unit_type};

    for (int k = 0; k < kEntryRows; ++k) {
        out.entries[k][at] = standing ? held[k] : kAbsent;
    }

    out.trained[at] = standing && (entry.trained || (executes && (entry.kind == kTrain || entry.kind == kTrainDirection)));
}

template <typename Slot, typename Cell>
__global__ void __launch_bounds__(nanorts::whole_act::kMaxThreads) act_kernel(Inputs<Slot, Cell> in, Outputs out, Extents size, Strategy strategy) {
    extern __shared__ uint64_t memory[];
    const int64_t lane_index = blockIdx.x;
    const int32_t capacity = size.capacity;
    const bool owns = static_cast<int32_t>(threadIdx.x) < capacity;
    const Lane<Slot, Cell> lane(in, lane_index, size);
    const Shared shared = Shared::carve(memory, size);
    Facts* facts = shared.facts;
    const int32_t own = gathered(lane.player, nanorts::whole_act::kPlayers);
    const int32_t other = gathered(1 - lane.player, nanorts::whole_act::kPlayers);

    mark_pending_claims(lane, shared);
    Unit unit{kAbsent, kAbsent, 0, 0, 0, false, false};
    Entry entry{};
    const uint32_t next_order = static_cast<uint32_t>(in.next_order.at(lane_index)[0]);

    if (owns) {
        load_own_slot(in, lane, shared, lane_index, unit, entry);
    }

    __syncthreads();
    flag_cells(lane, shared);
    const bool worker = unit.present && lane.at(unit.type, kCanHarvest) != 0;
    find_second_worker_and_base(shared, worker, unit);

    if (owns) {
        unit.rank = rank_among_present(shared.slots, capacity, unit.birth);
    }

    Closest enemy;
    Closest resource;
    Closest stockpile;
    closest_units(lane, unit, worker, enemy, resource, stockpile);
    __syncthreads();

    decide(lane, shared.flags, facts, strategy, entry, unit, enemy, resource, stockpile, next_order);

    const bool alive = entry.kind != kNoAction && entry.unit_birth == unit.birth;
    const bool standing = owns && alive && !completed(lane, entry, unit);
    const bool executes = standing && unit.available;
    const Usage usage{facts->reserved[own], lane.balance(own), facts->reserved[other] == 0u || static_cast<int32_t>(facts->reserved[other]) <= lane.balance(other)};
    Row row{};

    if (owns) {
        row = plan(lane, shared.flags, usage, entry, unit);
        record_row(shared, entry, unit, row, standing, executes);
    }

    __syncthreads();
    const int32_t settled = owns ? queue_slot_and_settle(shared.slots, capacity, entry, executes) : 0;

    run_ordered_pass(shared, size, usage);

    Desire decided;
    int32_t rank = kAbsent;
    bool filler = false;

    if (owns) {
        decided_desire(shared.slots, row, decided, rank);
        filler = unit.available && rank == kAbsent;

        if (filler) {
            shared.slots.marks[threadIdx.x] |= kFiller;
        }
    }

    __syncthreads();

    if (!owns) {
        return;
    }

    int32_t filler_rank = 0;

    for (int32_t slot = 0; slot < capacity; ++slot) {
        filler_rank += (shared.slots.marks[slot] & kFiller) != 0 && shared.slots.birth[slot] < unit.birth;
    }

    write_commands(out, lane_index, capacity, lane.player, decided, rank, filler, facts->kept_count + filler_rank);
    write_carry(out, lane_index, capacity, entry, settled, standing, executes);

    if (threadIdx.x == 0) {
        out.next_order[lane_index] = facts->standing;
    }
}

template <typename Slot, typename Cell>
ffi::Error launch(cudaStream_t stream, nanorts::whole_act::Operands& operands, const Outputs& outputs, const Extents& size, const Strategy& strategy, int64_t lanes) {
    Inputs<Slot, Cell> in{};

    if (auto e = nanorts::whole_act::bind_state(operands, size, in.state); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{size.types} * kTableColumns, in.table); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.player); e.failure()) {
        return e;
    }

    for (int k = 0; k < kEntryRows; ++k) {
        if (auto e = operands.next(int64_t{size.capacity}, in.entries[k]); e.failure()) {
            return e;
        }
    }

    if (auto e = operands.next(int64_t{size.capacity}, in.trained); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.next_order); e.failure()) {
        return e;
    }

    const int threads = nanorts::whole_act::threads_for(size.capacity);
    const int64_t shared = shared_bytes(size.cells, size.capacity);
    act_kernel<Slot, Cell><<<static_cast<unsigned>(lanes), threads, static_cast<size_t>(shared), stream>>>(in, outputs, size, strategy);
    return nanorts::launched();
}

constexpr int64_t kReadOperands = nanorts::whole_act::kStateOperands + 2;
constexpr int64_t kCarryLeaves = kEntryRows + 2;

ffi::Error Act(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets results, std::string_view build_id, int64_t code, int64_t trained) {
    constexpr std::string_view kTarget = "nanorts_abstraction_act";

    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (args.size() != kReadOperands + kCarryLeaves || results.size() != kCarryLeaves + 1) {
        return ffi::Error::InvalidArgument(
            "nanorts_abstraction_act takes the State leaves, the unit table, the player and "
            "the carry, and returns the carry and the Commands");
    }

    auto commands = results.get<ffi::AnyBuffer>(kCarryLeaves);

    if (!commands.has_value()) {
        return commands.error();
    }

    ffi::AnyBuffer table_rows = *args.get<ffi::AnyBuffer>(nanorts::whole_act::kStateOperands);
    const auto issued = (*commands)->dimensions();
    const auto table = table_rows.dimensions();

    if ((*commands)->element_type() != ffi::S32 || issued.size() != 4 || issued[1] != nanorts::whole_act::kCommandFields || issued[2] != nanorts::whole_act::kPlayers || table.size() != 3 || table[2] != kTableColumns || table_rows.element_type() != ffi::S32) {
        return ffi::Error::InvalidArgument("the Commands are `[lanes, 6, 2, C]` int32 and the unit table `[1, T, 6]` int32");
    }

    const int64_t lanes = issued[0];
    nanorts::whole_act::Operands operands(kTarget, args, lanes);
    Extents size{};
    ffi::DataType slot = ffi::DataType::INVALID;
    ffi::DataType cell = ffi::DataType::INVALID;

    if (auto e = nanorts::whole_act::read_state(operands, size, slot, cell); e.failure()) {
        return e;
    }

    size.capacity = static_cast<int32_t>(issued[3]);
    size.neutral = size.entities - nanorts::whole_act::kPlayers * size.capacity;
    size.types = static_cast<int32_t>(table[1]);
    const int64_t shared = shared_bytes(size.cells, size.capacity);

    if (size.cells <= 0 || size.capacity <= 0 || size.neutral < 0 || size.types <= 0 || shared > nanorts::whole_act::kSharedLimit || nanorts::whole_act::threads_for(size.capacity) > nanorts::whole_act::kMaxThreads || (code != kWorkerRush && code != kBarracksRush && code != kWorkerDefense && code != kBarracksDefense) || (trains_at_barracks(code) && (trained < 0 || trained >= size.types)) || size.types <= kTypeWorker) {
        return ffi::Error::InvalidArgument(
            "a strategy and trained type the table holds, over a lane whose act fits one "
            "block's threads and shared memory");
    }

    Outputs outputs{};

    for (int k = 0; k < kEntryRows; ++k) {
        if (auto e = nanorts::whole_act::carry_result(operands, results, k, kReadOperands + k, lanes, size.capacity, outputs.entries[k]); e.failure()) {
            return e;
        }
    }

    if (auto e = nanorts::whole_act::carry_result(operands, results, kEntryRows, kReadOperands + kEntryRows, lanes, size.capacity, outputs.trained); e.failure()) {
        return e;
    }

    if (auto e = nanorts::whole_act::carry_result(operands, results, kEntryRows + 1, kReadOperands + kEntryRows + 1, lanes, 1, outputs.next_order); e.failure()) {
        return e;
    }

    outputs.commands = static_cast<int32_t*>((*commands)->untyped_data());

    if (lanes == 0) {
        return ffi::Error::Success();
    }

    const Strategy strategy{static_cast<int32_t>(code), static_cast<int32_t>(trains_at_barracks(code) ? trained : kAbsent)};
    using ffi::DataType;

    if (slot == DataType::S16 && cell == DataType::S16) {
        return launch<int16_t, int16_t>(stream, operands, outputs, size, strategy, lanes);
    }

    if (slot == DataType::S16) {
        return launch<int16_t, int32_t>(stream, operands, outputs, size, strategy, lanes);
    }

    if (cell == DataType::S16) {
        return launch<int32_t, int16_t>(stream, operands, outputs, size, strategy, lanes);
    }

    return launch<int32_t, int32_t>(stream, operands, outputs, size, strategy, lanes);
}

const nanorts::Kernels nanorts_abstraction_act_kernels("nanorts_abstraction_act", {nanorts::kernel(act_kernel<int16_t, int16_t>), nanorts::kernel(act_kernel<int16_t, int32_t>), nanorts::kernel(act_kernel<int32_t, int16_t>), nanorts::kernel(act_kernel<int32_t, int32_t>)});

}

NANORTS_HANDLER(nanorts_abstraction_act, Act,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .RemainingArgs()
                    .RemainingRets()
                    .Attr<std::string_view>("build_id")
                    .Attr<int64_t>("strategy")
                    .Attr<int64_t>("trained"));
