#include <cstdint>

#include "nanorts_ffi.cuh"
#include "random_draw.cuh"
#include "whole_act.cuh"

namespace {

namespace ffi = xla::ffi;
namespace draw = nanorts::random_draw;
using nanorts::whole_act::Extents;
using nanorts::whole_act::Lanes;
using nanorts::whole_act::StateRows;

constexpr int64_t kBytesPerCell = 8;
constexpr int64_t kBytesPerBoardWord = 68;
constexpr int64_t kLaneBytes = 16;
constexpr int64_t kBytesPerCandidate = 11;
constexpr int64_t kBytesPerType = 4;
constexpr int64_t kBytesPerSlot = 31;

constexpr int32_t kAbsent = -1;
constexpr int32_t kLast = 2147483647;
constexpr int32_t kAfterEveryAttack = 1073741824;
constexpr int kDirections = 4;
constexpr int32_t kNoneDuration = 10;
constexpr int16_t kLargestFact = 32767;
constexpr uint32_t kFullWarp = 0xffffffffu;
constexpr uint32_t kTails = 0x80000000u;

constexpr int32_t kKindAbsent = -1;
constexpr int32_t kKindNone = 0;
constexpr int32_t kKindMove = 1;
constexpr int32_t kKindHarvest = 2;
constexpr int32_t kKindReturn = 3;
constexpr int32_t kKindProduce = 4;
constexpr int32_t kKindAttack = 5;

constexpr int32_t kOwnerNeutral = -1;
constexpr int32_t kTypeResource = 0;
constexpr int32_t kTypeBase = 1;
constexpr int32_t kTypeBarracks = 2;
constexpr int32_t kTypeWorker = 3;
constexpr int32_t kTypeHeavy = 5;
constexpr int32_t kTypeRanged = 6;

constexpr int32_t kSoldier = 0;
constexpr int32_t kHarvester = 1;
constexpr int32_t kBuilder = 2;
constexpr int32_t kJobCount = 3;
constexpr int32_t kSoldierAttackScore = 1;
constexpr int32_t kWorkerAttackScore = 2;
constexpr int32_t kArrivalScore = 1;
constexpr int32_t kBuildSiteScore = 8;
constexpr int kBuildSiteRadius = 1;
constexpr int kBaseSiteRadius = 2;

constexpr int kCanHarvest = 0;
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

constexpr int32_t kWordBits = 32;
constexpr int32_t kKinds = 6;
constexpr int32_t kMoveColumn = kKinds;
constexpr int32_t kHarvestColumn = kMoveColumn + kDirections;
constexpr int32_t kReturnColumn = kHarvestColumn + kDirections;
constexpr int32_t kProduceDirectionColumn = kReturnColumn + kDirections;
constexpr int32_t kProduceTypeColumn = kProduceDirectionColumn + kDirections;

constexpr int kOffsetAxes = 2;

constexpr int kEnemy = 0;
constexpr int kResource = 1;
constexpr int kOwnBase = 2;
constexpr int kFields = 3;

constexpr uint8_t kSlotAvailable = 1;
constexpr uint8_t kSlotCarrying = 2;
constexpr uint8_t kSlotGives = 4;
constexpr uint8_t kSlotClaims = 8;
constexpr uint8_t kSlotProduces = 16;
constexpr uint8_t kSlotBase = 32;
constexpr uint8_t kSlotRewritten = 64;

constexpr uint8_t kMarked = 1;
constexpr uint8_t kAccepted = 2;
constexpr uint8_t kRemaining = 4;

__host__ __device__ constexpr int32_t board_words(int64_t cells) {
    return static_cast<int32_t>((cells + kWordBits - 1) / kWordBits);
}

constexpr int64_t shared_bytes(int64_t cells, int64_t capacity, int64_t candidates, int64_t types) {
    return kBytesPerCell * cells + kBytesPerBoardWord * board_words(cells) + kLaneBytes + kBytesPerCandidate * candidates + kBytesPerType * types + kBytesPerSlot * capacity;
}

struct Window {
    int32_t words;
    int32_t offsets;
    int32_t candidates;
};

template <typename Slot, typename Cell>
struct Inputs {
    StateRows<Slot, Cell> state;
    Lanes<int32_t> table;
    Lanes<int32_t> player;
    Lanes<int32_t> none_duration;
    Lanes<uint32_t> legality;
    Lanes<uint32_t> key;
    Lanes<int32_t> offsets;
    Lanes<int8_t> job;
    Lanes<int32_t> holder;
    Lanes<bool> harvester_given;
    Lanes<bool> builder_given;
};

struct Outputs {
    int8_t* job;
    int32_t* holder;
    bool* harvester_given;
    bool* builder_given;
    int32_t* commands;
};

struct Facts {
    int32_t visits;
    int32_t no_barracks;
    uint32_t harvester_given;
    uint32_t builder_given;
};

static_assert(sizeof(Facts) <= kLaneBytes, "the lane's facts outgrew their bytes");
static_assert(nanorts::whole_act::kSharedOptInLimit / kBytesPerCell < kLargestFact, "a cell's fact outgrew int16");

struct Shared {
    Facts* facts;
    int32_t* birth;
    int32_t* cell;
    int32_t* claimed;
    int32_t* claim_cell;
    int32_t* cost;
    int32_t* chosen;
    int32_t* order;
    int32_t* score;
    int32_t* list_position;
    int32_t* unaffordable;
    uint32_t* frontier;
    uint32_t* reached;
    uint32_t* touched;
    uint32_t* passable;
    uint32_t* not_first;
    uint32_t* not_last;
    int16_t* fact;
    int16_t* rewrite;
    int8_t* unit_type;
    int8_t* scored_as;
    uint8_t* slot;
    uint8_t* around;
    uint8_t* candidate;

    __device__ static Shared carve(uint64_t* shared, const Extents& size, const Window& window) {
        const int32_t capacity = size.capacity;
        const int32_t cells = size.cells;
        const int32_t candidates = window.candidates;
        const int32_t words = board_words(cells);
        Shared carved;
        char* const base = reinterpret_cast<char*>(shared);
        carved.facts = reinterpret_cast<Facts*>(base);
        carved.birth = reinterpret_cast<int32_t*>(base + kLaneBytes);
        carved.cell = carved.birth + capacity;
        carved.claimed = carved.cell + capacity;
        carved.claim_cell = carved.claimed + capacity;
        carved.cost = carved.claim_cell + capacity;
        carved.chosen = carved.cost + capacity;
        carved.order = carved.chosen + capacity;
        carved.score = carved.order + capacity;
        carved.list_position = carved.score + candidates;
        carved.unaffordable = carved.list_position + candidates;
        carved.frontier = reinterpret_cast<uint32_t*>(carved.unaffordable + size.types);
        carved.reached = carved.frontier + 2 * kFields * words;
        carved.touched = carved.reached + kFields * words;
        carved.passable = carved.touched + kFields * words;
        carved.not_first = carved.passable + kFields * words;
        carved.not_last = carved.not_first + words;
        carved.fact = reinterpret_cast<int16_t*>(carved.not_last + words);
        carved.rewrite = carved.fact + kFields * cells;
        carved.unit_type = reinterpret_cast<int8_t*>(carved.rewrite + candidates);
        carved.scored_as = carved.unit_type + capacity;
        carved.slot = reinterpret_cast<uint8_t*>(carved.scored_as + capacity);
        carved.around = carved.slot + capacity;
        carved.candidate = carved.around + 2 * cells;
        return carved;
    }
};

__device__ int32_t clip(int32_t value, int32_t low, int32_t high) {
    return value < low ? low : value > high ? high : value;
}

__device__ int32_t gathered(int32_t index, int32_t size) {
    return clip(index < 0 ? index + size : index, 0, size - 1);
}

__device__ int32_t wrapped_sub(int32_t left, int32_t right) {
    return static_cast<int32_t>(static_cast<uint32_t>(left) - static_cast<uint32_t>(right));
}

__device__ int32_t floor_div(int32_t value, int32_t divisor) {
    const int32_t quotient = value / divisor;
    return quotient - ((value % divisor != 0) && ((value < 0) != (divisor < 0)));
}

__device__ int32_t floor_mod(int32_t value, int32_t divisor) {
    return value - floor_div(value, divisor) * divisor;
}

__device__ int32_t step_x(int32_t direction) {
    const int32_t known = clip(direction, 0, kDirections - 1);
    return known == 1 ? 1 : known == 3 ? -1 : 0;
}

__device__ int32_t step_y(int32_t direction) {
    const int32_t known = clip(direction, 0, kDirections - 1);
    return known == 0 ? -1 : known == 2 ? 1 : 0;
}

__device__ int32_t stepped(int32_t column, int32_t row, int32_t direction, const Extents& size, bool& on_board) {
    const int32_t x = column + step_x(direction);
    const int32_t y = row + step_y(direction);
    on_board = x >= 0 && x < size.width && y >= 0 && y < size.height;
    return clip(y * size.width + x, 0, size.cells - 1);
}

__device__ int32_t window_target(int32_t column, int32_t row, const int32_t* offset, const Extents& size) {
    const int32_t x = column + offset[0];
    const int32_t y = row + offset[1];
    const bool on_board = x >= 0 && x < size.width && y >= 0 && y < size.height;
    return on_board ? y * size.width + x : kAbsent;
}

__device__ int32_t adjacent_order(const int32_t* offset) {
    for (int32_t direction = 0; direction < kDirections; ++direction) {
        if (offset[0] == step_x(direction) && offset[1] == step_y(direction)) {
            return direction;
        }
    }

    return kAfterEveryAttack;
}

__device__ bool directional(int32_t kind) {
    return kind == kKindMove || kind == kKindHarvest || kind == kKindReturn || kind == kKindProduce;
}

__device__ unsigned long long birth_key(int32_t birth, int32_t slot) {
    return static_cast<unsigned long long>(static_cast<uint32_t>(birth) ^ 0x80000000u) << 32 | static_cast<uint32_t>(slot);
}

__device__ int32_t warp_min(int32_t value) {
#if __CUDA_ARCH__ >= 800
    return __reduce_min_sync(kFullWarp, value);
#else
    for (int distance = 16; distance > 0; distance >>= 1) {
        value = min(value, __shfl_xor_sync(kFullWarp, value, distance));
    }

    return value;
#endif
}

__device__ int32_t warp_max(int32_t value) {
#if __CUDA_ARCH__ >= 800
    return __reduce_max_sync(kFullWarp, value);
#else
    for (int distance = 16; distance > 0; distance >>= 1) {
        value = max(value, __shfl_xor_sync(kFullWarp, value, distance));
    }

    return value;
#endif
}

__device__ unsigned long long warp_min(unsigned long long value) {
#if __CUDA_ARCH__ >= 800
    const uint32_t high = __reduce_min_sync(kFullWarp, static_cast<uint32_t>(value >> 32));
    const uint32_t low = __reduce_min_sync(kFullWarp, static_cast<uint32_t>(value >> 32) == high ? static_cast<uint32_t>(value) : ~0u);
    return static_cast<unsigned long long>(high) << 32 | low;
#else
    for (int distance = 16; distance > 0; distance >>= 1) {
        const unsigned long long other = __shfl_xor_sync(kFullWarp, value, distance);
        value = other < value ? other : value;
    }

    return value;
#endif
}

__device__ uint32_t warp_or(uint32_t value) {
#if __CUDA_ARCH__ >= 800
    return __reduce_or_sync(kFullWarp, value);
#else
    for (int distance = 16; distance > 0; distance >>= 1) {
        value |= __shfl_xor_sync(kFullWarp, value, distance);
    }

    return value;
#endif
}

__device__ int32_t in_direction(const int32_t (&values)[kDirections], int32_t direction) {
    int32_t picked = values[0];

#pragma unroll
    for (int32_t other = 1; other < kDirections; ++other) {
        picked = direction == other ? values[other] : picked;
    }

    return picked;
}

struct Candidate {
    int32_t kind = kKindNone;
    int32_t direction = kAbsent;
    int32_t produced_type = kAbsent;
    int32_t offset = kAbsent;

    __device__ Candidate(int32_t index, const Window& window, int32_t types) {
        if (index < window.offsets) {
            kind = kKindAttack;
            offset = index;
        } else if ((index -= window.offsets) < kDirections) {
            kind = kKindHarvest;
            direction = index;
        } else if ((index -= kDirections) < kDirections) {
            kind = kKindReturn;
            direction = index;
        } else if ((index -= kDirections) < kDirections * types) {
            kind = kKindProduce;
            produced_type = index / kDirections;
            direction = index % kDirections;
        } else if ((index -= kDirections * types) < kDirections) {
            kind = kKindMove;
            direction = index;
        }
    }
};

__device__ bool is_set(const uint32_t* row, int32_t column) {
    return (row[column / kWordBits] >> (column % kWordBits) & 1u) != 0u;
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
    const int32_t* none_duration;
    const Slot* entity_at;
    const int32_t* resources;
    const bool* terrain;
    const int32_t* table;
    const uint32_t* legality;
    draw::Key key;
    const int32_t* offsets;
    int32_t player;
    int32_t own;
    Extents size;
    Window window;

    __device__ Lane(const Inputs<Slot, Cell>& in, int64_t lane, const Extents& extents, const Window& candidates) : birth(in.state.birth.at(lane)), unit_type(in.state.unit_type.at(lane)), owner(in.state.owner.at(lane)), cell(in.state.cell.at(lane)), carried(in.state.carried.at(lane)), pending_kind(in.state.pending_kind.at(lane)), pending_direction(in.state.pending_direction.at(lane)), pending_produced_type(in.state.pending_produced_type.at(lane)), none_duration(in.none_duration.at(lane)), entity_at(in.state.entity_at.at(lane)), resources(in.state.resources.at(lane)), terrain(in.state.terrain.at(lane)), table(in.table.at(lane)), legality(in.legality.at(lane)), key{in.key.at(lane)[0], in.key.at(lane)[1]}, offsets(in.offsets.at(lane)), player(in.player.at(lane)[0]), own(gathered(player, nanorts::whole_act::kPlayers)), size(extents), window(candidates) {}

    __device__ int32_t at(int32_t type, int column) const {
        return table[clip(type, 0, size.types - 1) * kTableColumns + column];
    }

    __device__ int32_t own_slot(int32_t local) const {
        return size.neutral + own * size.capacity + local;
    }

    __device__ bool available(int32_t local) const {
        const int32_t slot = own_slot(local);
        return birth[slot] != kAbsent && pending_kind[slot] == kKindAbsent;
    }

    __device__ const uint32_t* row(int32_t local) const {
        return legality + (static_cast<int64_t>(own) * size.capacity + local) * window.words;
    }

    __device__ const int32_t* offset(int32_t index) const {
        return offsets + clip(index, 0, window.offsets - 1) * kOffsetAxes;
    }

    __device__ bool marked(int32_t local, const Candidate& candidate) const {
        const uint32_t* words = row(local);

        switch (candidate.kind) {
            case kKindAttack:
                return is_set(words, kProduceTypeColumn + size.types + candidate.offset);
            case kKindHarvest:
                return is_set(words, kHarvestColumn + candidate.direction);
            case kKindReturn:
                return is_set(words, kReturnColumn + candidate.direction);
            case kKindProduce:
                return is_set(words, kProduceTypeColumn + candidate.produced_type) && is_set(words, kProduceDirectionColumn + candidate.direction);
            case kKindMove:
                return is_set(words, kMoveColumn + candidate.direction);
            default:
                return is_set(words, kKindNone);
        }
    }
};

template <typename Slot, typename Cell>
__device__ uint8_t units_around(const Lane<Slot, Cell>& lane, int32_t cell, int radius) {
    const int32_t x = cell % lane.size.width;
    const int32_t y = cell / lane.size.width;
    uint8_t count = 0;

    for (int32_t row = max(0, y - radius); row <= min(lane.size.height - 1, y + radius); ++row) {
        for (int32_t column = max(0, x - radius); column <= min(lane.size.width - 1, x + radius); ++column) {
            count += static_cast<int32_t>(lane.entity_at[row * lane.size.width + column]) != kAbsent;
        }
    }

    return count;
}

template <typename Slot, typename Cell>
__device__ void read_slots(const Lane<Slot, Cell>& lane, const Inputs<Slot, Cell>& in, const Shared& shared, int64_t lane_index) {
    const Extents& size = lane.size;

    if (threadIdx.x == 0) {
        *shared.facts = Facts{0, 1, 0u, 0u};
    }

    for (int32_t word = threadIdx.x; word < kFields * board_words(size.cells); word += blockDim.x) {
        shared.frontier[word] = 0u;
    }

    const int32_t local = threadIdx.x;

    if (local >= size.capacity) {
        return;
    }

    const int32_t slot = lane.own_slot(local);
    const int32_t birth = lane.birth[slot];
    const int32_t cell = static_cast<int32_t>(lane.cell[slot]);
    const int32_t type = lane.unit_type[slot];
    const bool available = lane.available(local);
    const int32_t kind = lane.pending_kind[slot];
    const bool busy = birth != kAbsent && kind != kKindAbsent;
    const int32_t parameter = kind == kKindNone ? lane.none_duration[slot] : directional(kind) ? lane.pending_direction[slot] : kAbsent;
    bool on_board = false;
    const int32_t claim_cell = stepped(floor_mod(cell, size.width), floor_div(cell, size.width), parameter, size, on_board);
    const bool claims = busy && parameter >= 0 && parameter < kDirections && on_board;
    const bool produces = busy && kind == kKindProduce;
    const int32_t produced = clip(lane.pending_produced_type[slot], 0, size.types - 1);
    const bool gives = available && type == kTypeWorker && in.holder.at(lane_index)[local] != birth;
    shared.birth[local] = birth;
    shared.cell[local] = cell;
    shared.claimed[local] = kAbsent;
    shared.claim_cell[local] = claim_cell;
    shared.cost[local] = produces ? lane.at(produced, kCost) : 0;
    shared.chosen[local] = kAbsent;
    shared.unit_type[local] = static_cast<int8_t>(type);
    shared.slot[local] = (available ? kSlotAvailable : 0) | (lane.carried[slot] > 0 ? kSlotCarrying : 0) | (gives ? kSlotGives : 0) | (claims ? kSlotClaims : 0) | (produces ? kSlotProduces : 0) | (type == kTypeBase ? kSlotBase : 0);
}

template <typename Slot, typename Cell>
__device__ void count_sites(const Lane<Slot, Cell>& lane, const Shared& shared) {
    const Extents& size = lane.size;

    for (int32_t cell = threadIdx.x; cell < size.cells; cell += blockDim.x) {
        shared.around[cell] = units_around(lane, cell, kBuildSiteRadius);
        shared.around[size.cells + cell] = units_around(lane, cell, kBaseSiteRadius);
    }
}

template <typename Slot, typename Cell>
__device__ void keep_jobs(const Lane<Slot, Cell>& lane, const Inputs<Slot, Cell>& in, const Outputs& out, int64_t lane_index) {
    const int32_t local = threadIdx.x;

    if (local < lane.size.capacity) {
        const int64_t row = lane_index * lane.size.capacity + local;
        out.job[row] = in.job.at(lane_index)[local];
        out.holder[row] = in.holder.at(lane_index)[local];
    }
}

__device__ void mark_target(const Shared& shared, int field, int32_t cell, int32_t cells) {
    atomicOr(&shared.frontier[field * board_words(cells) + cell / kWordBits], 1u << (cell % kWordBits));
}

template <typename Slot, typename Cell>
__device__ void mark_targets_and_give_jobs(const Lane<Slot, Cell>& lane, const Inputs<Slot, Cell>& in, const Outputs& out, const Shared& shared, int64_t lane_index) {
    const Extents& size = lane.size;

    for (int32_t slot = threadIdx.x; slot < size.entities; slot += blockDim.x) {
        if (lane.birth[slot] == kAbsent) {
            continue;
        }

        const int32_t type = lane.unit_type[slot];
        const int32_t owner = lane.owner[slot];

        if (type == kTypeBarracks) {
            shared.facts->no_barracks = 0;
        }

        int32_t cell = static_cast<int32_t>(lane.cell[slot]);

        if (cell < 0) {
            cell += size.cells;
        }

        if (cell < 0 || cell >= size.cells) {
            continue;
        }

        if (owner == 1 - lane.player) {
            mark_target(shared, kEnemy, cell, size.cells);
        }

        if (type == kTypeResource && owner == kOwnerNeutral) {
            mark_target(shared, kResource, cell, size.cells);
        }

        if (type == kTypeBase && owner == lane.player) {
            mark_target(shared, kOwnBase, cell, size.cells);
        }
    }

    const int32_t local = threadIdx.x;

    if (local >= size.capacity) {
        return;
    }

    const int32_t birth = shared.birth[local];
    const bool available = (shared.slot[local] & kSlotAvailable) != 0;
    const int32_t key = available ? birth : kLast;
    int32_t givers_before = 0;
    int32_t rank = 0;

    for (int32_t other = 0; other < size.capacity; ++other) {
        const int32_t other_birth = shared.birth[other];
        const bool other_available = (shared.slot[other] & kSlotAvailable) != 0;
        givers_before += (shared.slot[other] & kSlotGives) != 0 && other_birth < birth;
        const int32_t other_key = other_available ? other_birth : kLast;
        rank += other_key < key || (other_key == key && other < local);
    }

    shared.order[rank] = local;

    if (available) {
        atomicAdd(&shared.facts->visits, 1);
    }

    const bool harvester_given = in.harvester_given.at(lane_index)[0];
    const int32_t harvesters_left = harvester_given ? 0 : 1;
    const int32_t builders_left = harvesters_left + (in.builder_given.at(lane_index)[0] ? 0 : 1);
    const int32_t next_job = givers_before < harvesters_left ? kHarvester : givers_before < builders_left ? kBuilder : kSoldier;
    const bool gives = (shared.slot[local] & kSlotGives) != 0;
    const int8_t held = in.job.at(lane_index)[local];
    const int32_t job = gives ? next_job : held;
    const int64_t row = lane_index * size.capacity + local;
    out.job[row] = gives ? static_cast<int8_t>(job) : held;
    out.holder[row] = gives ? birth : in.holder.at(lane_index)[local];

    if (gives && job == kHarvester) {
        atomicOr(&shared.facts->harvester_given, 1u);
    }

    if (gives && job == kBuilder) {
        atomicOr(&shared.facts->builder_given, 1u);
    }

    const bool worker = shared.unit_type[local] == kTypeWorker;
    shared.scored_as[local] = static_cast<int8_t>(clip(worker ? job : next_job, 0, kJobCount - 1));
}

template <bool kOpenOnly>
__device__ uint32_t board_word(const uint32_t* board, const uint32_t* open, int32_t word, int32_t words) {
    if (word < 0 || word >= words) {
        return 0u;
    }

    return kOpenOnly ? board[word] & open[word] : board[word];
}

template <bool kOpenOnly>
__device__ uint32_t neighbours(const uint32_t* board, const uint32_t* open, int32_t word, const Extents& size, uint32_t not_first, uint32_t not_last) {
    const int32_t words = board_words(size.cells);
    const int32_t rows = size.width / kWordBits;
    const int32_t shift = size.width % kWordBits;
    const uint32_t before = board_word<kOpenOnly>(board, open, word - 1, words);
    const uint32_t here = board_word<kOpenOnly>(board, open, word, words);
    const uint32_t after = board_word<kOpenOnly>(board, open, word + 1, words);
    uint32_t from_above = __funnelshift_l(before, here, shift);
    uint32_t from_below = __funnelshift_r(here, after, shift);

    if (rows > 0) {
        from_above = __funnelshift_l(board_word<kOpenOnly>(board, open, word - rows - 1, words), board_word<kOpenOnly>(board, open, word - rows, words), shift);
        from_below = __funnelshift_r(board_word<kOpenOnly>(board, open, word + rows, words), board_word<kOpenOnly>(board, open, word + rows + 1, words), shift);
    }

    const uint32_t from_left = __funnelshift_l(before, here, 1) & not_first;
    const uint32_t from_right = __funnelshift_r(here, after, 1) & not_last;
    return from_above | from_below | from_left | from_right;
}

__device__ int32_t board_field(int32_t board, int32_t words) {
    int32_t field = 0;

    for (int later = 1; later < kFields; ++later) {
        field += board >= later * words;
    }

    return field;
}

template <typename Slot, typename Cell>
__device__ void load_boards(const Lane<Slot, Cell>& lane, const Shared& shared) {
    const Extents& size = lane.size;
    const int32_t words = board_words(size.cells);
    const int32_t bit = static_cast<int32_t>(threadIdx.x) % kWordBits;
    const int16_t unreached = static_cast<int16_t>(size.cells + 1);

    for (int32_t first = static_cast<int32_t>(threadIdx.x) - bit; first < words * kWordBits; first += blockDim.x) {
        const int32_t cell = first + bit;
        const int32_t word = first / kWordBits;
        const bool inside = cell < size.cells;
        const int32_t column = inside ? cell % size.width : 0;
        const bool open = inside && !lane.terrain[cell];
        const bool empty = inside && static_cast<int32_t>(lane.entity_at[cell]) == kAbsent;
        const uint32_t not_first = __ballot_sync(kFullWarp, inside && column != 0);
        const uint32_t not_last = __ballot_sync(kFullWarp, inside && column != size.width - 1);

        for (int field = 0; field < kFields; ++field) {
            const uint32_t targets = shared.frontier[field * words + word];
            const bool target = (targets >> bit & 1u) != 0u;
            const uint32_t through = __ballot_sync(kFullWarp, open && (empty || target));

            if (bit == 0) {
                shared.passable[field * words + word] = through;
                shared.reached[field * words + word] = through & targets;
                shared.touched[field * words + word] = 0u;
            }

            if (inside) {
                shared.fact[field * size.cells + cell] = unreached;
            }
        }

        if (bit == 0) {
            shared.not_first[word] = not_first;
            shared.not_last[word] = not_last;
        }
    }
}

template <typename Slot, typename Cell>
__device__ void measure_distances(const Lane<Slot, Cell>& lane, const Shared& shared) {
    const Extents& size = lane.size;
    const int32_t words = board_words(size.cells);
    const int32_t boards = kFields * words;
    load_boards(lane, shared);
    __syncthreads();

    for (int32_t level = 0;; ++level) {
        const uint32_t* const frontier = shared.frontier + (level & 1) * boards;
        uint32_t* const next_frontier = shared.frontier + ((level + 1) & 1) * boards;
        bool grew = false;

        for (int32_t board = threadIdx.x; board < boards; board += blockDim.x) {
            const int32_t field = board_field(board, words);
            const int32_t word = board - field * words;
            const uint32_t* const field_frontier = frontier + field * words;
            const uint32_t not_first = shared.not_first[word];
            const uint32_t not_last = shared.not_last[word];
            const uint32_t grown = neighbours<false>(field_frontier, nullptr, word, size, not_first, not_last);
            const uint32_t measured = level == 0 ? neighbours<true>(field_frontier, shared.passable + field * words, word, size, not_first, not_last) : grown;
            const uint32_t touched = shared.touched[board];
            const uint32_t reached = shared.reached[board];
            const uint32_t fresh = grown & shared.passable[board] & ~reached;
            shared.touched[board] = touched | measured;
            shared.reached[board] = reached | fresh;
            next_frontier[board] = fresh;
            grew = grew || fresh != 0u;

            for (uint32_t first_touch = measured & ~touched; first_touch != 0u; first_touch &= first_touch - 1u) {
                const int32_t cell = word * kWordBits + __ffs(static_cast<int>(first_touch)) - 1;

                if (cell < size.cells) {
                    shared.fact[field * size.cells + cell] = static_cast<int16_t>(level + 1);
                }
            }
        }

        if (__syncthreads_or(grew) == 0) {
            break;
        }
    }
}

struct Unit {
    int32_t slot;
    bool available;
    int32_t cell;
    int32_t x;
    int32_t y;
    int32_t type;
    bool fights;
    bool carrying;
    int32_t scored_as;
};

__device__ int32_t neighbour(const Unit& unit, int32_t direction, const Extents& size) {
    bool on_board = false;
    return stepped(unit.x, unit.y, direction, size, on_board);
}

struct Around {
    uint32_t planned;
    uint32_t claimed;
    int32_t first_position[kDirections];
    int32_t first_rewrite[kDirections];
    int32_t rewrite_slot[kDirections];
};

template <typename Slot, typename Cell>
__device__ Around around(const Lane<Slot, Cell>& lane, const Shared& shared, const Unit& unit, bool can_produce) {
    const Extents& size = lane.size;
    Around found{};
    int32_t neighbours[kDirections];
    bool on_board[kDirections];
    unsigned long long rewrite_key[kDirections];

#pragma unroll
    for (int32_t direction = 0; direction < kDirections; ++direction) {
        neighbours[direction] = stepped(unit.x, unit.y, direction, size, on_board[direction]);
        found.first_position[direction] = kLast;
        found.first_rewrite[direction] = kLast;
        rewrite_key[direction] = ~0ull;
    }

    const bool barracks = unit.type == kTypeBarracks;

    for (int32_t slot = threadIdx.x; slot < size.capacity; slot += nanorts::whole_act::kWarp) {
        const uint8_t flags = shared.slot[slot];
        const bool in_play = (flags & kSlotRewritten) == 0;
        const bool claims = (flags & kSlotClaims) != 0 && in_play;
        const bool pair = (flags & kSlotProduces) != 0 && in_play && (flags & kSlotBase) != 0 && barracks;
        const int32_t claimed = shared.claimed[slot];
        const int32_t claim_cell = shared.claim_cell[slot];
        const int32_t birth = shared.birth[slot];

#pragma unroll
        for (int32_t direction = 0; direction < kDirections; ++direction) {
            const int32_t neighbour = neighbours[direction];

            if (on_board[direction] && claimed == neighbour) {
                found.planned |= 1u << direction;
            }

            if (claims && claim_cell == neighbour && on_board[direction]) {
                found.claimed |= 1u << direction;
                found.first_position[direction] = min(found.first_position[direction], birth);
            } else if (pair) {
                const unsigned long long key = birth_key(birth, slot);

                if (key < rewrite_key[direction]) {
                    rewrite_key[direction] = key;
                }
            }
        }
    }

    const uint32_t marks = warp_or(found.planned | found.claimed << kDirections);
    found.planned = marks & ((1u << kDirections) - 1u);
    found.claimed = marks >> kDirections;

    if (can_produce) {
#pragma unroll
        for (int32_t direction = 0; direction < kDirections; ++direction) {
            found.first_position[direction] = warp_min(found.first_position[direction]);
        }
    }

    if (can_produce && barracks) {
#pragma unroll
        for (int32_t direction = 0; direction < kDirections; ++direction) {
            const unsigned long long key = warp_min(rewrite_key[direction]);
            found.first_rewrite[direction] = key == ~0ull ? kLast : static_cast<int32_t>(static_cast<uint32_t>(key >> 32) ^ 0x80000000u);
            found.rewrite_slot[direction] = key == ~0ull ? 0 : static_cast<int32_t>(key & 0xffffffffu);
        }
    }

    return found;
}

template <typename Slot, typename Cell>
__device__ void first_unaffordable(const Lane<Slot, Cell>& lane, const Shared& shared, const Unit& unit) {
    const Extents& size = lane.size;
    const int32_t balance = lane.resources[lane.own];
    const bool barracks = unit.type == kTypeBarracks;
    const uint32_t* words = lane.row(unit.slot);

    for (int32_t type = 0; type < size.types; ++type) {
        if (!is_set(words, kProduceTypeColumn + type)) {
            continue;
        }

        const int32_t price = lane.at(type, kCost);
        int32_t first = kLast;

        for (int32_t slot = threadIdx.x; slot < size.capacity; slot += nanorts::whole_act::kWarp) {
            const uint8_t flags = shared.slot[slot];
            const bool producing = (flags & kSlotProduces) != 0 && (flags & kSlotRewritten) == 0;
            const bool pair = producing && (flags & kSlotBase) != 0 && barracks;

            if (producing && !pair && wrapped_sub(balance, shared.cost[slot]) < price) {
                first = min(first, shared.birth[slot]);
            }
        }

        first = warp_min(first);

        if (threadIdx.x == 0) {
            shared.unaffordable[type] = first;
        }
    }

    __syncwarp();
}

template <typename Slot, typename Cell>
__device__ int32_t scored(const Lane<Slot, Cell>& lane, const Shared& shared, const Unit& unit, const Candidate& candidate, int32_t facts_cell) {
    const Extents& size = lane.size;
    const int32_t enemy = shared.fact[kEnemy * size.cells + facts_cell];
    const int32_t resource = shared.fact[kResource * size.cells + facts_cell];
    const int32_t own_base = shared.fact[kOwnBase * size.cells + facts_cell];
    const int32_t worst = -size.cells;
    const bool idle = candidate.kind == kKindNone;
    const bool attack = candidate.kind == kKindAttack;
    const bool produce = candidate.kind == kKindProduce;

    if (unit.type == kTypeBase) {
        if (idle) {
            return worst;
        }

        if (unit.scored_as == kSoldier) {
            return -enemy;
        }

        if (unit.scored_as == kHarvester) {
            return -resource;
        }

        return shared.around[size.cells + facts_cell];
    }

    const int32_t soldier = idle || produce ? worst : attack ? kSoldierAttackScore : -enemy;

    if (unit.fights) {
        return soldier;
    }

    if (unit.type == kTypeBarracks) {
        if (idle || candidate.produced_type == kTypeHeavy) {
            return worst;
        }

        const bool far_for_ranged = enemy > size.width + size.height / 4 && candidate.produced_type == kTypeRanged;
        return -enemy + static_cast<int32_t>(far_for_ranged);
    }

    if (unit.type != kTypeWorker) {
        return worst;
    }

    if (unit.scored_as == kSoldier) {
        return soldier;
    }

    const int32_t gather = unit.carrying ? (candidate.kind == kKindReturn ? kArrivalScore : -own_base) : (candidate.kind == kKindHarvest ? kArrivalScore : -resource);

    if (attack) {
        return kWorkerAttackScore;
    }

    if (unit.scored_as == kHarvester) {
        return produce || idle ? worst : gather;
    }

    if (idle) {
        return worst;
    }

    const bool no_barracks = shared.facts->no_barracks != 0;
    return no_barracks && produce ? kBuildSiteScore - shared.around[facts_cell] : gather;
}

template <typename Slot, typename Cell>
__device__ void judge(const Lane<Slot, Cell>& lane, const Shared& shared, const Unit& unit, const Around& found) {
    const Extents& size = lane.size;
    const Window& window = lane.window;
    const int32_t attack_range = lane.at(unit.type, kAttackRange);
    int32_t top = -kLast;

    for (int32_t index = threadIdx.x; index < window.candidates; index += nanorts::whole_act::kWarp) {
        const Candidate candidate(index, window, size.types);
        const bool marked = unit.available && lane.marked(unit.slot, candidate);
        const int32_t known = clip(candidate.direction, 0, kDirections - 1);
        int32_t target_cell = kAbsent;

        if (candidate.kind == kKindAttack) {
            target_cell = window_target(unit.x, unit.y, lane.offset(candidate.offset), size);
        } else if (candidate.direction != kAbsent) {
            target_cell = neighbour(unit, known, size);
        }

        const int32_t facts_cell = clip(target_cell, 0, size.cells - 1);
        const int32_t score = scored(lane, shared, unit, candidate, facts_cell);
        int32_t list_position = kAfterEveryAttack + index;

        if (candidate.kind == kKindAttack) {
            if (attack_range == 1) {
                list_position = adjacent_order(lane.offset(candidate.offset));
            } else {
                const int32_t occupant = static_cast<int32_t>(lane.entity_at[facts_cell]);
                list_position = lane.birth[clip(occupant, 0, size.entities - 1)];
            }
        }

        const bool planned = (found.planned >> known & 1u) != 0;
        bool rejected = planned || (found.claimed >> known & 1u) != 0;
        int32_t rewrite = kAbsent;

        if (candidate.kind == kKindProduce && marked) {
            const int32_t first_rewrite = in_direction(found.first_rewrite, known);
            const int32_t first_reject = min(in_direction(found.first_position, known), shared.unaffordable[candidate.produced_type]);
            rejected = planned || first_reject < first_rewrite;

            if (!planned && first_rewrite < first_reject) {
                rewrite = in_direction(found.rewrite_slot, known);
            }
        }

        const bool accepted = !(directional(candidate.kind) && rejected);
        shared.score[index] = score;
        shared.list_position[index] = list_position;
        shared.rewrite[index] = static_cast<int16_t>(rewrite);
        shared.candidate[index] = (marked ? kMarked : 0) | (accepted ? kAccepted : 0);

        if (marked && accepted) {
            top = max(top, score);
        }
    }

    top = warp_max(top);

    for (int32_t index = threadIdx.x; index < window.candidates; index += nanorts::whole_act::kWarp) {
        if ((shared.candidate[index] & kMarked) != 0 && shared.score[index] == top) {
            shared.candidate[index] |= kRemaining;
        }
    }

    __syncwarp();
}

__device__ int32_t drawn(const Shared& shared, const Window& window, draw::Key key, int32_t local, uint32_t attempt, bool& left) {
    int32_t first = kLast;
    int32_t lowest = window.candidates;
    int32_t held = 0;

    for (int32_t index = threadIdx.x; index < window.candidates; index += nanorts::whole_act::kWarp) {
        if ((shared.candidate[index] & kRemaining) == 0) {
            continue;
        }

        first = min(first, shared.list_position[index]);
        lowest = min(lowest, index);
        ++held;
    }

    const uint32_t holders = __ballot_sync(kFullWarp, held > 0);
    left = __popc(holders) > 1 || __any_sync(kFullWarp, held > 1);

    if (!left) {
        return holders == 0u ? kAbsent : __shfl_sync(kFullWarp, lowest, __ffs(static_cast<int>(holders)) - 1);
    }

    first = warp_min(first);
    const draw::Key attempt_key = draw::split(draw::split(key, static_cast<uint32_t>(local)), attempt);
    int32_t latest = -1;

    for (int32_t index = threadIdx.x; index < window.candidates; index += nanorts::whole_act::kWarp) {
        if ((shared.candidate[index] & kRemaining) == 0) {
            continue;
        }

        const int32_t list_position = shared.list_position[index];

        if (list_position == first || (draw::bits(attempt_key, static_cast<uint32_t>(index)) & kTails) == 0u) {
            latest = max(latest, list_position);
        }
    }

    latest = warp_max(latest);
    int32_t chosen = window.candidates;

    for (int32_t index = threadIdx.x; index < window.candidates; index += nanorts::whole_act::kWarp) {
        if ((shared.candidate[index] & kRemaining) != 0 && shared.list_position[index] == latest) {
            chosen = min(chosen, index);
        }
    }

    chosen = warp_min(chosen);
    return chosen == window.candidates ? 0 : chosen;
}

template <typename Slot, typename Cell>
__device__ void walk(const Lane<Slot, Cell>& lane, const Shared& shared) {
    const Extents& size = lane.size;
    const int32_t visits = shared.facts->visits;

    for (int32_t position = 0; position < visits; ++position) {
        const int32_t local = shared.order[position];
        const uint8_t flags = shared.slot[local];
        const int32_t cell = shared.cell[local];
        const int32_t type = shared.unit_type[local];
        const bool fights = lane.at(type, kCanAttack) != 0 && lane.at(type, kCanHarvest) == 0;
        const Unit unit{local, (flags & kSlotAvailable) != 0, cell, floor_mod(cell, size.width), floor_div(cell, size.width), type, fights, (flags & kSlotCarrying) != 0, shared.scored_as[local]};
        const uint32_t* words = lane.row(local);
        bool marked_type = false;

        for (int32_t produced = threadIdx.x; produced < size.types; produced += nanorts::whole_act::kWarp) {
            marked_type = marked_type || is_set(words, kProduceTypeColumn + produced);
        }

        const bool can_produce = __any_sync(kFullWarp, marked_type);
        const Around found = around(lane, shared, unit, can_produce);

        if (can_produce) {
            first_unaffordable(lane, shared, unit);
        }

        judge(lane, shared, unit, found);

        for (uint32_t attempt = 0;; ++attempt) {
            bool left = false;
            const int32_t chosen = drawn(shared, lane.window, lane.key, local, attempt, left);
            const int32_t settled_choice = unit.available ? chosen : kAbsent;
            const int32_t known = clip(settled_choice, 0, lane.window.candidates - 1);
            const bool refused = settled_choice == kAbsent || (shared.candidate[known] & kAccepted) == 0;

            if (refused && left) {
                __syncwarp();

                if (static_cast<int32_t>(threadIdx.x) == chosen % nanorts::whole_act::kWarp) {
                    shared.candidate[chosen] &= static_cast<uint8_t>(~kRemaining);
                }

                __syncwarp();
                continue;
            }

            if (threadIdx.x == 0) {
                const Candidate candidate(known, lane.window, size.types);
                shared.chosen[local] = settled_choice;

                if (settled_choice != kAbsent && directional(candidate.kind)) {
                    shared.claimed[local] = neighbour(unit, clip(candidate.direction, 0, kDirections - 1), size);
                }

                const int32_t rewrite = shared.rewrite[known];

                if (settled_choice != kAbsent && rewrite >= 0 && rewrite < size.capacity) {
                    shared.slot[rewrite] |= kSlotRewritten;
                }
            }

            __syncwarp();
            break;
        }
    }
}

template <typename Slot, typename Cell>
__device__ void write_commands(const Lane<Slot, Cell>& lane, const Shared& shared, int32_t* commands, int64_t lane_index) {
    using nanorts::whole_act::kCommandFields;
    using nanorts::whole_act::kPlayers;
    const Extents& size = lane.size;
    const int32_t local = threadIdx.x;
    const int32_t chosen = shared.chosen[local];
    const int32_t cell = shared.cell[local];
    const bool issued = chosen != kAbsent;
    const Candidate candidate(clip(chosen, 0, lane.window.candidates - 1), lane.window, size.types);
    int32_t command[kCommandFields];
    command[kCommandKind] = issued ? candidate.kind : kKindAbsent;
    command[kCommandDirection] = issued ? candidate.direction : kAbsent;
    command[kCommandNoneDuration] = issued && candidate.kind == kKindNone ? kNoneDuration : 0;
    command[kCommandTargetCell] = issued && candidate.kind == kKindAttack ? window_target(floor_mod(cell, size.width), floor_div(cell, size.width), lane.offset(candidate.offset), size) : kAbsent;
    command[kCommandProducedType] = issued ? candidate.produced_type : kAbsent;
    command[kCommandRequestOrder] = kAbsent;
    int32_t unissued[kCommandFields];
    unissued[kCommandKind] = kKindAbsent;
    unissued[kCommandDirection] = kAbsent;
    unissued[kCommandNoneDuration] = 0;
    unissued[kCommandTargetCell] = kAbsent;
    unissued[kCommandProducedType] = kAbsent;
    unissued[kCommandRequestOrder] = kAbsent;
    int32_t* row = commands + lane_index * kCommandFields * kPlayers * size.capacity;

    for (int field = 0; field < kCommandFields; ++field) {
        for (int32_t who = 0; who < kPlayers; ++who) {
            row[(field * kPlayers + who) * size.capacity + local] = who == lane.player ? command[field] : unissued[field];
        }
    }
}

template <typename Slot, typename Cell>
__global__ void __launch_bounds__(nanorts::whole_act::kMaxThreads) act_kernel(Inputs<Slot, Cell> in, Outputs out, Extents size, Window window) {
    extern __shared__ uint64_t memory[];
    const int64_t lane_index = blockIdx.x;
    const Lane<Slot, Cell> lane(in, lane_index, size, window);
    const Shared shared = Shared::carve(memory, size, window);
    read_slots(lane, in, shared, lane_index);
    const bool own_available = static_cast<int32_t>(threadIdx.x) < size.capacity && (shared.slot[threadIdx.x] & kSlotAvailable) != 0;

    if (__syncthreads_or(own_available) != 0) {
        count_sites(lane, shared);
        mark_targets_and_give_jobs(lane, in, out, shared, lane_index);
        __syncthreads();
        measure_distances(lane, shared);
        __syncthreads();

        if (threadIdx.x < nanorts::whole_act::kWarp) {
            walk(lane, shared);
        }

        __syncthreads();
    } else {
        keep_jobs(lane, in, out, lane_index);
    }

    if (threadIdx.x == 0) {
        out.harvester_given[lane_index] = in.harvester_given.at(lane_index)[0] || shared.facts->harvester_given != 0u;
        out.builder_given[lane_index] = in.builder_given.at(lane_index)[0] || shared.facts->builder_given != 0u;
    }

    if (static_cast<int32_t>(threadIdx.x) < size.capacity) {
        write_commands(lane, shared, out.commands, lane_index);
    }
}

template <typename Slot, typename Cell>
ffi::Error launch(cudaStream_t stream, nanorts::whole_act::Operands& operands, const Outputs& outputs, const Extents& size, const Window& window, int64_t lanes) {
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

    if (auto e = operands.next(int64_t{size.entities}, in.none_duration); e.failure()) {
        return e;
    }

    const int64_t rows = int64_t{nanorts::whole_act::kPlayers} * size.capacity;

    if (auto e = operands.next(rows * window.words, in.legality); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{2}, in.key); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{window.offsets} * kOffsetAxes, in.offsets); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{size.capacity}, in.job); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{size.capacity}, in.holder); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.harvester_given); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{1}, in.builder_given); e.failure()) {
        return e;
    }

    const int threads = nanorts::whole_act::threads_for(size.capacity);
    const int64_t shared = shared_bytes(size.cells, size.capacity, window.candidates, size.types);
    static nanorts::whole_act::SharedOptIn opt_in;

    if (auto e = opt_in.allow(operands, nanorts::kernel(act_kernel<Slot, Cell>), shared); e.failure()) {
        return e;
    }

    act_kernel<Slot, Cell><<<static_cast<unsigned>(lanes), threads, static_cast<size_t>(shared), stream>>>(in, outputs, size, window);
    return nanorts::launched();
}

constexpr int64_t kNoneDurationOperand = nanorts::whole_act::kStateOperands + 2;
constexpr int64_t kLegalityOperand = kNoneDurationOperand + 1;
constexpr int64_t kKeyOperand = kLegalityOperand + 1;
constexpr int64_t kOffsetsOperand = kKeyOperand + 1;
constexpr int64_t kReadOperands = kOffsetsOperand + 1;
constexpr int64_t kCarryLeaves = 4;

ffi::Error check_extras(const nanorts::whole_act::Operands& operands, int64_t capacity, int64_t types, Window& window) {
    auto none_duration = operands.at(kNoneDurationOperand);
    auto words = operands.at(kLegalityOperand);
    auto key = operands.at(kKeyOperand);
    auto offsets = operands.at(kOffsetsOperand);

    if (!none_duration.has_value()) {
        return none_duration.error();
    }

    if (!words.has_value()) {
        return words.error();
    }

    if (!key.has_value()) {
        return key.error();
    }

    if (!offsets.has_value()) {
        return offsets.error();
    }

    const auto steps = offsets->dimensions();

    if (offsets->element_type() != ffi::DataType::S32 || steps.size() != 3u || steps[1] <= 0 || steps[2] != kOffsetAxes) {
        return operands.refused("the attack offsets are `[1, O, 2]` int32");
    }

    window.offsets = static_cast<int32_t>(steps[1]);
    window.candidates = window.offsets + 2 * kDirections + kDirections * static_cast<int32_t>(types) + kDirections + 1;
    const int64_t columns = kProduceTypeColumn + types + window.offsets;
    const auto rows = words->dimensions();

    if (words->element_type() != ffi::DataType::U32 || rows.size() != 4u || rows[1] != nanorts::whole_act::kPlayers || rows[2] != capacity || rows[3] != (columns + kWordBits - 1) / kWordBits) {
        return operands.refused("the Legality words are `[lanes, 2, C, W]` uint32, with W words holding a row of the table's types and offsets");
    }

    window.words = static_cast<int32_t>(rows[3]);

    if (none_duration->element_type() != ffi::DataType::S32 || key->element_type() != ffi::DataType::U32) {
        return operands.refused("the Pending NONE durations are int32 and the key uint32 data");
    }

    return ffi::Error::Success();
}

ffi::Error Act(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets results, std::string_view build_id) {
    constexpr std::string_view kTarget = "nanorts_obi_act";

    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (args.size() != kReadOperands + kCarryLeaves || results.size() != kCarryLeaves + 1) {
        return ffi::Error::InvalidArgument(
            "nanorts_obi_act takes the State leaves, the unit table, the player, the Pending NONE "
            "durations, the Legality words, the key, the attack offsets and the jobs, and returns "
            "the jobs and the Commands");
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
    Window window{};

    if (auto e = check_extras(operands, size.capacity, size.types, window); e.failure()) {
        return e;
    }

    const int64_t shared = shared_bytes(size.cells, size.capacity, window.candidates, size.types);

    if (size.cells <= 0 || size.cells >= kLargestFact || size.capacity <= 0 || size.neutral < 0 || size.types <= 0 || shared > nanorts::whole_act::kSharedOptInLimit || nanorts::whole_act::threads_for(size.capacity) > nanorts::whole_act::kMaxThreads) {
        return ffi::Error::InvalidArgument("a lane whose act fits one block's threads and shared memory");
    }

    Outputs outputs{};
    const int64_t job = kReadOperands;

    if (auto e = nanorts::whole_act::carry_result(operands, results, 0, job, lanes, size.capacity, outputs.job); e.failure()) {
        return e;
    }

    if (auto e = nanorts::whole_act::carry_result(operands, results, 1, job + 1, lanes, size.capacity, outputs.holder); e.failure()) {
        return e;
    }

    if (auto e = nanorts::whole_act::carry_result(operands, results, 2, job + 2, lanes, 1, outputs.harvester_given); e.failure()) {
        return e;
    }

    if (auto e = nanorts::whole_act::carry_result(operands, results, 3, job + 3, lanes, 1, outputs.builder_given); e.failure()) {
        return e;
    }

    outputs.commands = static_cast<int32_t*>((*commands)->untyped_data());

    if (lanes == 0) {
        return ffi::Error::Success();
    }

    using ffi::DataType;

    if (slot == DataType::S16 && cell == DataType::S16) {
        return launch<int16_t, int16_t>(stream, operands, outputs, size, window, lanes);
    }

    if (slot == DataType::S16) {
        return launch<int16_t, int32_t>(stream, operands, outputs, size, window, lanes);
    }

    if (cell == DataType::S16) {
        return launch<int32_t, int16_t>(stream, operands, outputs, size, window, lanes);
    }

    return launch<int32_t, int32_t>(stream, operands, outputs, size, window, lanes);
}

const nanorts::Kernels nanorts_obi_act_kernels("nanorts_obi_act", {nanorts::kernel(act_kernel<int16_t, int16_t>), nanorts::kernel(act_kernel<int16_t, int32_t>), nanorts::kernel(act_kernel<int32_t, int16_t>), nanorts::kernel(act_kernel<int32_t, int32_t>)});

}

NANORTS_HANDLER(nanorts_obi_act, Act,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .RemainingArgs()
                    .RemainingRets()
                    .Attr<std::string_view>("build_id"));
