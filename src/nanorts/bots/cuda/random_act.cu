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

constexpr int64_t kBytesPerCell = 1;
constexpr int64_t kLaneBytes = 32;
constexpr int64_t kBytesPerSlot = 13;

constexpr int64_t kEveryUnit = 0;
constexpr int64_t kOneUnit = 1;

constexpr uint32_t kUnitStream = 0;
constexpr uint32_t kActionStream = 1;

constexpr int32_t kAbsent = -1;
constexpr int32_t kLast = 2147483647;
constexpr int kDirections = 4;
constexpr int32_t kNoneDuration = 10;

constexpr int32_t kKindAbsent = -1;
constexpr int32_t kKindNone = 0;
constexpr int32_t kKindMove = 1;
constexpr int32_t kKindHarvest = 2;
constexpr int32_t kKindReturn = 3;
constexpr int32_t kKindProduce = 4;
constexpr int32_t kKindAttack = 5;

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

constexpr uint8_t kClaims = 1;
constexpr uint8_t kFree = 2;
constexpr uint8_t kFiller = 4;

constexpr int64_t shared_bytes(int64_t cells, int64_t capacity) {
    return kBytesPerCell * cells + kLaneBytes + kBytesPerSlot * capacity;
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
    Lanes<uint32_t> legality;
    Lanes<uint32_t> key;
    Lanes<float> log_weight;
    Lanes<int32_t> offsets;
};

struct Facts {
    uint32_t reserved[nanorts::whole_act::kPlayers];
    unsigned long long pick;
    int32_t keepable;
};

static_assert(sizeof(Facts) <= kLaneBytes, "the lane's facts outgrew their bytes");

struct Shared {
    Facts* facts;
    int32_t* birth;
    int32_t* claimed;
    int32_t* cost;
    uint8_t* marks;
    uint8_t* blocked;

    __device__ static Shared carve(uint64_t* shared, const Extents& size) {
        Shared carved;
        char* const base = reinterpret_cast<char*>(shared);
        carved.facts = reinterpret_cast<Facts*>(base);
        carved.birth = reinterpret_cast<int32_t*>(base + kLaneBytes);
        carved.claimed = carved.birth + size.capacity;
        carved.cost = carved.claimed + size.capacity;
        carved.marks = reinterpret_cast<uint8_t*>(carved.cost + size.capacity);
        carved.blocked = carved.marks + size.capacity;
        return carved;
    }
};

__device__ int32_t clip(int32_t value, int32_t low, int32_t high) {
    return value < low ? low : value > high ? high : value;
}

__device__ int32_t gathered(int32_t index, int32_t size) {
    return clip(index < 0 ? index + size : index, 0, size - 1);
}

__device__ int32_t wrapped_add(int32_t left, int32_t right) {
    return static_cast<int32_t>(static_cast<uint32_t>(left) + static_cast<uint32_t>(right));
}

__device__ int32_t step_x(int32_t direction) {
    const int32_t known = clip(direction, 0, kDirections - 1);
    return known == 1 ? 1 : known == 3 ? -1 : 0;
}

__device__ int32_t step_y(int32_t direction) {
    const int32_t known = clip(direction, 0, kDirections - 1);
    return known == 0 ? -1 : known == 2 ? 1 : 0;
}

__device__ int32_t neighbour(int32_t cell, int32_t direction, int32_t width) {
    return wrapped_add(cell, step_y(direction) * width + step_x(direction));
}

__device__ int32_t floor_div(int32_t value, int32_t divisor) {
    const int32_t quotient = value / divisor;
    return quotient - ((value % divisor != 0) && ((value < 0) != (divisor < 0)));
}

__device__ int32_t floor_mod(int32_t value, int32_t divisor) {
    return value - floor_div(value, divisor) * divisor;
}

__device__ bool affordable(int32_t cost, int32_t reserved, int32_t balance) {
    return reserved == 0 || wrapped_add(cost, reserved) <= balance;
}

__device__ unsigned long long min_key(int32_t order, int32_t slot) {
    return static_cast<unsigned long long>(static_cast<uint32_t>(order) ^ 0x80000000u) << 32 | static_cast<uint32_t>(slot);
}

__device__ unsigned long long max_key(float score, int32_t slot) {
    const uint32_t bits = __float_as_uint(score);
    const uint32_t ordered = (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
    return static_cast<unsigned long long>(ordered) << 32 | (0xffffffffu - static_cast<uint32_t>(slot));
}

__device__ bool is_set(const uint32_t* row, int32_t column) {
    return (row[column / kWordBits] >> (column % kWordBits) & 1u) != 0u;
}

template <typename Slot, typename Cell>
struct Lane {
    const int32_t* birth;
    const Cell* cell;
    const int8_t* pending_kind;
    const int8_t* pending_direction;
    const int8_t* pending_produced_type;
    const int32_t* resources;
    const int32_t* table;
    const uint32_t* legality;
    draw::Key key;
    const float* log_weight;
    const int32_t* offsets;
    int32_t player;
    int32_t own;
    int32_t other;
    Extents size;

    __device__ Lane(const Inputs<Slot, Cell>& in, int64_t lane, const Extents& extents) : birth(in.state.birth.at(lane)), cell(in.state.cell.at(lane)), pending_kind(in.state.pending_kind.at(lane)), pending_direction(in.state.pending_direction.at(lane)), pending_produced_type(in.state.pending_produced_type.at(lane)), resources(in.state.resources.at(lane)), table(in.table.at(lane)), legality(in.legality.at(lane)), key{in.key.at(lane)[0], in.key.at(lane)[1]}, log_weight(in.log_weight.at(lane)), offsets(in.offsets.at(lane)), player(in.player.at(lane)[0]), own(gathered(player, nanorts::whole_act::kPlayers)), other(gathered(1 - player, nanorts::whole_act::kPlayers)), size(extents) {}

    __device__ int32_t cost(int32_t type) const {
        return table[clip(type, 0, size.types - 1) * kTableColumns + kCost];
    }

    __device__ int32_t own_slot(int32_t local) const {
        return size.neutral + own * size.capacity + local;
    }

    __device__ bool available(int32_t local) const {
        const int32_t slot = own_slot(local);
        return birth[slot] != kAbsent && pending_kind[slot] == kKindAbsent;
    }

    __device__ const uint32_t* row(int32_t local, int32_t words) const {
        return legality + (static_cast<int64_t>(own) * size.capacity + local) * words;
    }
};

struct Choice {
    int32_t kind = kKindAttack;
    int32_t direction = kAbsent;
    int32_t target_cell = kAbsent;
    int32_t produced_type = kAbsent;
    int32_t claimed_cell = kAbsent;
    int32_t cost = 0;
};

struct Usage {
    int32_t reserved;
    int32_t balance;
    bool other_affordable;
};

template <typename Slot, typename Cell>
__device__ void mark_pending_claims(const Lane<Slot, Cell>& lane, const Shared& shared) {
    const Extents& size = lane.size;

    if (threadIdx.x == 0) {
        *shared.facts = Facts{};
    }

    for (int32_t cell = threadIdx.x; cell < size.cells; cell += blockDim.x) {
        shared.blocked[cell] = 0;
    }

    __syncthreads();

    for (int32_t slot = threadIdx.x; slot < size.entities; slot += blockDim.x) {
        if (lane.birth[slot] == kAbsent) {
            continue;
        }

        const int32_t kind = lane.pending_kind[slot];

        if (kind == kKindMove || kind == kKindProduce) {
            int32_t target = neighbour(static_cast<int32_t>(lane.cell[slot]), lane.pending_direction[slot], size.width);

            if (target < 0) {
                target += size.cells;
            }

            if (target >= 0 && target < size.cells) {
                shared.blocked[target] = 1;
            }
        }

        if (kind == kKindProduce && slot >= size.neutral) {
            const int32_t block = (slot - size.neutral) / size.capacity;
            atomicAdd(&shared.facts->reserved[block], static_cast<uint32_t>(lane.cost(lane.pending_produced_type[slot])));
        }
    }

    __syncthreads();
}

template <typename Slot, typename Cell>
__device__ int32_t drawn_candidate(const Lane<Slot, Cell>& lane, draw::Key key, uint32_t first, int32_t local, const Window& window) {
    const int32_t types = lane.size.types;
    const int32_t attack_column = kProduceTypeColumn + types;
    const uint32_t* row = lane.row(local, window.words);
    draw::Best best;
    int32_t index = 0;
    const auto offer = [&](bool marked) {
        if (marked) {
            best.offer(draw::score(key, first + static_cast<uint32_t>(index), lane.log_weight[index]), index);
        }

        ++index;
    };

    for (int32_t offset = 0; offset < window.offsets; ++offset) {
        offer(is_set(row, attack_column + offset));
    }

    for (int32_t direction = 0; direction < kDirections; ++direction) {
        offer(is_set(row, kHarvestColumn + direction));
    }

    for (int32_t direction = 0; direction < kDirections; ++direction) {
        offer(is_set(row, kReturnColumn + direction));
    }

    for (int32_t direction = 0; direction < kDirections; ++direction) {
        for (int32_t type = 0; type < types; ++type) {
            offer(is_set(row, kProduceDirectionColumn + direction) && is_set(row, kProduceTypeColumn + type));
        }
    }

    for (int32_t direction = 0; direction < kDirections; ++direction) {
        offer(is_set(row, kMoveColumn + direction));
    }

    offer(is_set(row, kKindNone));
    return best.index;
}

template <typename Slot, typename Cell>
__device__ Choice chosen(const Lane<Slot, Cell>& lane, int32_t index, int32_t cell, const Window& window) {
    const int32_t types = lane.size.types;
    const int32_t width = lane.size.width;
    Choice choice;
    int32_t offset = kAbsent;

    if (index < window.offsets) {
        offset = index;
    } else if ((index -= window.offsets) < kDirections) {
        choice.kind = kKindHarvest;
        choice.direction = index;
    } else if ((index -= kDirections) < kDirections) {
        choice.kind = kKindReturn;
        choice.direction = index;
    } else if ((index -= kDirections) < kDirections * types) {
        choice.kind = kKindProduce;
        choice.direction = index / types;
        choice.produced_type = index % types;
    } else if ((index -= kDirections * types) < kDirections) {
        choice.kind = kKindMove;
        choice.direction = index;
    } else {
        choice.kind = kKindNone;
    }

    if (choice.kind == kKindAttack) {
        const int32_t* step = lane.offsets + clip(offset, 0, window.offsets - 1) * kOffsetAxes;
        const int32_t x = floor_mod(cell, width) + step[0];
        const int32_t y = floor_div(cell, width) + step[1];
        const bool on_board = x >= 0 && x < width && y >= 0 && y < lane.size.height;
        choice.target_cell = on_board ? y * width + x : kAbsent;
    }

    if (choice.kind == kKindMove || choice.kind == kKindProduce) {
        choice.claimed_cell = neighbour(cell, choice.direction, width);
    }

    if (choice.kind == kKindProduce) {
        choice.cost = lane.cost(choice.produced_type);
    }

    return choice;
}

__device__ bool blocked_by_pending(const Shared& shared, const Extents& size, const Choice& choice) {
    return choice.claimed_cell != kAbsent && shared.blocked[clip(choice.claimed_cell, 0, size.cells - 1)] != 0;
}

__device__ bool claims_before(const Shared& shared, int32_t earlier, int32_t later) {
    return (shared.marks[earlier] & kClaims) != 0 && (shared.marks[later] & kClaims) != 0 && shared.claimed[earlier] == shared.claimed[later] && shared.birth[earlier] < shared.birth[later];
}

__device__ void write_commands(int32_t* commands, int64_t lane_index, int32_t capacity, int32_t player, bool issued, bool kept, const Choice& choice, int32_t request_order) {
    using nanorts::whole_act::kCommandFields;
    using nanorts::whole_act::kPlayers;
    const int32_t kind = kept ? choice.kind : kKindNone;
    int32_t command[kCommandFields];
    command[kCommandKind] = issued ? kind : kKindAbsent;
    command[kCommandDirection] = issued && kept ? choice.direction : kAbsent;
    command[kCommandNoneDuration] = issued && kind == kKindNone ? kNoneDuration : 0;
    command[kCommandTargetCell] = issued && kept ? choice.target_cell : kAbsent;
    command[kCommandProducedType] = issued && kept ? choice.produced_type : kAbsent;
    command[kCommandRequestOrder] = issued ? request_order : kAbsent;
    int32_t unissued[kCommandFields];
    unissued[kCommandKind] = kKindAbsent;
    unissued[kCommandDirection] = kAbsent;
    unissued[kCommandNoneDuration] = 0;
    unissued[kCommandTargetCell] = kAbsent;
    unissued[kCommandProducedType] = kAbsent;
    unissued[kCommandRequestOrder] = kAbsent;
    int32_t* row = commands + lane_index * kCommandFields * kPlayers * capacity;

    for (int field = 0; field < kCommandFields; ++field) {
        for (int32_t who = 0; who < kPlayers; ++who) {
            row[(field * kPlayers + who) * capacity + threadIdx.x] = who == player ? command[field] : unissued[field];
        }
    }
}

template <typename Slot, typename Cell>
__device__ void act_on_every_unit(const Lane<Slot, Cell>& lane, const Shared& shared, const Usage& usage, const Window& window, int32_t* commands, int64_t lane_index) {
    const int32_t capacity = lane.size.capacity;
    const int32_t local = threadIdx.x;
    const bool owns = local < capacity;
    bool available = false;
    int32_t birth = kAbsent;
    Choice choice;

    if (owns) {
        available = lane.available(local);
        birth = lane.birth[lane.own_slot(local)];
        const int32_t index = drawn_candidate(lane, lane.key, static_cast<uint32_t>(local) * static_cast<uint32_t>(window.candidates), local, window);
        choice = chosen(lane, index, static_cast<int32_t>(lane.cell[lane.own_slot(local)]), window);
    }

    const bool eligible = available && !blocked_by_pending(shared, lane.size, choice) && usage.other_affordable;
    const bool costing = eligible && choice.cost > 0;
    const bool free = eligible && choice.cost <= 0 && affordable(0, usage.reserved, usage.balance);

    if (owns) {
        shared.birth[local] = birth;
        shared.claimed[local] = choice.claimed_cell;
        shared.cost[local] = choice.cost;
        shared.marks[local] = (choice.claimed_cell != kAbsent ? kClaims : 0) | (free ? kFree : 0);
    }

    __syncthreads();
    bool taken = false;

    if (owns) {
        for (int32_t earlier = 0; earlier < capacity; ++earlier) {
            taken |= (shared.marks[earlier] & kFree) != 0 && claims_before(shared, earlier, local);
        }
    }

    bool kept = false;
    int32_t reserve = usage.reserved;

    for (;;) {
        const bool keepable = costing && !kept && !taken && affordable(choice.cost, reserve, usage.balance);

        if (local == 0) {
            shared.facts->pick = ~0ull;
            shared.facts->keepable = 0;
        }

        __syncthreads();

        if (owns) {
            atomicMin(&shared.facts->pick, min_key(keepable ? birth : kLast, local));

            if (keepable) {
                shared.facts->keepable = 1;
            }
        }

        __syncthreads();

        if (shared.facts->keepable == 0) {
            break;
        }

        const int32_t first = static_cast<int32_t>(shared.facts->pick & 0xffffffffu);
        kept |= local == first;
        taken |= owns && claims_before(shared, first, local);
        reserve = wrapped_add(reserve, shared.cost[first]);
        __syncthreads();
    }

    if (!owns) {
        return;
    }

    kept |= free && !taken;
    write_commands(commands, lane_index, capacity, lane.player, available, kept, choice, kAbsent);
}

template <typename Slot, typename Cell>
__device__ void act_on_one_unit(const Lane<Slot, Cell>& lane, const Shared& shared, const Usage& usage, const Window& window, int32_t* commands, int64_t lane_index) {
    const int32_t capacity = lane.size.capacity;
    const int32_t local = threadIdx.x;
    const bool owns = local < capacity;
    bool available = false;
    int32_t birth = kAbsent;

    if (owns) {
        available = lane.available(local);
        birth = lane.birth[lane.own_slot(local)];
        const draw::Key unit_key = draw::split(lane.key, kUnitStream);
        const float score = available ? draw::score(unit_key, static_cast<uint32_t>(local), 0.0f) : draw::kUnmarked;
        atomicMax(&shared.facts->pick, max_key(score, local));
    }

    __syncthreads();
    const int32_t drawn_unit = static_cast<int32_t>(0xffffffffu - static_cast<uint32_t>(shared.facts->pick));
    const bool chosen_unit = available && local == drawn_unit;
    Choice choice;
    bool consistent = false;

    if (chosen_unit) {
        const int32_t index = drawn_candidate(lane, draw::split(lane.key, kActionStream), 0u, local, window);
        choice = chosen(lane, index, static_cast<int32_t>(lane.cell[lane.own_slot(local)]), window);
        consistent = !blocked_by_pending(shared, lane.size, choice) && affordable(choice.cost, usage.reserved, usage.balance) && usage.other_affordable;
    }

    const bool filler = available && !chosen_unit;

    if (owns) {
        shared.birth[local] = birth;
        shared.marks[local] = filler ? kFiller : 0;
    }

    __syncthreads();

    if (!owns) {
        return;
    }

    const int32_t order = filler ? birth : kLast;
    int32_t rank = 0;

    for (int32_t slot = 0; slot < capacity; ++slot) {
        const int32_t before = (shared.marks[slot] & kFiller) != 0 ? shared.birth[slot] : kLast;
        rank += before < order || (before == order && slot < local);
    }

    write_commands(commands, lane_index, capacity, lane.player, available, chosen_unit && consistent, choice, chosen_unit ? 0 : rank + 1);
}

template <typename Slot, typename Cell>
__global__ void __launch_bounds__(nanorts::whole_act::kMaxThreads) act_kernel(Inputs<Slot, Cell> in, int32_t* commands, Extents size, Window window, int32_t units) {
    extern __shared__ uint64_t memory[];
    const int64_t lane_index = blockIdx.x;
    const Lane<Slot, Cell> lane(in, lane_index, size);
    const Shared shared = Shared::carve(memory, size);
    mark_pending_claims(lane, shared);
    const int32_t other_reserved = static_cast<int32_t>(shared.facts->reserved[lane.other]);
    const Usage usage{static_cast<int32_t>(shared.facts->reserved[lane.own]), lane.resources[lane.own], affordable(0, other_reserved, lane.resources[lane.other])};

    if (units == kOneUnit) {
        act_on_one_unit(lane, shared, usage, window, commands, lane_index);
    } else {
        act_on_every_unit(lane, shared, usage, window, commands, lane_index);
    }
}

template <typename Slot, typename Cell>
ffi::Error launch(cudaStream_t stream, nanorts::whole_act::Operands& operands, int32_t* commands, const Extents& size, const Window& window, int64_t units, int64_t lanes) {
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

    const int64_t rows = int64_t{nanorts::whole_act::kPlayers} * size.capacity;

    if (auto e = operands.next(rows * window.words, in.legality); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{2}, in.key); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{window.candidates}, in.log_weight); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{window.offsets} * kOffsetAxes, in.offsets); e.failure()) {
        return e;
    }

    const int threads = nanorts::whole_act::threads_for(size.capacity);
    const int64_t shared = shared_bytes(size.cells, size.capacity);
    static nanorts::whole_act::SharedOptIn opt_in;

    if (auto e = opt_in.allow(operands, nanorts::kernel(act_kernel<Slot, Cell>), shared); e.failure()) {
        return e;
    }

    act_kernel<Slot, Cell><<<static_cast<unsigned>(lanes), threads, static_cast<size_t>(shared), stream>>>(in, commands, size, window, static_cast<int32_t>(units));
    return nanorts::launched();
}

constexpr int64_t kLegalityOperand = nanorts::whole_act::kStateOperands + 2;
constexpr int64_t kKeyOperand = kLegalityOperand + 1;
constexpr int64_t kLogWeightOperand = kKeyOperand + 1;
constexpr int64_t kOffsetsOperand = kLogWeightOperand + 1;
constexpr int64_t kReadOperands = kOffsetsOperand + 1;

ffi::Error check_extras(const nanorts::whole_act::Operands& operands, int64_t capacity, int64_t types, Window& window) {
    auto words = operands.at(kLegalityOperand);
    auto key = operands.at(kKeyOperand);
    auto log_weight = operands.at(kLogWeightOperand);
    auto offsets = operands.at(kOffsetsOperand);

    if (!words.has_value()) {
        return words.error();
    }

    if (!key.has_value()) {
        return key.error();
    }

    if (!log_weight.has_value()) {
        return log_weight.error();
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

    if (key->element_type() != ffi::DataType::U32 || log_weight->element_type() != ffi::DataType::F32) {
        return operands.refused("the key is uint32 data and the log weights float32");
    }

    return ffi::Error::Success();
}

ffi::Error Act(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets results, std::string_view build_id, int64_t units) {
    constexpr std::string_view kTarget = "nanorts_random_act";

    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (args.size() != kReadOperands || results.size() != 1) {
        return ffi::Error::InvalidArgument(
            "nanorts_random_act takes the State leaves, the unit table, the player, the "
            "Legality words, the key, the log weights and the attack offsets, and returns the "
            "Commands");
    }

    auto commands = results.get<ffi::AnyBuffer>(0);

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

    const int64_t shared = shared_bytes(size.cells, size.capacity);

    if (size.cells <= 0 || size.capacity <= 0 || size.neutral < 0 || size.types <= 0 || shared > nanorts::whole_act::kSharedOptInLimit || nanorts::whole_act::threads_for(size.capacity) > nanorts::whole_act::kMaxThreads || (units != kEveryUnit && units != kOneUnit)) {
        return ffi::Error::InvalidArgument(
            "an act of every unit or of one, over a lane whose act fits one block's threads "
            "and shared memory");
    }

    int32_t* written = static_cast<int32_t*>((*commands)->untyped_data());

    if (lanes == 0) {
        return ffi::Error::Success();
    }

    using ffi::DataType;

    if (slot == DataType::S16 && cell == DataType::S16) {
        return launch<int16_t, int16_t>(stream, operands, written, size, window, units, lanes);
    }

    if (slot == DataType::S16) {
        return launch<int16_t, int32_t>(stream, operands, written, size, window, units, lanes);
    }

    if (cell == DataType::S16) {
        return launch<int32_t, int16_t>(stream, operands, written, size, window, units, lanes);
    }

    return launch<int32_t, int32_t>(stream, operands, written, size, window, units, lanes);
}

const nanorts::Kernels nanorts_random_act_kernels("nanorts_random_act", {nanorts::kernel(act_kernel<int16_t, int16_t>), nanorts::kernel(act_kernel<int16_t, int32_t>), nanorts::kernel(act_kernel<int32_t, int16_t>), nanorts::kernel(act_kernel<int32_t, int32_t>)});

}

NANORTS_HANDLER(nanorts_random_act, Act,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .RemainingArgs()
                    .RemainingRets()
                    .Attr<std::string_view>("build_id")
                    .Attr<int64_t>("units"));
