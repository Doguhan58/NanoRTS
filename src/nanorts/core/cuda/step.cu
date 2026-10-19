#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "admission.cuh"
#include "completion.cuh"
#include "nanorts_ffi.cuh"
#include "outcome.cuh"
#include "pending.cuh"

namespace {

namespace ffi = xla::ffi;
using namespace nanorts::core;

constexpr int kWarpsPerBlock = 4;

constexpr int64_t kSharedBytes = 48 * 1024;

struct PackedRetainedActor {
    int32_t birth;
    int32_t carried;
    int32_t cell;
    int32_t target_cell;
    int8_t kind;
    int8_t direction;
    int8_t produced_type;
    int8_t owner;
    int8_t unit_type;

    __device__ RetainedActor retained() const {
        return RetainedActor{kind, direction, target_cell, produced_type, birth, owner, unit_type, cell, carried};
    }
};

__host__ __device__ inline int64_t warp_bytes(int64_t owned) {
    return owned * static_cast<int64_t>(2 * sizeof(int32_t) + sizeof(PackedRetainedActor));
}

struct Lanes {
    char* data;
    int64_t stride;

    template <typename T>
    __device__ T* at(int64_t lane) const {
        return reinterpret_cast<T*>(data + lane * stride);
    }
};

enum Operand : int {
    kInKey,
    kInKind,
    kInDirection,
    kInNoneDuration,
    kInTargetCell,
    kInProducedType,
    kInRequestOrder,
    kInTerrain,
    kInTable,
    kInTimings,
    kInLeaves,
    kInOperands = kInLeaves + 23,
};

enum Leaf : int {
    kBirth,
    kUnitType,
    kOwner,
    kCell,
    kHitPoints,
    kCarriedResources,
    kPendingKind,
    kPendingDirection,
    kPendingNoneDuration,
    kPendingTargetCell,
    kPendingProducedType,
    kPendingIssuedAt,
    kPendingCompletesAt,
    kPendingOrder,
    kEntityAt,
    kTime,
    kPlayerResources,
    kNextBirth,
    kNextPendingOrder,
    kAlternatingConflictCount,
    kGameOver,
    kWinner,
    kSpawnOverflow,
    kStateLeaves,
};

enum Result : int {
    kAdmittedHarvest = kStateLeaves,
    kAdmittedReturn,
    kAdmittedProduce,
    kAdmittedAttack,
    kIsMember,
    kInt32Records,
    kIndexRecords,
    kInt8Records,
    kResults,
};

struct StepLanes {
    Lanes operand[kInOperands];
    Lanes result[kResults];
};

template <typename Slot, typename Cell>
__device__ LaneState<Slot, Cell> state_of(const Lanes* leaf, int64_t environment) {
    LaneState<Slot, Cell> state{};
    state.birth = leaf[kBirth].at<int32_t>(environment);
    state.unit_type = leaf[kUnitType].at<int8_t>(environment);
    state.owner = leaf[kOwner].at<int8_t>(environment);
    state.cell = leaf[kCell].at<Cell>(environment);
    state.hit_points = leaf[kHitPoints].at<int16_t>(environment);
    state.carried_resources = leaf[kCarriedResources].at<int32_t>(environment);
    state.pending_kind = leaf[kPendingKind].at<int8_t>(environment);
    state.pending_direction = leaf[kPendingDirection].at<int8_t>(environment);
    state.pending_none_duration = leaf[kPendingNoneDuration].at<int32_t>(environment);
    state.pending_target_cell = leaf[kPendingTargetCell].at<Cell>(environment);
    state.pending_produced_type = leaf[kPendingProducedType].at<int8_t>(environment);
    state.pending_issued_at = leaf[kPendingIssuedAt].at<int32_t>(environment);
    state.pending_completes_at = leaf[kPendingCompletesAt].at<int32_t>(environment);
    state.pending_order = leaf[kPendingOrder].at<int32_t>(environment);
    state.entity_at = leaf[kEntityAt].at<Slot>(environment);
    state.time = leaf[kTime].at<int32_t>(environment);
    state.player_resources = leaf[kPlayerResources].at<int32_t>(environment);
    state.next_birth = leaf[kNextBirth].at<int32_t>(environment);
    state.next_pending_order = leaf[kNextPendingOrder].at<int32_t>(environment);
    state.alternating_conflict_count = leaf[kAlternatingConflictCount].at<int32_t>(environment);
    state.spawn_overflow = leaf[kSpawnOverflow].at<int32_t>(environment);
    return state;
}

__device__ void clear_admitted_counters(const Admitted& admitted, int32_t types) {
    for (int player = 0; player < kPlayers; ++player) {
        admitted.harvest[player] = 0;
        admitted.return_[player] = 0;

        for (int type = 0; type < types; ++type) {
            admitted.produce[player * types + type] = 0;
        }

        for (int target_owner = 0; target_owner < kTargetOwners; ++target_owner) {
            admitted.attack_by_target_owner[player * kTargetOwners + target_owner] = 0;
        }
    }
}

template <typename Slot, typename Cell>
__device__ void retire_batch(const LaneState<Slot, Cell>& state, const Extents& size, int32_t time) {
    for (int32_t slot = threadIdx.x % 32; slot < size.entities; slot += 32) {
        const int32_t kind = state.pending_kind[slot];

        if (kind != kKindAbsent && state.pending_completes_at[slot] <= time) {
            state.clear_pending(slot);
        }
    }

    __syncwarp();
}

template <typename Slot, typename Cell>
__device__ void retain_batch(const LaneState<Slot, Cell>& state, const Extents& size, const int32_t* member_order, int32_t members, uint8_t* is_member, PackedRetainedActor* retained_actors) {
    const int lane = threadIdx.x % 32;

    for (int32_t slot = lane; slot < size.entities; slot += 32) {
        is_member[slot] = 0;
    }

    __syncwarp();

    for (int32_t position = lane; position < members; position += 32) {
        is_member[member_order[position]] = 1;
        const RetainedActor retained = retained_actor_at(state, member_order[position]);
        retained_actors[position] = PackedRetainedActor{
            retained.birth, retained.carried, retained.cell, retained.target_cell, static_cast<int8_t>(retained.kind), static_cast<int8_t>(retained.direction), static_cast<int8_t>(retained.produced_type), static_cast<int8_t>(retained.owner), static_cast<int8_t>(retained.unit_type),
        };
    }

    __syncwarp();
}

template <typename Slot, typename Cell>
__device__ int32_t admit_requests(const LaneState<Slot, Cell>& state, const Extents& size, const Content& content, const CommandTable& commands, Strategy strategy, nanorts::draws::Key root, const Admitted& admitted, int32_t* request_order, int32_t* held, uint64_t* listed) {
    const bool leader = threadIdx.x % 32 == 0;
    const int32_t requests = order_requests(state, size, commands, request_order, listed);
    int32_t holders = order_holders(state, size, held, listed);

    for (int32_t position = 0; position < requests; ++position) {
        const int32_t slot = request_order[position];
        const int32_t local = slot - size.neutral;
        const bool appends = state.pending_order[slot] == kAbsent;
        admit_one(state, size, content, held, holders, strategy, root, admitted, slot, commands.at(local));

        if (appends) {
            if (leader) {
                held[holders] = slot;
            }

            ++holders;
        }

        __syncwarp();
    }

    return holders;
}

template <typename Slot, typename Cell>
__global__ void __launch_bounds__(kWarpsPerBlock * 32) step(StepLanes lanes, Extents size, Strategy strategy, int64_t environments) {
    using Index = std::conditional_t<sizeof(Slot) == 2 && sizeof(Cell) == 2, int16_t, int32_t>;
    extern __shared__ __align__(8) int32_t working[];
    const int64_t environment = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;

    if (environment >= environments) {
        return;
    }

    const int lane = threadIdx.x % 32;
    const bool leader = lane == 0;
    const Lanes* operands = lanes.operand;
    const Lanes* results = lanes.result;

    LaneState<Slot, Cell> state = state_of<Slot, Cell>(results, environment);
    state.terrain = operands[kInTerrain].at<const uint8_t>(environment);
    const Content content{UnitTable{operands[kInTable].at<const int32_t>(environment), size.types}, operands[kInTimings].at<const int32_t>(environment)};
    const CommandTable commands{
        operands[kInKind].at<const int32_t>(environment), operands[kInDirection].at<const int32_t>(environment), operands[kInNoneDuration].at<const int32_t>(environment), operands[kInTargetCell].at<const int32_t>(environment), operands[kInProducedType].at<const int32_t>(environment), operands[kInRequestOrder].at<const int32_t>(environment),
    };

    const uint32_t* key = operands[kInKey].at<const uint32_t>(environment);
    const nanorts::draws::Key root{key[0], key[1]};
    const Admitted admitted{
        results[kAdmittedHarvest].at<int32_t>(environment),
        results[kAdmittedReturn].at<int32_t>(environment),
        results[kAdmittedProduce].at<int32_t>(environment),
        results[kAdmittedAttack].at<int32_t>(environment),
    };

    uint8_t* is_member = results[kIsMember].at<uint8_t>(environment);
    int32_t* int32_records = results[kInt32Records].at<int32_t>(environment);
    Index* index_records = results[kIndexRecords].at<Index>(environment);
    int8_t* int8_records = results[kInt8Records].at<int8_t>(environment);
    char* warp_memory = reinterpret_cast<char*>(working) + (threadIdx.x / 32) * warp_bytes(size.owned);
    int32_t* order = reinterpret_cast<int32_t*>(warp_memory);
    int32_t* held = order + size.owned;
    PackedRetainedActor* retained_actors = reinterpret_cast<PackedRetainedActor*>(held + size.owned);

    if (leader) {
        clear_admitted_counters(admitted, size.types);
    }

    __syncwarp();

    uint64_t* listed = reinterpret_cast<uint64_t*>(retained_actors);
    const int32_t holders = admit_requests(state, size, content, commands, strategy, root, admitted, order, held, listed);

    if (leader) {
        *state.time = wrapping_add(*state.time, 1);
    }

    __syncwarp();
    const int32_t time = *state.time;
    const int32_t members = batch_order(state, held, holders, order);
    retain_batch(state, size, order, members, is_member, retained_actors);
    retire_batch(state, size, time);

    for (int32_t position = 0; position < members; ++position) {
        const int32_t slot = order[position];
        Row records;
        take_turn(state, size, content, slot, retained_actors[position].retained(), records);

        if (leader) {
            records.store(int32_records + static_cast<int64_t>(slot) * kInt32Columns, index_records + static_cast<int64_t>(slot) * kIndexColumns, int8_records + static_cast<int64_t>(slot) * kInt8Columns);
        }

        __syncwarp();
    }

    decide(state, size, results[kGameOver].at<uint8_t>(environment), results[kWinner].at<int32_t>(environment));
}

ffi::ErrorOr<Lanes> lanes_of(const ffi::AnyBuffer& buffer, int64_t environments, int64_t elements, ffi::DataType type, std::string_view name) {
    const int64_t extent = nanorts::lanes(buffer);

    if (buffer.element_type() != type) {
        return ffi::Error::InvalidArgument("nanorts_step: " + std::string(name) + " has another dtype");
    }

    if (extent == 0 || (extent != 1 && extent != environments)) {
        return ffi::Error::InvalidArgument("nanorts_step: " + std::string(name) + " holds one row for every lane or one shared row");
    }

    const int64_t row_bytes = static_cast<int64_t>(buffer.size_bytes()) / extent;

    if (row_bytes != elements * static_cast<int64_t>(ffi::ByteWidth(type))) {
        return ffi::Error::InvalidArgument("nanorts_step: " + std::string(name) + " has another row");
    }

    return Lanes{static_cast<char*>(buffer.untyped_data()), extent == 1 ? 0 : row_bytes};
}

ffi::Error Step(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets rets, std::string_view build_id, int64_t strategy) {
    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (args.size() != kInOperands || rets.size() != kResults || strategy < kCancelBoth || strategy > kCancelAlternating) {
        return ffi::Error::InvalidArgument("nanorts_step takes 33 operands and a strategy, and returns 30 results");
    }

    std::vector<ffi::AnyBuffer> operand;
    std::vector<ffi::AnyBuffer> result;

    for (size_t k = 0; k < args.size(); ++k) {
        auto buffer = args.get<ffi::AnyBuffer>(k);

        if (!buffer.has_value()) {
            return buffer.error();
        }

        operand.push_back(*buffer);
    }

    for (size_t k = 0; k < rets.size(); ++k) {
        auto buffer = rets.get<ffi::AnyBuffer>(k);

        if (!buffer.has_value()) {
            return buffer.error();
        }

        result.push_back(**buffer);
    }

    const int64_t environments = nanorts::lanes(result[0]);
    auto slots = operand[kInLeaves + kBirth].dimensions();
    auto board = operand[kInLeaves + kEntityAt].dimensions();
    auto owned = operand[kInKind].dimensions();
    auto table = operand[kInTimings].dimensions();

    if (slots.size() != 2 || board.size() != 3 || owned.size() != 3 || table.size() != 3 || table[2] != kTimings || owned[1] != kPlayers) {
        return ffi::Error::InvalidArgument(
            "nanorts_step takes `birth[L, E]`, `entity_at[L, H, W]`, `kind[L, 2, C]` and "
            "`timings[L, T, 7]`");
    }

    Extents size{};
    size.height = static_cast<int32_t>(board[1]);
    size.width = static_cast<int32_t>(board[2]);
    size.cells = size.height * size.width;
    size.entities = static_cast<int32_t>(slots[1]);
    size.capacity = static_cast<int32_t>(owned[2]);
    size.owned = kPlayers * size.capacity;
    size.neutral = size.entities - size.owned;
    size.types = static_cast<int32_t>(table[1]);

    if (size.neutral < 0 || size.types <= 0 || size.cells <= 0) {
        return ffi::Error::InvalidArgument("nanorts_step: the extents disagree");
    }

    using ffi::DataType;
    const DataType cell = operand[kInLeaves + kCell].element_type();
    const DataType slot = operand[kInLeaves + kEntityAt].element_type();

    if ((cell != DataType::S16 && cell != DataType::S32) || (slot != DataType::S16 && slot != DataType::S32)) {
        return ffi::Error::InvalidArgument("nanorts_step: an index leaf is neither int16 nor int32");
    }

    struct Expected {
        int index;
        int64_t elements;
        DataType type;
        const char* name;
    };

    const int64_t commands = size.owned;
    const Expected reads[] = {
        {kInKey, 2, DataType::U32, "key"},
        {kInKind, commands, DataType::S32, "kind"},
        {kInDirection, commands, DataType::S32, "direction"},
        {kInNoneDuration, commands, DataType::S32, "none_duration"},
        {kInTargetCell, commands, DataType::S32, "target_cell"},
        {kInProducedType, commands, DataType::S32, "produced_type"},
        {kInRequestOrder, commands, DataType::S32, "request_order"},
        {kInTerrain, size.cells, DataType::PRED, "terrain"},
        {kInTable, int64_t{size.types} * (kProduces + size.types), DataType::S32, "table"},
        {kInTimings, int64_t{size.types} * kTimings, DataType::S32, "timings"},
    };

    StepLanes lanes{};

    for (const Expected& e : reads) {
        auto read = lanes_of(operand[e.index], environments, e.elements, e.type, e.name);

        if (read.has_error()) {
            return read.error();
        }

        lanes.operand[e.index] = *read;
    }

    const int64_t per_slot = size.entities;
    const Expected leaves[kStateLeaves] = {
        {kBirth, per_slot, DataType::S32, "birth"},
        {kUnitType, per_slot, DataType::S8, "unit_type"},
        {kOwner, per_slot, DataType::S8, "owner"},
        {kCell, per_slot, cell, "cell"},
        {kHitPoints, per_slot, DataType::S16, "hit_points"},
        {kCarriedResources, per_slot, DataType::S32, "carried_resources"},
        {kPendingKind, per_slot, DataType::S8, "pending_kind"},
        {kPendingDirection, per_slot, DataType::S8, "pending_direction"},
        {kPendingNoneDuration, per_slot, DataType::S32, "pending_none_duration"},
        {kPendingTargetCell, per_slot, cell, "pending_target_cell"},
        {kPendingProducedType, per_slot, DataType::S8, "pending_produced_type"},
        {kPendingIssuedAt, per_slot, DataType::S32, "pending_issued_at"},
        {kPendingCompletesAt, per_slot, DataType::S32, "pending_completes_at"},
        {kPendingOrder, per_slot, DataType::S32, "pending_order"},
        {kEntityAt, size.cells, slot, "entity_at"},
        {kTime, 1, DataType::S32, "time"},
        {kPlayerResources, kPlayers, DataType::S32, "player_resources"},
        {kNextBirth, 1, DataType::S32, "next_birth"},
        {kNextPendingOrder, 1, DataType::S32, "next_pending_order"},
        {kAlternatingConflictCount, 1, DataType::S32, "alternating_conflict_count"},
        {kGameOver, 1, DataType::PRED, "game_over"},
        {kWinner, 1, DataType::S32, "winner"},
        {kSpawnOverflow, kPlayers, DataType::S32, "spawn_overflow"},
    };

    for (const Expected& e : leaves) {
        const ffi::AnyBuffer& source = operand[kInLeaves + e.index];
        ffi::AnyBuffer& place = result[e.index];

        if (nanorts::lanes(source) != environments || source.size_bytes() != place.size_bytes()) {
            return ffi::Error::InvalidArgument("nanorts_step: a written leaf's result has its operand's lanes");
        }

        auto read = lanes_of(source, environments, e.elements, e.type, e.name);

        if (read.has_error()) {
            return read.error();
        }

        auto written = lanes_of(place, environments, e.elements, e.type, e.name);

        if (written.has_error()) {
            return written.error();
        }

        if (source.untyped_data() != place.untyped_data() && place.size_bytes() > 0 && cudaMemcpyAsync(place.untyped_data(), source.untyped_data(), place.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            return ffi::Error::Internal("nanorts_step: cannot copy a leaf into place");
        }

        lanes.result[e.index] = *written;
    }

    const Expected writes[] = {
        {kAdmittedHarvest, kPlayers, DataType::S32, "admitted_harvest"},
        {kAdmittedReturn, kPlayers, DataType::S32, "admitted_return"},
        {kAdmittedProduce, kPlayers * int64_t{size.types}, DataType::S32, "admitted_produce"},
        {kAdmittedAttack, kPlayers * int64_t{kTargetOwners}, DataType::S32, "admitted_attack_by_target_owner"},
        {kIsMember, per_slot, DataType::PRED, "member"},
        {kInt32Records, per_slot * kInt32Columns, DataType::S32, "wide"},
        {kIndexRecords, per_slot * kIndexColumns, slot == DataType::S16 && cell == DataType::S16 ? DataType::S16 : DataType::S32, "index"},
        {kInt8Records, per_slot * kInt8Columns, DataType::S8, "narrow"},
    };

    for (const Expected& e : writes) {
        auto written = lanes_of(result[e.index], environments, e.elements, e.type, e.name);

        if (written.has_error()) {
            return written.error();
        }

        lanes.result[e.index] = *written;
    }

    const int64_t per_warp = warp_bytes(size.owned);
    const int64_t warps = std::min<int64_t>(kWarpsPerBlock, kSharedBytes / per_warp);

    if (warps < 1) {
        return ffi::Error::InvalidArgument("nanorts_step: one warp's working set exceeds the shared memory of a block");
    }

    if (environments == 0) {
        return ffi::Error::Success();
    }

    const unsigned blocks = static_cast<unsigned>((environments + warps - 1) / warps);
    const unsigned threads = static_cast<unsigned>(warps * 32);
    const size_t shared = static_cast<size_t>(warps * per_warp);
    const Strategy chosen = static_cast<Strategy>(strategy);

    if (slot == DataType::S16 && cell == DataType::S16) {
        step<int16_t, int16_t><<<blocks, threads, shared, stream>>>(lanes, size, chosen, environments);
    } else if (slot == DataType::S16) {
        step<int16_t, int32_t><<<blocks, threads, shared, stream>>>(lanes, size, chosen, environments);
    } else if (cell == DataType::S16) {
        step<int32_t, int16_t><<<blocks, threads, shared, stream>>>(lanes, size, chosen, environments);
    } else {
        step<int32_t, int32_t><<<blocks, threads, shared, stream>>>(lanes, size, chosen, environments);
    }

    return nanorts::launched();
}

const nanorts::Kernels nanorts_step_kernels("nanorts_step", {nanorts::kernel(step<int16_t, int16_t>), nanorts::kernel(step<int16_t, int32_t>), nanorts::kernel(step<int32_t, int16_t>), nanorts::kernel(step<int32_t, int32_t>)});

}

NANORTS_HANDLER(nanorts_step, Step,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .RemainingArgs()
                    .RemainingRets()
                    .Attr<std::string_view>("build_id")
                    .Attr<int64_t>("strategy"));
