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

constexpr int64_t kCacheLineBytes = 128;

template <typename Cell>
struct PackedRetainedActor {
    int32_t birth;
    int32_t carried;
    Cell cell;
    Cell target_cell;
    int8_t kind;
    int8_t direction;
    int8_t produced_type;
    int8_t owner;
    int8_t unit_type;

    __device__ RetainedActor retained() const {
        return RetainedActor{kind, direction, target_cell, produced_type, birth, owner, unit_type, cell, carried};
    }
};

template <typename Slot, typename Cell>
__host__ __device__ inline int64_t warp_bytes(int64_t owned) {
    return owned * static_cast<int64_t>(2 * sizeof(Slot) + sizeof(PackedRetainedActor<Cell>));
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

__host__ __device__ inline const Lanes& row_of(const StepLanes& lanes, int row) {
    return row < kInOperands ? lanes.operand[row] : lanes.result[row - kInOperands];
}

__device__ inline void prefetch_bytes(const char* row, int64_t bytes) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(row);
    const uintptr_t line = (begin & ~static_cast<uintptr_t>(kCacheLineBytes - 1)) + (threadIdx.x % 32) * kCacheLineBytes;

    if (line < begin + bytes) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(line < begin ? begin : line));
    }
}

struct LaneRows {
    const StepLanes& lanes;
    int64_t environment;

    template <typename T>
    __device__ T* operand(int index) const {
        return lanes.operand[index].at<T>(environment);
    }

    template <typename T>
    __device__ T* result(int index) const {
        return lanes.result[index].at<T>(environment);
    }

    __device__ void prefetch(int row) const {
        const Lanes& place = row_of(lanes, row);
        prefetch_bytes(place.at<const char>(environment), place.stride);
    }
};

constexpr int kRows = kInOperands + kResults;

constexpr int kGroupsPerLaunch = 16;

struct GroupLanes {
    char* data[kGroupsPerLaunch][kRows];
    int64_t stride[kRows];
    uint32_t first_block[kGroupsPerLaunch];
    uint32_t lane_end[kGroupsPerLaunch];
};

struct GroupRows {
    char* const* data;
    const int64_t* stride;
    int64_t environment;

    template <typename T>
    __device__ T* operand(int index) const {
        return reinterpret_cast<T*>(data[index] + environment * stride[index]);
    }

    template <typename T>
    __device__ T* result(int index) const {
        return reinterpret_cast<T*>(data[kInOperands + index] + environment * stride[kInOperands + index]);
    }

    __device__ void prefetch(int row) const {
        prefetch_bytes(data[row] + environment * stride[row], stride[row]);
    }
};

template <typename Slot, typename Cell, typename Rows>
__device__ LaneState<Slot, Cell> state_of(const Rows& rows) {
    LaneState<Slot, Cell> state{};
    state.birth = rows.template result<int32_t>(kBirth);
    state.unit_type = rows.template result<int8_t>(kUnitType);
    state.owner = rows.template result<int8_t>(kOwner);
    state.cell = rows.template result<Cell>(kCell);
    state.hit_points = rows.template result<int16_t>(kHitPoints);
    state.carried_resources = rows.template result<int32_t>(kCarriedResources);
    state.pending_kind = rows.template result<int8_t>(kPendingKind);
    state.pending_direction = rows.template result<int8_t>(kPendingDirection);
    state.pending_none_duration = rows.template result<int32_t>(kPendingNoneDuration);
    state.pending_target_cell = rows.template result<Cell>(kPendingTargetCell);
    state.pending_produced_type = rows.template result<int8_t>(kPendingProducedType);
    state.pending_issued_at = rows.template result<int32_t>(kPendingIssuedAt);
    state.pending_completes_at = rows.template result<int32_t>(kPendingCompletesAt);
    state.pending_order = rows.template result<int32_t>(kPendingOrder);
    state.entity_at = rows.template result<Slot>(kEntityAt);
    state.time = rows.template result<int32_t>(kTime);
    state.player_resources = rows.template result<int32_t>(kPlayerResources);
    state.next_birth = rows.template result<int32_t>(kNextBirth);
    state.next_pending_order = rows.template result<int32_t>(kNextPendingOrder);
    state.alternating_conflict_count = rows.template result<int32_t>(kAlternatingConflictCount);
    state.spawn_overflow = rows.template result<int32_t>(kSpawnOverflow);
    return state;
}

template <typename Rows>
__device__ void carry_row(const Rows& rows, int leaf, int elements) {
    const int32_t* from = rows.template operand<const int32_t>(kInLeaves + leaf);
    int32_t* to = rows.template result<int32_t>(leaf);

    for (int k = 0; k < elements; ++k) {
        to[k] = from[k];
    }
}

template <typename Rows>
__device__ void carry_counters(const Rows& rows) {
    carry_row(rows, kTime, 1);
    carry_row(rows, kPlayerResources, kPlayers);
    carry_row(rows, kNextBirth, 1);
    carry_row(rows, kNextPendingOrder, 1);
    carry_row(rows, kAlternatingConflictCount, 1);
    carry_row(rows, kSpawnOverflow, kPlayers);
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
__device__ void retain_batch(const LaneState<Slot, Cell>& state, const Extents& size, const Slot* member_order, int32_t members, uint8_t* is_member, PackedRetainedActor<Cell>* retained_actors) {
    const int lane = threadIdx.x % 32;

    for (int32_t slot = lane; slot < size.entities; slot += 32) {
        is_member[slot] = 0;
    }

    __syncwarp();

    for (int32_t position = lane; position < members; position += 32) {
        is_member[member_order[position]] = 1;
        const RetainedActor retained = retained_actor_at(state, member_order[position]);
        retained_actors[position] = PackedRetainedActor<Cell>{
            retained.birth, retained.carried, static_cast<Cell>(retained.cell), static_cast<Cell>(retained.target_cell), static_cast<int8_t>(retained.kind), static_cast<int8_t>(retained.direction), static_cast<int8_t>(retained.produced_type), static_cast<int8_t>(retained.owner), static_cast<int8_t>(retained.unit_type),
        };
    }

    __syncwarp();
}

template <typename Slot, typename Cell>
__device__ int32_t admit_requests(const LaneState<Slot, Cell>& state, const Extents& size, const Content& content, const CommandTable& commands, Strategy strategy, nanorts::draws::Key root, const Admitted& admitted, Slot* request_order, Slot* held, uint64_t* listed) {
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
                held[holders] = static_cast<Slot>(slot);
            }

            ++holders;
        }

        __syncwarp();
    }

    return holders;
}

template <typename Rows>
__device__ void prefetch_scanned_rows(const Rows& rows) {
    rows.prefetch(kInOperands + kBirth);
    rows.prefetch(kInKind);
    rows.prefetch(kInOperands + kPendingOrder);
    rows.prefetch(kInOperands + kPendingKind);
}

template <typename Slot, typename Cell, typename Rows>
__device__ __forceinline__ void step_lane(const Rows& rows, const Extents& size, Strategy strategy) {
    using Index = std::conditional_t<sizeof(Slot) == 2 && sizeof(Cell) == 2, int16_t, int32_t>;
    extern __shared__ __align__(8) int32_t working[];
    const int lane = threadIdx.x % 32;
    const bool leader = lane == 0;

    LaneState<Slot, Cell> state = state_of<Slot, Cell>(rows);
    state.terrain = rows.template operand<const uint8_t>(kInTerrain);
    const Content content{UnitTable{rows.template operand<const int32_t>(kInTable), size.types}, rows.template operand<const int32_t>(kInTimings)};
    const CommandTable commands{
        rows.template operand<const int32_t>(kInKind), rows.template operand<const int32_t>(kInDirection), rows.template operand<const int32_t>(kInNoneDuration), rows.template operand<const int32_t>(kInTargetCell), rows.template operand<const int32_t>(kInProducedType), rows.template operand<const int32_t>(kInRequestOrder),
    };

    prefetch_scanned_rows(rows);

    const uint32_t* key = rows.template operand<const uint32_t>(kInKey);
    const nanorts::draws::Key root{key[0], key[1]};
    const Admitted admitted{
        rows.template result<int32_t>(kAdmittedHarvest),
        rows.template result<int32_t>(kAdmittedReturn),
        rows.template result<int32_t>(kAdmittedProduce),
        rows.template result<int32_t>(kAdmittedAttack),
    };

    uint8_t* is_member = rows.template result<uint8_t>(kIsMember);
    int32_t* int32_records = rows.template result<int32_t>(kInt32Records);
    Index* index_records = rows.template result<Index>(kIndexRecords);
    int8_t* int8_records = rows.template result<int8_t>(kInt8Records);
    char* warp_memory = reinterpret_cast<char*>(working) + (threadIdx.x / 32) * warp_bytes<Slot, Cell>(size.owned);
    Slot* order = reinterpret_cast<Slot*>(warp_memory);
    Slot* held = order + size.owned;
    PackedRetainedActor<Cell>* retained_actors = reinterpret_cast<PackedRetainedActor<Cell>*>(held + size.owned);

    if (leader) {
        carry_counters(rows);
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

    decide(state, size, rows.template result<uint8_t>(kGameOver), rows.template result<int32_t>(kWinner));
}

template <typename Slot, typename Cell>
__global__ void __launch_bounds__(kWarpsPerBlock * 32) step(StepLanes lanes, Extents size, Strategy strategy, int64_t environments) {
    const uint32_t environment = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (environment >= environments) {
        return;
    }

    step_lane<Slot, Cell>(LaneRows{lanes, environment}, size, strategy);
}

template <typename Slot, typename Cell>
__global__ void __launch_bounds__(kWarpsPerBlock * 32) step_groups(const __grid_constant__ GroupLanes lanes, Extents size, Strategy strategy) {
    int32_t group = 0;

#pragma unroll
    for (int32_t later = 1; later < kGroupsPerLaunch; ++later) {
        group += blockIdx.x >= lanes.first_block[later];
    }

    const uint32_t environment = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (environment >= lanes.lane_end[group]) {
        return;
    }

    step_lane<Slot, Cell>(GroupRows{lanes.data[group], lanes.stride, environment}, size, strategy);
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

struct BoundLanes {
    StepLanes lanes;
    Extents size;
    ffi::DataType slot;
    ffi::DataType cell;
    int64_t environments;
};

ffi::Error collect_buffers(ffi::RemainingArgs args, ffi::RemainingRets rets, std::vector<ffi::AnyBuffer>& operand, std::vector<ffi::AnyBuffer>& result) {
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

    return ffi::Error::Success();
}

ffi::ErrorOr<BoundLanes> bind_lanes(cudaStream_t stream, const ffi::AnyBuffer* operand, ffi::AnyBuffer* result) {
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
        const bool aliased = e.index < kTime;

        if (nanorts::lanes(place) != environments || (aliased && (nanorts::lanes(source) != environments || source.size_bytes() != place.size_bytes()))) {
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

        if (aliased && source.untyped_data() != place.untyped_data() && place.size_bytes() > 0 && cudaMemcpyAsync(place.untyped_data(), source.untyped_data(), place.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            return ffi::Error::Internal("nanorts_step: cannot copy a leaf into place");
        }

        if (!aliased) {
            lanes.operand[kInLeaves + e.index] = *read;
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

    return BoundLanes{lanes, size, slot, cell, environments};
}

template <typename Launch>
void with_index_types(ffi::DataType slot, ffi::DataType cell, Launch launch) {
    using ffi::DataType;

    if (slot == DataType::S16 && cell == DataType::S16) {
        launch(int16_t{}, int16_t{});
    } else if (slot == DataType::S16) {
        launch(int16_t{}, int32_t{});
    } else if (cell == DataType::S16) {
        launch(int32_t{}, int16_t{});
    } else {
        launch(int32_t{}, int32_t{});
    }
}

constexpr int64_t kIndexableThreads = int64_t{1} << 32;

struct Geometry {
    int64_t warps;
    unsigned threads;
    size_t shared;

    int64_t blocks(int64_t environments) const {
        return (environments + warps - 1) / warps;
    }

    bool indexable(int64_t blocks) const {
        return blocks * threads <= kIndexableThreads;
    }
};

ffi::ErrorOr<Geometry> geometry_for(const BoundLanes& lanes, const char* target) {
    int64_t per_warp = 0;
    with_index_types(lanes.slot, lanes.cell, [&](auto slot, auto cell) { per_warp = warp_bytes<decltype(slot), decltype(cell)>(lanes.size.owned); });
    const int64_t warps = std::min<int64_t>(kWarpsPerBlock, kSharedBytes / per_warp);

    if (warps < 1) {
        return ffi::Error::InvalidArgument(std::string(target) + ": one warp's working set exceeds the shared memory of a block");
    }

    return Geometry{warps, static_cast<unsigned>(warps * 32), static_cast<size_t>(warps * per_warp)};
}

ffi::Error Step(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets rets, std::string_view build_id, int64_t strategy) {
    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (args.size() != kInOperands || rets.size() != kResults || strategy < kCancelBoth || strategy > kCancelAlternating) {
        return ffi::Error::InvalidArgument("nanorts_step takes 33 operands and a strategy, and returns 31 results");
    }

    std::vector<ffi::AnyBuffer> operand;
    std::vector<ffi::AnyBuffer> result;

    if (ffi::Error error = collect_buffers(args, rets, operand, result); error.failure()) {
        return error;
    }

    auto bound = bind_lanes(stream, operand.data(), result.data());

    if (bound.has_error()) {
        return bound.error();
    }

    auto geometry = geometry_for(*bound, "nanorts_step");

    if (geometry.has_error()) {
        return geometry.error();
    }

    if (!geometry->indexable(geometry->blocks(bound->environments))) {
        return ffi::Error::InvalidArgument("nanorts_step: one launch indexes at most 2^27 lanes");
    }

    if (bound->environments == 0) {
        return ffi::Error::Success();
    }

    const BoundLanes& lanes = *bound;
    const Geometry& launch = *geometry;
    const Strategy chosen = static_cast<Strategy>(strategy);

    with_index_types(lanes.slot, lanes.cell, [&](auto slot, auto cell) {
        step<decltype(slot), decltype(cell)><<<static_cast<unsigned>(launch.blocks(lanes.environments)), launch.threads, launch.shared, stream>>>(lanes.lanes, lanes.size, chosen, lanes.environments);
    });

    return nanorts::launched();
}

bool same_extents(const Extents& a, const Extents& b) {
    return a.width == b.width && a.height == b.height && a.cells == b.cells && a.entities == b.entities && a.neutral == b.neutral && a.capacity == b.capacity && a.owned == b.owned && a.types == b.types;
}

ffi::Error StepGroups(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets rets, std::string_view build_id, int64_t strategy, int64_t groups) {
    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (groups < 1 || args.size() != static_cast<size_t>(groups * kInOperands) || rets.size() != static_cast<size_t>(groups * kResults) || strategy < kCancelBoth || strategy > kCancelAlternating) {
        return ffi::Error::InvalidArgument("nanorts_step_groups takes 33 operands per group and a strategy, and returns 31 results per group");
    }

    std::vector<ffi::AnyBuffer> operand;
    std::vector<ffi::AnyBuffer> result;

    if (ffi::Error error = collect_buffers(args, rets, operand, result); error.failure()) {
        return error;
    }

    std::vector<BoundLanes> bound;

    for (int64_t group = 0; group < groups; ++group) {
        auto group_lanes = bind_lanes(stream, operand.data() + group * kInOperands, result.data() + group * kResults);

        if (group_lanes.has_error()) {
            return group_lanes.error();
        }

        if (!bound.empty() && (!same_extents(group_lanes->size, bound[0].size) || group_lanes->slot != bound[0].slot || group_lanes->cell != bound[0].cell)) {
            return ffi::Error::InvalidArgument("nanorts_step_groups: the groups' Buckets disagree");
        }

        bound.push_back(*group_lanes);
    }

    GroupLanes table{};

    for (int row = 0; row < kRows; ++row) {
        bool strided = false;

        for (const BoundLanes& group_lanes : bound) {
            if (group_lanes.environments < 2) {
                continue;
            }

            const int64_t stride = row_of(group_lanes.lanes, row).stride;

            if (strided && stride != table.stride[row]) {
                return ffi::Error::InvalidArgument("nanorts_step_groups: the groups' rows disagree");
            }

            table.stride[row] = stride;
            strided = true;
        }
    }

    const Extents& size = bound[0].size;
    auto geometry = geometry_for(bound[0], "nanorts_step_groups");

    if (geometry.has_error()) {
        return geometry.error();
    }

    const Geometry& launch = *geometry;
    const Strategy chosen = static_cast<Strategy>(strategy);

    for (int64_t chunk = 0; chunk < groups; chunk += kGroupsPerLaunch) {
        const int64_t chunk_groups = std::min<int64_t>(kGroupsPerLaunch, groups - chunk);
        int64_t blocks = 0;

        for (int32_t group = 0; group < kGroupsPerLaunch; ++group) {
            if (group >= chunk_groups) {
                table.first_block[group] = UINT32_MAX;
                table.lane_end[group] = 0;
                continue;
            }

            const BoundLanes& group_lanes = bound[static_cast<size_t>(chunk + group)];
            const int64_t first = blocks * launch.warps;

            for (int row = 0; row < kRows; ++row) {
                table.data[group][row] = reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(row_of(group_lanes.lanes, row).data) - static_cast<uintptr_t>(first * table.stride[row]));
            }

            table.first_block[group] = static_cast<uint32_t>(blocks);
            table.lane_end[group] = static_cast<uint32_t>(first + group_lanes.environments);
            blocks += launch.blocks(group_lanes.environments);

            if (!launch.indexable(blocks)) {
                return ffi::Error::InvalidArgument("nanorts_step_groups: one launch indexes at most 2^27 lanes");
            }
        }

        if (blocks == 0) {
            continue;
        }

        with_index_types(bound[0].slot, bound[0].cell, [&](auto slot, auto cell) {
            step_groups<decltype(slot), decltype(cell)><<<static_cast<unsigned>(blocks), launch.threads, launch.shared, stream>>>(table, size, chosen);
        });

        if (ffi::Error error = nanorts::launched(); error.failure()) {
            return error;
        }
    }

    return ffi::Error::Success();
}

const nanorts::Kernels nanorts_step_kernels("nanorts_step", {nanorts::kernel(step<int16_t, int16_t>), nanorts::kernel(step<int16_t, int32_t>), nanorts::kernel(step<int32_t, int16_t>), nanorts::kernel(step<int32_t, int32_t>)});

const nanorts::Kernels nanorts_step_groups_kernels("nanorts_step_groups", {nanorts::kernel(step_groups<int16_t, int16_t>), nanorts::kernel(step_groups<int16_t, int32_t>), nanorts::kernel(step_groups<int32_t, int16_t>), nanorts::kernel(step_groups<int32_t, int32_t>)});

}

NANORTS_HANDLER(nanorts_step, Step,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .RemainingArgs()
                    .RemainingRets()
                    .Attr<std::string_view>("build_id")
                    .Attr<int64_t>("strategy"));

NANORTS_HANDLER(nanorts_step_groups, StepGroups,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .RemainingArgs()
                    .RemainingRets()
                    .Attr<std::string_view>("build_id")
                    .Attr<int64_t>("strategy")
                    .Attr<int64_t>("groups"));
