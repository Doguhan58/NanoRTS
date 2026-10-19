#include <cstdint>
#include <string>
#include <string_view>

#include "../kinds/cuda/action_kind.cuh"
#include "../kinds/cuda/attack.cuh"
#include "../kinds/cuda/harvest.cuh"
#include "../kinds/cuda/move.cuh"
#include "../kinds/cuda/none.cuh"
#include "../kinds/cuda/produce.cuh"
#include "../kinds/cuda/return_.cuh"
#include "board.cuh"
#include "nanorts_ffi.cuh"

namespace {

namespace ffi = xla::ffi;
using namespace nanorts::core;

constexpr int kWarpsPerBlock = 8;
constexpr int kKinds = 6;
constexpr unsigned kAllLanes = 0xffffffffu;

constexpr int kMoveColumn = kKinds;
constexpr int kHarvestColumn = kMoveColumn + kDirections;
constexpr int kReturnColumn = kHarvestColumn + kDirections;
constexpr int kProduceDirectionColumn = kReturnColumn + kDirections;
constexpr int kProduceTypeColumn = kProduceDirectionColumn + kDirections;

constexpr int64_t kSharedBytes = 48 * 1024;

constexpr int kWordBits = 32;
constexpr int kWords = 3;

template <typename T>
struct Lanes {
    const T* data;
    int64_t stride;

    __device__ const T* at(int64_t lane) const {
        return data + lane * stride;
    }
};

struct Inputs {
    Lanes<int32_t> birth;
    Lanes<int8_t> unit_type;
    Lanes<int8_t> owner;
    Lanes<int32_t> carried;
    Lanes<void> cell;
    Lanes<void> entity_at;
    Lanes<int32_t> resources;
    Lanes<uint8_t> terrain;
    Lanes<int32_t> table;
};

struct Extents {
    int32_t width;
    int32_t height;
    int32_t cells;
    int32_t entities;
    int32_t neutral;
    int32_t owned;
    int32_t types;
    int32_t offsets;
    int32_t side;
    int32_t radius;
    int32_t columns;
};

template <>
__device__ const void* Lanes<void>::at(int64_t lane) const {
    return static_cast<const char*>(data) + lane * stride;
}

struct OwnedUnit {
    Actor actor;
    int32_t cell;
};

template <typename Slot>
struct Staged {
    int32_t* listed_count;
    int32_t* carried;
    int32_t* cell;
    Slot* entity_at;
    int16_t* listed;
    int8_t* actor_type;
    int8_t* unit_type;
    int8_t* owner;
    uint8_t* terrain;

    static int64_t bytes(const Extents& size) {
        return 4 + int64_t{size.owned} * (4 + 4 + 2 + 1) + int64_t{size.cells} * (sizeof(Slot) + 1) + int64_t{size.entities} * 2;
    }

    __device__ static Staged at(char* base, const Extents& size) {
        Staged staged;
        staged.listed_count = reinterpret_cast<int32_t*>(base);
        staged.carried = staged.listed_count + 1;
        staged.cell = staged.carried + size.owned;
        staged.entity_at = reinterpret_cast<Slot*>(staged.cell + size.owned);
        staged.listed = reinterpret_cast<int16_t*>(staged.entity_at + size.cells);
        staged.actor_type = reinterpret_cast<int8_t*>(staged.listed + size.owned);
        staged.unit_type = staged.actor_type + size.owned;
        staged.owner = staged.unit_type + size.entities;
        staged.terrain = reinterpret_cast<uint8_t*>(staged.owner + size.entities);
        return staged;
    }
};

template <typename Slot>
__device__ int8_t owner_at(const Board<Slot>& board, int32_t cell) {
    const int32_t occupant = static_cast<int32_t>(board.entity_at[cell]);
    return occupant == kAbsent ? kOwnerNeutral : board.owner[clip(occupant, 0, board.entity_capacity - 1)];
}

template <typename Cell>
__device__ OwnedUnit owned_unit_at(const Inputs& inputs, int64_t environment, const Extents& size, int32_t local) {
    const int32_t slot = size.neutral + local;
    const bool active = inputs.birth.at(environment)[slot] != kAbsent;
    return OwnedUnit{
        Actor{active ? clip(inputs.unit_type.at(environment)[slot], 0, size.types - 1) : -1, inputs.owner.at(environment)[slot], inputs.carried.at(environment)[slot]},
        static_cast<int32_t>(static_cast<const Cell*>(inputs.cell.at(environment))[slot]),
    };
}

struct Rows {
    using Element = uint8_t;
    uint8_t* row;

    __device__ static int32_t row_bytes(const Extents& size) {
        return size.columns;
    }

    __device__ Rows(uint8_t* out, int64_t slot, const Extents& size) : row(out + slot * size.columns) {}

    __device__ void clear(const Extents& size, int lane) {
        for (int column = lane; column < size.columns; column += 32) {
            row[column] = 0;
        }
    }

    __device__ void put(int first, bool in_row, bool bit, int lane) {
        if (in_row) {
            row[first + lane] = bit;
        }
    }

    __device__ void store(int) {}
};

struct Words {
    using Element = uint32_t;
    uint32_t* words;
    uint32_t word = 0;

    __device__ static int32_t row_bytes(const Extents&) {
        return kWords * static_cast<int32_t>(sizeof(uint32_t));
    }

    __device__ Words(uint32_t* out, int64_t slot, const Extents&) : words(out + slot * kWords) {}

    __device__ void clear(const Extents&, int lane) {
        if (lane < kWords) {
            words[lane] = 0;
        }
    }

    __device__ void put(int first, bool in_row, bool bit, int lane) {
        const uint32_t ballot = __ballot_sync(kAllLanes, in_row && bit);
        const int shift = first - kWordBits * lane;
        word |= shift >= 0 ? __funnelshift_lc(0u, ballot, static_cast<unsigned>(shift)) : __funnelshift_rc(ballot, 0u, static_cast<unsigned>(-shift));
    }

    __device__ void store(int lane) {
        if (lane < kWords) {
            words[lane] = word;
        }
    }
};

template <typename RowForm>
__device__ bool put_produce_types(const UnitTable& table, const Actor& actor, int32_t balance, const Extents& size, bool has_free_direction, RowForm& form, int lane) {
    bool any_type = false;

    for (int base = 0; base < size.types; base += 32) {
        const int type = base + lane;
        const bool valid = type < size.types && produce_type_is_valid(table, actor, balance, type);
        form.put(kProduceTypeColumn + base, type < size.types, valid && has_free_direction, lane);
        any_type |= __any_sync(kAllLanes, valid);
    }

    return any_type;
}

template <typename RowForm, typename Slot>
__device__ bool put_attack_offsets(const Board<Slot>& board, const UnitTable& table, const OwnedUnit& unit, const Extents& size, RowForm& form, int lane) {
    const Actor& actor = unit.actor;
    const int32_t actor_x = floor_mod(unit.cell, size.width);
    const int32_t actor_y = floor_div(unit.cell, size.width);
    const int32_t attack_range = table.at(actor.unit_type, kAttackRange);
    const int attack_column = kProduceTypeColumn + size.types;
    bool any_attack = false;

    for (int base = 0; base < size.offsets; base += 32) {
        const int offset = base + lane;
        bool attack = false;

        if (offset < size.offsets) {
            const int32_t dx = offset % size.side - size.radius;
            const int32_t dy = offset / size.side - size.radius;
            const int32_t target_x = actor_x + dx;
            const int32_t target_y = actor_y + dy;
            const bool on_board = target_x >= 0 && target_x < size.width && target_y >= 0 && target_y < size.height;
            const int8_t target_owner = on_board ? owner_at(board, target_y * size.width + target_x) : kOwnerNeutral;
            attack = can_attack(table, actor, target_owner, dx * dx + dy * dy <= attack_range * attack_range);
        }

        form.put(attack_column + base, offset < size.offsets, attack, lane);
        any_attack |= __any_sync(kAllLanes, attack);
    }

    return any_attack;
}

template <typename RowForm, typename Slot>
__device__ void write_row(const Board<Slot>& board, const UnitTable& table, const OwnedUnit& unit, int32_t balance, const Extents& size, RowForm form, int lane) {
    if (unit.actor.unit_type < 0) {
        form.clear(size, lane);
        return;
    }

    const Actor& actor = unit.actor;

    const bool is_direction_lane = lane < kDirections;
    const Neighbour adjacent = neighbour(board, unit.cell, lane % kDirections);
    const unsigned move_directions = __ballot_sync(kAllLanes, is_direction_lane && move_is_valid(table, actor, adjacent));
    const unsigned harvest_directions = __ballot_sync(kAllLanes, is_direction_lane && harvest_is_valid(table, actor, adjacent));
    const unsigned return_directions = __ballot_sync(kAllLanes, is_direction_lane && return_is_valid(table, actor, adjacent));
    const unsigned free_directions = __ballot_sync(kAllLanes, is_direction_lane && adjacent.free);

    const bool any_type = put_produce_types(table, actor, balance, size, free_directions != 0, form, lane);

    const bool any_attack = put_attack_offsets(board, table, unit, size, form, lane);

    bool bit = false;

    if (lane < kMoveColumn) {
        const bool kind_bits[kKinds] = {none_is_valid(), move_directions != 0, harvest_directions != 0, return_directions != 0, any_type && free_directions != 0, any_attack};
        bit = kind_bits[lane];
    } else if (lane < kHarvestColumn) {
        bit = (move_directions >> (lane - kMoveColumn)) & 1u;
    } else if (lane < kReturnColumn) {
        bit = (harvest_directions >> (lane - kHarvestColumn)) & 1u;
    } else if (lane < kProduceDirectionColumn) {
        bit = (return_directions >> (lane - kReturnColumn)) & 1u;
    } else if (lane < kProduceTypeColumn) {
        bit = ((free_directions >> (lane - kProduceDirectionColumn)) & 1u) && any_type;
    }

    form.put(0, lane < kProduceTypeColumn, bit, lane);
    form.store(lane);
}

__device__ void clear_inactive(char* begin, int32_t owned, int32_t row_bytes, const int8_t* actor_type) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(begin);
    const uintptr_t end = start + static_cast<uintptr_t>(owned) * row_bytes;
    const uint32_t row = static_cast<uint32_t>(row_bytes);

    for (uintptr_t span = (start & ~uintptr_t{15}) + 16 * uintptr_t{threadIdx.x}; span < end; span += 16 * uintptr_t{blockDim.x}) {
        const uintptr_t low = span < start ? start : span;
        const uintptr_t high = span + 16 < end ? span + 16 : end;
        const int32_t first = static_cast<int32_t>(static_cast<uint32_t>(low - start) / row);
        const int32_t last = static_cast<int32_t>(static_cast<uint32_t>(high - 1 - start) / row);
        bool holds_active_slot = false;

        for (int32_t local = first; local <= last; ++local) {
            holds_active_slot |= actor_type[local] >= 0;
        }

        if (!holds_active_slot && low == span && high == span + 16) {
            *reinterpret_cast<uint4*>(span) = make_uint4(0u, 0u, 0u, 0u);
            continue;
        }

        for (int32_t local = first; local <= last; ++local) {
            if (actor_type[local] >= 0) {
                continue;
            }

            const uintptr_t from = start + static_cast<uintptr_t>(local) * row_bytes;
            const uintptr_t to = from + row_bytes;

            for (uintptr_t byte = from < low ? low : from; byte < (to < high ? to : high); ++byte) {
                *reinterpret_cast<char*>(byte) = 0;
            }
        }
    }
}

template <typename Slot>
__device__ void list_active_slots(const Staged<Slot>& staged, const Extents& size, int lane) {
    int32_t count = 0;

    for (int32_t base = 0; base < size.owned; base += 32) {
        const int32_t local = base + lane;
        const bool active = local < size.owned && staged.actor_type[local] >= 0;
        const unsigned active_lanes = __ballot_sync(kAllLanes, active);

        if (active) {
            staged.listed[count + __popc(active_lanes & ((1u << lane) - 1u))] = static_cast<int16_t>(local);
        }

        count += __popc(active_lanes);
    }

    if (lane == 0) {
        *staged.listed_count = count;
    }
}

template <typename Slot, typename Cell>
__device__ void stage_lane(const Staged<Slot>& staged, const Board<Slot>& board, const Inputs& inputs, int64_t environment, const Extents& size) {
    for (int32_t cell = threadIdx.x; cell < size.cells; cell += blockDim.x) {
        staged.entity_at[cell] = board.entity_at[cell];
        staged.terrain[cell] = board.terrain[cell];
    }

    for (int32_t slot = threadIdx.x; slot < size.entities; slot += blockDim.x) {
        staged.unit_type[slot] = board.unit_type[slot];
        staged.owner[slot] = board.owner[slot];
    }

    for (int32_t local = threadIdx.x; local < size.owned; local += blockDim.x) {
        const OwnedUnit unit = owned_unit_at<Cell>(inputs, environment, size, local);
        staged.actor_type[local] = static_cast<int8_t>(unit.actor.unit_type);
        staged.carried[local] = unit.actor.carried;
        staged.cell[local] = unit.cell;
    }
}

template <typename RowForm, typename Slot, typename Cell, bool kStaged>
__global__ void __launch_bounds__(kWarpsPerBlock * 32) legality(Inputs inputs, typename RowForm::Element* output, Extents size) {
    extern __shared__ __align__(16) char shared[];
    const int64_t environment = blockIdx.x;
    const Board<Slot> board{size.width, size.height, size.entities, size.types, static_cast<const Slot*>(inputs.entity_at.at(environment)), inputs.terrain.at(environment), inputs.unit_type.at(environment), inputs.owner.at(environment)};
    const UnitTable table{inputs.table.at(environment), size.types};
    const int32_t* resources = inputs.resources.at(environment);
    const int lane = threadIdx.x % 32;

    if constexpr (kStaged) {
        const Staged<Slot> staged = Staged<Slot>::at(shared, size);
        stage_lane<Slot, Cell>(staged, board, inputs, environment, size);
        __syncthreads();

        if (threadIdx.x < 32) {
            list_active_slots(staged, size, lane);
        }

        __syncthreads();
        const Board<Slot> shared_board{size.width, size.height, size.entities, size.types, staged.entity_at, staged.terrain, staged.unit_type, staged.owner};

        for (int32_t entry = threadIdx.x / 32; entry < *staged.listed_count; entry += kWarpsPerBlock) {
            const int32_t local = staged.listed[entry];
            const OwnedUnit unit{Actor{staged.actor_type[local], staged.owner[size.neutral + local], staged.carried[local]}, staged.cell[local]};
            write_row(shared_board, table, unit, resources[clip(unit.actor.owner, 0, 1)], size, RowForm(output, environment * size.owned + local, size), lane);
        }

        const int32_t row_bytes = RowForm::row_bytes(size);
        clear_inactive(reinterpret_cast<char*>(output) + environment * size.owned * row_bytes, size.owned, row_bytes, staged.actor_type);
    } else {
        for (int32_t local = threadIdx.x / 32; local < size.owned; local += kWarpsPerBlock) {
            const OwnedUnit unit = owned_unit_at<Cell>(inputs, environment, size, local);
            write_row(board, table, unit, resources[clip(unit.actor.owner, 0, 1)], size, RowForm(output, environment * size.owned + local, size), lane);
        }
    }
}

ffi::ErrorOr<int64_t> stride(std::string_view target, const ffi::AnyBuffer& buffer, int64_t lanes, int64_t row_elements) {
    const int64_t extent = nanorts::lanes(buffer);

    if (extent == 0 || (extent != 1 && extent != lanes)) {
        return ffi::Error::InvalidArgument(std::string(target) + ": an operand holds one row for every lane or one shared row");
    }

    const int64_t row_bytes = static_cast<int64_t>(buffer.size_bytes()) / extent;

    if (row_bytes != row_elements * static_cast<int64_t>(ffi::ByteWidth(buffer.element_type()))) {
        return ffi::Error::InvalidArgument(std::string(target) + ": an operand has another row");
    }

    return extent == 1 ? int64_t{0} : row_bytes;
}

template <typename T>
Lanes<T> lanes_of(const ffi::AnyBuffer& buffer, int64_t stride_bytes) {
    return Lanes<T>{static_cast<const T*>(buffer.untyped_data()), stride_bytes / static_cast<int64_t>(sizeof(T))};
}

template <>
Lanes<void> lanes_of<void>(const ffi::AnyBuffer& buffer, int64_t stride_bytes) {
    return Lanes<void>{buffer.untyped_data(), stride_bytes};
}

template <typename RowForm, typename Slot, typename Cell>
void launch(cudaStream_t stream, int64_t lanes, const Inputs& inputs, typename RowForm::Element* output, const Extents& size) {
    const unsigned blocks = static_cast<unsigned>(lanes);
    const int64_t staged = Staged<Slot>::bytes(size);

    if (staged <= kSharedBytes) {
        const size_t bytes = static_cast<size_t>(staged);
        legality<RowForm, Slot, Cell, true><<<blocks, kWarpsPerBlock * 32, bytes, stream>>>(inputs, output, size);
    } else {
        legality<RowForm, Slot, Cell, false><<<blocks, kWarpsPerBlock * 32, 0, stream>>>(inputs, output, size);
    }
}

bool index_type(ffi::DataType type) {
    return type == ffi::DataType::S16 || type == ffi::DataType::S32;
}

struct Operands {
    ffi::AnyBuffer birth;
    ffi::AnyBuffer unit_type;
    ffi::AnyBuffer owner;
    ffi::AnyBuffer carried;
    ffi::AnyBuffer cell;
    ffi::AnyBuffer entity_at;
    ffi::AnyBuffer resources;
    ffi::AnyBuffer terrain;
    ffi::AnyBuffer table;
};

template <typename RowForm>
ffi::Error run(std::string_view target, std::string_view build_id, cudaStream_t stream, const Operands& buffers, ffi::AnyBuffer& result, int64_t offsets) {
    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    using ffi::DataType;

    if (buffers.birth.element_type() != DataType::S32 || buffers.unit_type.element_type() != DataType::S8 || buffers.owner.element_type() != DataType::S8 || buffers.carried.element_type() != DataType::S32 || !index_type(buffers.cell.element_type()) || !index_type(buffers.entity_at.element_type()) || buffers.resources.element_type() != DataType::S32 || buffers.terrain.element_type() != DataType::PRED || buffers.table.element_type() != DataType::S32) {
        return ffi::Error::InvalidArgument(std::string(target) + ": a buffer has another dtype");
    }

    auto grid = result.dimensions();
    auto board = buffers.entity_at.dimensions();
    auto slots = buffers.birth.dimensions();
    auto types = buffers.table.dimensions();

    if (grid.size() != 4 || grid[1] != 2 || board.size() != 3 || slots.size() != 2 || types.size() != 3) {
        return ffi::Error::InvalidArgument(std::string(target) +
                                           ": the result is `[lanes, 2, C, _]`, `entity_at` `[lanes, H, W]`, `birth` "
                                           "`[lanes, E]` and the table `[1, T, 7 + T]`");
    }

    const int64_t lanes = grid[0];
    Extents size{};
    size.height = static_cast<int32_t>(board[1]);
    size.width = static_cast<int32_t>(board[2]);
    size.cells = size.height * size.width;
    size.entities = static_cast<int32_t>(slots[1]);
    size.owned = static_cast<int32_t>(grid[1] * grid[2]);
    size.neutral = size.entities - size.owned;
    size.types = static_cast<int32_t>(types[1]);
    size.offsets = static_cast<int32_t>(offsets);
    size.columns = kProduceTypeColumn + size.types + size.offsets;
    size.side = 1;

    while ((size.side + 2) * (size.side + 2) <= size.offsets) {
        size.side += 2;
    }

    size.radius = (size.side - 1) / 2;

    if (size.neutral < 0 || size.types <= 0 || types[2] != kProduces + types[1] || size.offsets <= 0 || size.side * size.side != size.offsets || size.cells <= 0) {
        return ffi::Error::InvalidArgument(std::string(target) + ": the extents disagree");
    }

    if (lanes == 0) {
        return ffi::Error::Success();
    }

    struct Operand {
        const ffi::AnyBuffer& buffer;
        int64_t elements;
    };

    const Operand operands[] = {
        {buffers.birth, size.entities},
        {buffers.unit_type, size.entities},
        {buffers.owner, size.entities},
        {buffers.carried, size.entities},
        {buffers.cell, size.entities},
        {buffers.entity_at, size.cells},
        {buffers.resources, 2},
        {buffers.terrain, size.cells},
        {buffers.table, int64_t{size.types} * (kProduces + size.types)},
    };

    int64_t strides[9];

    for (int k = 0; k < 9; ++k) {
        auto bytes = stride(target, operands[k].buffer, lanes, operands[k].elements);

        if (bytes.has_error()) {
            return bytes.error();
        }

        strides[k] = *bytes;
    }

    const Inputs inputs{
        lanes_of<int32_t>(buffers.birth, strides[0]), lanes_of<int8_t>(buffers.unit_type, strides[1]), lanes_of<int8_t>(buffers.owner, strides[2]), lanes_of<int32_t>(buffers.carried, strides[3]), lanes_of<void>(buffers.cell, strides[4]), lanes_of<void>(buffers.entity_at, strides[5]), lanes_of<int32_t>(buffers.resources, strides[6]), lanes_of<uint8_t>(buffers.terrain, strides[7]), lanes_of<int32_t>(buffers.table, strides[8]),
    };

    auto* output = static_cast<typename RowForm::Element*>(result.untyped_data());
    const bool narrow_slot = buffers.entity_at.element_type() == DataType::S16;
    const bool narrow_cell = buffers.cell.element_type() == DataType::S16;

    if (narrow_slot && narrow_cell) {
        launch<RowForm, int16_t, int16_t>(stream, lanes, inputs, output, size);
    } else if (narrow_slot) {
        launch<RowForm, int16_t, int32_t>(stream, lanes, inputs, output, size);
    } else if (narrow_cell) {
        launch<RowForm, int32_t, int16_t>(stream, lanes, inputs, output, size);
    } else {
        launch<RowForm, int32_t, int32_t>(stream, lanes, inputs, output, size);
    }

    return nanorts::launched();
}

ffi::Error Legality(cudaStream_t stream, ffi::AnyBuffer birth, ffi::AnyBuffer unit_type, ffi::AnyBuffer owner, ffi::AnyBuffer carried, ffi::AnyBuffer cell, ffi::AnyBuffer entity_at, ffi::AnyBuffer resources, ffi::AnyBuffer terrain, ffi::AnyBuffer table, ffi::Result<ffi::AnyBuffer> rows, std::string_view build_id) {
    constexpr std::string_view kTarget = "nanorts_legality";
    auto grid = rows->dimensions();
    auto types = table.dimensions();

    if (rows->element_type() != ffi::DataType::PRED || grid.size() != 4 || types.size() != 3) {
        return ffi::Error::InvalidArgument(std::string(kTarget) + ": `rows` is `bool[lanes, 2, C, 22 + T + O]` and the table `[1, T, 7 + T]`");
    }

    return run<Rows>(kTarget, build_id, stream, Operands{birth, unit_type, owner, carried, cell, entity_at, resources, terrain, table}, *rows, grid[3] - kProduceTypeColumn - types[1]);
}

ffi::Error LegalityWords(cudaStream_t stream, ffi::AnyBuffer birth, ffi::AnyBuffer unit_type, ffi::AnyBuffer owner, ffi::AnyBuffer carried, ffi::AnyBuffer cell, ffi::AnyBuffer entity_at, ffi::AnyBuffer resources, ffi::AnyBuffer terrain, ffi::AnyBuffer table, ffi::Result<ffi::AnyBuffer> words, std::string_view build_id, int64_t offsets) {
    constexpr std::string_view kTarget = "nanorts_legality_words";
    auto grid = words->dimensions();
    auto types = table.dimensions();

    if (words->element_type() != ffi::DataType::U32 || grid.size() != 4 || grid[3] != kWords || types.size() != 3) {
        return ffi::Error::InvalidArgument(std::string(kTarget) + ": `words` is `uint32[lanes, 2, C, 3]` and the table `[1, T, 7 + T]`");
    }

    if (kProduceTypeColumn + types[1] + offsets > kWords * kWordBits) {
        return ffi::Error::InvalidArgument(std::string(kTarget) + ": the row is wider than three words");
    }

    return run<Words>(kTarget, build_id, stream, Operands{birth, unit_type, owner, carried, cell, entity_at, resources, terrain, table}, *words, offsets);
}

template <typename RowForm>
std::vector<const void*> legality_kernels() {
    return {nanorts::kernel(legality<RowForm, int16_t, int16_t, true>), nanorts::kernel(legality<RowForm, int16_t, int16_t, false>), nanorts::kernel(legality<RowForm, int16_t, int32_t, true>), nanorts::kernel(legality<RowForm, int16_t, int32_t, false>), nanorts::kernel(legality<RowForm, int32_t, int16_t, true>), nanorts::kernel(legality<RowForm, int32_t, int16_t, false>), nanorts::kernel(legality<RowForm, int32_t, int32_t, true>), nanorts::kernel(legality<RowForm, int32_t, int32_t, false>)};
}

const nanorts::Kernels nanorts_legality_kernels("nanorts_legality", legality_kernels<Rows>());
const nanorts::Kernels nanorts_legality_words_kernels("nanorts_legality_words", legality_kernels<Words>());

}

NANORTS_HANDLER(nanorts_legality, Legality,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Ret<ffi::AnyBuffer>()
                    .Attr<std::string_view>("build_id"));

NANORTS_HANDLER(nanorts_legality_words, LegalityWords,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Ret<ffi::AnyBuffer>()
                    .Attr<std::string_view>("build_id")
                    .Attr<int64_t>("offsets"));
