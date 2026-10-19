#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>

#include "nanorts_ffi.cuh"

namespace {

namespace ffi = xla::ffi;

constexpr int32_t kAbsent = -1;
constexpr int32_t kKindNone = 0;
constexpr int32_t kKindMove = 1;
constexpr int32_t kKindHarvest = 2;
constexpr int32_t kKindReturn = 3;
constexpr int32_t kKindProduce = 4;
constexpr int32_t kKindAttack = 5;
constexpr int kPlayers = 2;
constexpr int kFactors = 7;
constexpr int kDirections = 4;
__constant__ int kDx[kDirections] = {0, 1, 0, -1};
__constant__ int kDy[kDirections] = {-1, 0, 1, 0};

constexpr int kMaxWarpsPerBlock = 8;
constexpr int64_t kSharedLimit = 48 * 1024;

struct Extents {
    int height;
    int width;
    int neutral;
    int capacity;
    int types;
    int costs;
    int side;
    __host__ __device__ int cells() const {
        return height * width;
    }

    __host__ __device__ int entities() const {
        return neutral + kPlayers * capacity;
    }

    __host__ __device__ int domain() const {
        return cells() + 2 * width + 1;
    }
};

__host__ __device__ inline int words(int bits) {
    return (bits + 31) / 32;
}

__host__ __device__ inline int64_t warp_words(const Extents& size) {
    return 2 * words(size.domain()) + words(size.cells()) + 2 * words(size.capacity) + 2 * static_cast<int64_t>(size.capacity);
}

template <typename Cell>
struct Operands {
    const int32_t* actions;
    const int32_t* birth;
    const int8_t* pending_kind;
    const Cell* cell;
    const int8_t* pending_direction;
    const int8_t* pending_produced_type;
    const int32_t* resources;
    const bool* move_direction;
    const bool* produce_direction;
    const bool* produce_type;
    const int32_t* unit_cost;
    int64_t actions_stride, birth_stride, pending_kind_stride, cell_stride, pending_direction_stride, pending_produced_type_stride, resources_stride, move_direction_stride, produce_direction_stride, produce_type_stride, unit_cost_stride;
};

struct Results {
    int32_t* field[6];
};

__device__ inline bool bit(const uint32_t* set, int index) {
    return (set[index >> 5] >> (index & 31)) & 1u;
}

__device__ inline void set_bit(uint32_t* set, int index) {
    atomicOr(&set[index >> 5], 1u << (index & 31));
}

__device__ inline int below(const uint32_t* set, int index) {
    int count = 0;

    for (int w = 0; w < (index >> 5); ++w) {
        count += __popc(set[w]);
    }

    return count + __popc(set[index >> 5] & ((1u << (index & 31)) - 1u));
}

__device__ inline int warp_sum(int value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }

    return value;
}

__device__ inline int delta(int direction, int width) {
    if (direction < 0 || direction >= kDirections) {
        return 0;
    }

    return kDx[direction] + kDy[direction] * width;
}

__device__ inline bool claims(int kind) {
    return kind == kKindMove || kind == kKindProduce;
}

__device__ inline int clamp(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

struct DecodedRow {
    int kind, direction, none_duration, target_cell, produced_type;
    int position;
    int cost;
};

__device__ DecodedRow decode(const int32_t* actions, const int32_t* unit_cost, int cell, const Extents& size) {
    const int32_t* factors = actions + static_cast<int64_t>(cell) * kFactors;
    DecodedRow decoded;
    decoded.kind = factors[0];
    decoded.direction = decoded.kind == kKindMove ? factors[1] : decoded.kind == kKindHarvest ? factors[2] : decoded.kind == kKindReturn ? factors[3] : decoded.kind == kKindProduce ? factors[4] : kAbsent;
    decoded.produced_type = decoded.kind == kKindProduce ? factors[5] : kAbsent;
    const int radius = (size.side - 1) / 2;
    const int offset = factors[6];
    const int x = cell % size.width + (offset % size.side - radius);
    const int y = cell / size.width + (offset / size.side - radius);
    const bool on_board = x >= 0 && x < size.width && y >= 0 && y < size.height;
    decoded.target_cell = decoded.kind == kKindAttack && on_board ? y * size.width + x : kAbsent;
    decoded.none_duration = decoded.kind == kKindNone ? -1 : 0;
    decoded.position = cell + delta(decoded.direction, size.width);
    decoded.cost = decoded.kind == kKindProduce ? unit_cost[clamp(factors[5], 0, size.costs - 1)] : 0;
    return decoded;
}

__device__ inline int filler_rank(const uint32_t* fillers, int capacity, const int32_t* birth, int start, int slot, int kept) {
    int rank = kept;

    for (int word = 0; word < words(capacity); ++word) {
        uint32_t set = fillers[word];

        while (set != 0u) {
            const int other = word * 32 + __ffs(set) - 1;
            set &= set - 1u;
            rank += birth[start + other] < birth[slot];
        }
    }

    return rank;
}

__device__ inline void reset_player_sets(uint32_t* source_cells, uint32_t* survived, uint32_t* fillers, uint32_t* used, const uint32_t* pending_claims, int cells, int capacity, int domain, int lane) {
    for (int word = lane; word < words(cells); word += 32) {
        source_cells[word] = 0u;
    }

    for (int word = lane; word < words(capacity); word += 32) {
        survived[word] = 0u;
        fillers[word] = 0u;
    }

    for (int word = lane; word < words(domain); word += 32) {
        used[word] = pending_claims[word];
    }

    __syncwarp();
}

template <typename Cell>
__device__ int list_source_cells(const int32_t* birth, const int8_t* pending_kind, const Cell* cell, int start, int capacity, uint32_t* source_cells, int lane) {
    int available = 0;

    for (int local = lane; local < capacity; local += 32) {
        const int slot = start + local;

        if (birth[slot] != kAbsent && pending_kind[slot] == kAbsent) {
            set_bit(source_cells, cell[slot]);
            ++available;
        }
    }

    return warp_sum(available);
}

template <typename Cell>
__device__ void mark_fillers(const int32_t* birth, const int8_t* pending_kind, const Cell* cell, int start, int capacity, const uint32_t* source_cells, const uint32_t* survived, uint32_t* fillers, int lane) {
    for (int local = lane; local < capacity; local += 32) {
        const int slot = start + local;

        if (birth[slot] == kAbsent || pending_kind[slot] != kAbsent) {
            continue;
        }

        if (!bit(survived, below(source_cells, cell[slot]))) {
            set_bit(fillers, local);
        }
    }

    __syncwarp();
}

template <typename Cell>
__global__ void decode_lanes(Operands<Cell> operands, Results results, Extents size, int64_t lanes) {
    extern __shared__ uint32_t shared[];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t environment = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + warp;

    if (environment >= lanes) {
        return;
    }

    const int cells = size.cells(), domain = size.domain(), capacity = size.capacity;
    uint32_t* pending_claims = shared + warp * warp_words(size);
    uint32_t* used = pending_claims + words(domain);
    uint32_t* source_cells = used + words(domain);
    uint32_t* survived = source_cells + words(cells);
    uint32_t* fillers = survived + words(capacity);
    int32_t* walk_claim = reinterpret_cast<int32_t*>(fillers + words(capacity));
    int32_t* walk_cost = walk_claim + capacity;

    const int32_t* actions = operands.actions + environment * operands.actions_stride;
    const int32_t* birth = operands.birth + environment * operands.birth_stride;
    const int8_t* pending_kind = operands.pending_kind + environment * operands.pending_kind_stride;
    const Cell* cell = operands.cell + environment * operands.cell_stride;
    const int8_t* pending_direction = operands.pending_direction + environment * operands.pending_direction_stride;
    const int8_t* pending_produced_type = operands.pending_produced_type + environment * operands.pending_produced_type_stride;
    const int32_t* balances = operands.resources + environment * operands.resources_stride;
    const bool* move_direction = operands.move_direction + environment * operands.move_direction_stride;
    const bool* produce_direction = operands.produce_direction + environment * operands.produce_direction_stride;
    const bool* produce_type = operands.produce_type + environment * operands.produce_type_stride;
    const int32_t* unit_cost = operands.unit_cost + environment * operands.unit_cost_stride;

    for (int word = lane; word < words(domain); word += 32) {
        pending_claims[word] = 0u;
    }

    __syncwarp();
    int spend[kPlayers] = {0, 0};

    for (int slot = lane; slot < size.entities(); slot += 32) {
        if (birth[slot] == kAbsent) {
            continue;
        }

        const int kind = pending_kind[slot];

        if (claims(kind)) {
            const int index = static_cast<int>(cell[slot]) + delta(pending_direction[slot], size.width) + size.width;

            if (index >= 0 && index < domain - 1) {
                set_bit(pending_claims, index);
            }
        }

        if (kind == kKindProduce && slot >= size.neutral) {
            spend[(slot - size.neutral) / capacity] += unit_cost[clamp(pending_produced_type[slot], 0, size.costs - 1)];
        }
    }

    int spent[kPlayers] = {warp_sum(spend[0]), warp_sum(spend[1])};
    __syncwarp();

    for (int player = 0; player < kPlayers; ++player) {
        const int start = size.neutral + player * capacity;
        reset_player_sets(source_cells, survived, fillers, used, pending_claims, cells, capacity, domain, lane);
        const int count = list_source_cells(birth, pending_kind, cell, start, capacity, source_cells, lane);
        __syncwarp();

        for (int local = lane; local < capacity; local += 32) {
            const int slot = start + local;

            if (birth[slot] == kAbsent || pending_kind[slot] != kAbsent) {
                continue;
            }

            const int unit_cell = cell[slot];
            const int source_rank = below(source_cells, unit_cell);
            const DecodedRow decoded = decode(actions + static_cast<int64_t>(player) * cells * kFactors, unit_cost, unit_cell, size);
            walk_claim[source_rank] = claims(decoded.kind) ? decoded.position + size.width : kAbsent;
            walk_cost[source_rank] = decoded.cost;
        }

        __syncwarp();

        if (lane == 0) {
            int spent_so_far[kPlayers] = {spent[0], spent[1]};

            for (int source_rank = 0; source_rank < count; ++source_rank) {
                const int index = walk_claim[source_rank];
                const bool claim = index != kAbsent;
                int spent_with_row[kPlayers] = {spent_so_far[0], spent_so_far[1]};
                spent_with_row[player] += walk_cost[source_rank];
                bool over_budget = false;

                for (int other_player = 0; other_player < kPlayers; ++other_player) {
                    over_budget |= spent_so_far[other_player] != 0 && spent_with_row[other_player] > 0 && spent_with_row[other_player] > balances[other_player];
                }

                const bool consistent = !(claim && bit(used, clamp(index, 0, domain - 1))) && !over_budget;

                if (!consistent) {
                    continue;
                }

                if (claim && index >= 0 && index < domain) {
                    set_bit(used, index);
                }

                spent_so_far[0] = spent_with_row[0];
                spent_so_far[1] = spent_with_row[1];
                survived[source_rank >> 5] |= 1u << (source_rank & 31);
            }
        }

        __syncwarp();

        mark_fillers(birth, pending_kind, cell, start, capacity, source_cells, survived, fillers, lane);
        int kept = 0;

        for (int word = 0; word < words(capacity); ++word) {
            kept += __popc(survived[word]);
        }

        int issued = 0;

        for (int local = lane; local < capacity; local += 32) {
            const int slot = start + local;
            const int64_t output_index = environment * kPlayers * capacity + static_cast<int64_t>(player) * capacity + local;
            int command[6] = {kAbsent, kAbsent, 0, kAbsent, kAbsent, kAbsent};

            if (birth[slot] != kAbsent && pending_kind[slot] == kAbsent) {
                const int unit_cell = cell[slot];
                const int source_rank = below(source_cells, unit_cell);

                if (bit(survived, source_rank)) {
                    const DecodedRow decoded = decode(actions + static_cast<int64_t>(player) * cells * kFactors, unit_cost, unit_cell, size);
                    command[0] = decoded.kind;
                    command[1] = decoded.direction;
                    command[2] = decoded.none_duration;
                    command[3] = decoded.target_cell;
                    command[4] = decoded.produced_type;
                    command[5] = below(survived, source_rank);

                    if (player + 1 < kPlayers && claims(decoded.kind)) {
                        const int known_direction = clamp(decoded.direction, 0, kDirections - 1);
                        const bool legal = decoded.kind == kKindMove ? move_direction[local * kDirections + known_direction] : produce_direction[local * kDirections + known_direction] && produce_type[local * size.types + clamp(decoded.produced_type, 0, size.types - 1)];

                        if (legal) {
                            const int index = decoded.position + size.width;

                            if (index >= 0 && index < domain - 1) {
                                set_bit(pending_claims, index);
                            }

                            if (decoded.kind == kKindProduce) {
                                issued += decoded.cost;
                            }
                        }
                    }
                } else {
                    command[0] = kKindNone;
                    command[2] = 1;
                    command[5] = filler_rank(fillers, capacity, birth, start, slot, kept);
                }
            }

            for (int column = 0; column < 6; ++column) {
                results.field[column][output_index] = command[column];
            }
        }

        if (player + 1 < kPlayers) {
            spent[player] += warp_sum(issued);
        }

        __syncwarp();
    }
}

int64_t stride(const ffi::AnyBuffer& buffer, int64_t lanes, int64_t row) {
    const int64_t extent = nanorts::lanes(buffer);
    const int64_t count = static_cast<int64_t>(buffer.element_count());

    if (extent == 1 && count == row) {
        return 0;
    }

    if (extent == lanes && count == lanes * row) {
        return row;
    }

    return -1;
}

template <typename Cell>
ffi::Error launch(cudaStream_t stream, Operands<Cell> operands, Results results, const Extents& size, int64_t lanes) {
    const int64_t per_warp = warp_words(size) * 4;
    const int warps = static_cast<int>(std::min<int64_t>(kMaxWarpsPerBlock, std::max<int64_t>(1, kSharedLimit / per_warp)));
    const int64_t blocks = (lanes + warps - 1) / warps;
    decode_lanes<Cell><<<static_cast<unsigned>(blocks), warps * 32, static_cast<size_t>(warps * per_warp), stream>>>(operands, results, size, lanes);
    return nanorts::launched();
}

ffi::Error GridnetDecode(cudaStream_t stream, ffi::AnyBuffer actions, ffi::AnyBuffer birth, ffi::AnyBuffer pending_kind, ffi::AnyBuffer cell, ffi::AnyBuffer pending_direction, ffi::AnyBuffer pending_produced_type, ffi::AnyBuffer player_resources, ffi::AnyBuffer move_direction, ffi::AnyBuffer produce_direction, ffi::AnyBuffer produce_type, ffi::AnyBuffer unit_cost, ffi::Result<ffi::AnyBuffer> kind, ffi::Result<ffi::AnyBuffer> direction, ffi::Result<ffi::AnyBuffer> none_duration, ffi::Result<ffi::AnyBuffer> target_cell, ffi::Result<ffi::AnyBuffer> produced_type, ffi::Result<ffi::AnyBuffer> request_order, std::string_view build_id, int64_t height, int64_t width, int64_t neutral, int64_t capacity, int64_t types, int64_t costs, int64_t side, int64_t shared_words) {
    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    const Extents e{static_cast<int>(height), static_cast<int>(width), static_cast<int>(neutral), static_cast<int>(capacity), static_cast<int>(types), static_cast<int>(costs), static_cast<int>(side)};

    if (height <= 0 || width <= 0 || neutral < 0 || capacity <= 0 || types <= 0 || costs <= 0 || side <= 0 || warp_words(e) * 4 > kSharedLimit) {
        return ffi::Error::InvalidArgument("nanorts_gridnet_decode: a Bucket outside the kernel's limits");
    }

    if (shared_words != warp_words(e)) {
        return ffi::Error::InvalidArgument("nanorts_gridnet_decode: the caller sized another shared-memory layout");
    }

    ffi::AnyBuffer* fields[6] = {&*kind, &*direction, &*none_duration, &*target_cell, &*produced_type, &*request_order};
    const int64_t lanes = nanorts::lanes(*fields[0]);
    Results out;

    for (int f = 0; f < 6; ++f) {
        if (fields[f]->element_type() != ffi::DataType::S32 || nanorts::lanes(*fields[f]) != lanes || static_cast<int64_t>(fields[f]->element_count()) != lanes * kPlayers * capacity) {
            return ffi::Error::InvalidArgument("nanorts_gridnet_decode: every result is int32[lanes, 2, capacity]");
        }

        out.field[f] = static_cast<int32_t*>(fields[f]->untyped_data());
    }

    if (lanes == 0) {
        return ffi::Error::Success();
    }

    struct Expected {
        const ffi::AnyBuffer* buffer;
        ffi::DataType type;
        int64_t row;
        int64_t* stride;
        const char* name;
    };

    int64_t strides[11];
    const int64_t entities = e.entities();
    const Expected expected[11] = {
        {&actions, ffi::DataType::S32, kPlayers * int64_t{e.cells()} * kFactors, &strides[0], "actions"},
        {&birth, ffi::DataType::S32, entities, &strides[1], "birth"},
        {&pending_kind, ffi::DataType::S8, entities, &strides[2], "pending_kind"},
        {&cell, cell.element_type(), entities, &strides[3], "cell"},
        {&pending_direction, ffi::DataType::S8, entities, &strides[4], "pending_direction"},
        {&pending_produced_type, ffi::DataType::S8, entities, &strides[5], "pending_produced_type"},
        {&player_resources, ffi::DataType::S32, kPlayers, &strides[6], "player_resources"},
        {&move_direction, ffi::DataType::PRED, capacity * kDirections, &strides[7], "move_direction"},
        {&produce_direction, ffi::DataType::PRED, capacity * kDirections, &strides[8], "produce_direction"},
        {&produce_type, ffi::DataType::PRED, capacity * types, &strides[9], "produce_type"},
        {&unit_cost, ffi::DataType::S32, costs, &strides[10], "unit_cost"},
    };

    for (const Expected& operand : expected) {
        *operand.stride = stride(*operand.buffer, lanes, operand.row);

        if (operand.buffer->element_type() != operand.type || *operand.stride < 0) {
            return ffi::Error::InvalidArgument(std::string("nanorts_gridnet_decode: operand ") + operand.name + " has another type or shape");
        }
    }

    auto fill = [&](auto typed_cell) {
        using Cell = std::remove_const_t<std::remove_pointer_t<decltype(typed_cell)>>;
        Operands<Cell> in;
        in.actions = static_cast<const int32_t*>(actions.untyped_data());
        in.birth = static_cast<const int32_t*>(birth.untyped_data());
        in.pending_kind = static_cast<const int8_t*>(pending_kind.untyped_data());
        in.cell = typed_cell;
        in.pending_direction = static_cast<const int8_t*>(pending_direction.untyped_data());
        in.pending_produced_type = static_cast<const int8_t*>(pending_produced_type.untyped_data());
        in.resources = static_cast<const int32_t*>(player_resources.untyped_data());
        in.move_direction = static_cast<const bool*>(move_direction.untyped_data());
        in.produce_direction = static_cast<const bool*>(produce_direction.untyped_data());
        in.produce_type = static_cast<const bool*>(produce_type.untyped_data());
        in.unit_cost = static_cast<const int32_t*>(unit_cost.untyped_data());
        in.actions_stride = strides[0];
        in.birth_stride = strides[1];
        in.pending_kind_stride = strides[2];
        in.cell_stride = strides[3];
        in.pending_direction_stride = strides[4];
        in.pending_produced_type_stride = strides[5];
        in.resources_stride = strides[6];
        in.move_direction_stride = strides[7];
        in.produce_direction_stride = strides[8];
        in.produce_type_stride = strides[9];
        in.unit_cost_stride = strides[10];
        return launch(stream, in, out, e, lanes);
    };

    switch (cell.element_type()) {
        case ffi::DataType::S16:
            return fill(static_cast<const int16_t*>(cell.untyped_data()));
        case ffi::DataType::S32:
            return fill(static_cast<const int32_t*>(cell.untyped_data()));
        default:
            return ffi::Error::InvalidArgument("nanorts_gridnet_decode: cell is int16 or int32");
    }
}

const nanorts::Kernels nanorts_gridnet_decode_kernels("nanorts_gridnet_decode", {nanorts::kernel(decode_lanes<int16_t>), nanorts::kernel(decode_lanes<int32_t>)});

}

NANORTS_HANDLER(nanorts_gridnet_decode, GridnetDecode,
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
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Ret<ffi::AnyBuffer>()
                    .Ret<ffi::AnyBuffer>()
                    .Ret<ffi::AnyBuffer>()
                    .Ret<ffi::AnyBuffer>()
                    .Ret<ffi::AnyBuffer>()
                    .Ret<ffi::AnyBuffer>()
                    .Attr<std::string_view>("build_id")
                    .Attr<int64_t>("height")
                    .Attr<int64_t>("width")
                    .Attr<int64_t>("neutral")
                    .Attr<int64_t>("capacity")
                    .Attr<int64_t>("types")
                    .Attr<int64_t>("costs")
                    .Attr<int64_t>("side")
                    .Attr<int64_t>("warp_words"));
