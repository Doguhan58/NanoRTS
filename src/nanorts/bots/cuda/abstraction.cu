#include <cstdint>
#include <initializer_list>

#include "astar.cuh"
#include "nanorts_ffi.cuh"

namespace {

namespace ffi = xla::ffi;

constexpr int kThreads = 32;
constexpr int64_t kSharedLimit = 48 * 1024;
constexpr int64_t kChainBytesPerCell = nanorts::astar::kScratchBytesPerCell + 1;
constexpr int64_t kStart = 0;
constexpr int64_t kGoal = 1;
constexpr int64_t kRadius = 2;
constexpr int64_t kDraws = 3;
constexpr int64_t kOptions = 4;
constexpr int64_t kSearchColumns = 5;
constexpr int kDirections = 4;
constexpr int32_t kKindAbsent = -1;
constexpr uint8_t kOpen = 1;
constexpr uint8_t kTaken = 2;

constexpr int64_t kRank = 0;
constexpr int64_t kSelector = 1;
constexpr int64_t kDirection = 2;
constexpr int64_t kDrew = 3;
constexpr int64_t kDecisionFields = 4;
constexpr int64_t kKeptCount = 0;
constexpr int64_t kCalls = 1;

constexpr int32_t kFellBack = 0;
constexpr int32_t kBlocked = 1;
constexpr int32_t kMoved = 2;

template <typename T>
struct Rows {
    const T* data;
    int64_t stride;
};

struct Columns {
    int64_t kind;
    int64_t claim;
    int64_t cost;
};

struct Pass {
    int64_t capacity;
    int64_t columns;
    Columns fallback;
    Columns blocked;
    int64_t draw_count;
    bool drawing;
};

struct Desire {
    bool eligible;
    int32_t claim;
    int32_t cost;
};

__device__ int32_t neighbour(int32_t cell, int direction, int width) {
    int32_t offset = -1;

    if (direction == nanorts::astar::kUp) {
        offset = -width;
    } else if (direction == nanorts::astar::kRight) {
        offset = 1;
    } else if (direction == nanorts::astar::kDown) {
        offset = width;
    }

    return static_cast<int32_t>(static_cast<uint32_t>(cell) + static_cast<uint32_t>(offset));
}

__device__ int drawn_direction(int32_t first, int32_t options) {
    for (int turn = 0; turn < kDirections; ++turn) {
        const int direction = static_cast<int>((static_cast<uint32_t>(first) + static_cast<uint32_t>(turn)) & 3u);

        if (((options >> direction) & 1) == 1) {
            return direction;
        }
    }

    return nanorts::astar::kAbsent;
}

__device__ Desire listed(const int32_t* entry, const Columns& columns) {
    return Desire{entry[columns.kind] != kKindAbsent, entry[columns.claim], entry[columns.cost]};
}

__device__ Desire moved(int32_t start, int32_t direction, int width) {
    return Desire{true, direction == nanorts::astar::kAbsent ? nanorts::astar::kAbsent : neighbour(start, direction, width), 0};
}

__global__ void __launch_bounds__(kThreads) chain_kernel(Rows<bool> free, Rows<bool> claimed, Rows<int32_t> rows, Rows<int32_t> queue, Rows<int32_t> count, Rows<bool> moves_allowed, Rows<int32_t> reserved, Rows<int32_t> balance, Rows<int32_t> draws, Pass pass, int width, int height, int32_t* out) {
    extern __shared__ uint64_t shared[];
    const int64_t lane = blockIdx.x;
    const int cells = width * height;
    nanorts::astar::Scratch scratch = nanorts::astar::Scratch::carve(shared, cells);
    uint8_t* flags = scratch.seen + cells;
    const bool* field = free.data + lane * free.stride;
    const bool* pending = claimed.data + lane * claimed.stride;

    for (int cell = threadIdx.x; cell < cells; cell += blockDim.x) {
        flags[cell] = (field[cell] ? kOpen : 0) | (pending[cell] ? kTaken : 0);
    }

    const int32_t* entries = rows.data + lane * rows.stride;
    const int32_t* order = queue.data + lane * queue.stride;
    const int32_t* drawn = draws.data + lane * draws.stride;
    const bool moves = moves_allowed.data[lane * moves_allowed.stride];
    const uint32_t reserve = static_cast<uint32_t>(reserved.data[lane * reserved.stride]);
    const int32_t held = balance.data[lane * balance.stride];
    int32_t* decisions = out + lane * (pass.capacity + 1) * kDecisionFields;
    const int32_t asked = count.data[lane * count.stride];
    const int64_t searched = asked < 0 ? 0 : (asked > pass.capacity ? pass.capacity : static_cast<int64_t>(asked));

    for (int64_t index = threadIdx.x; index < pass.capacity * kDecisionFields; index += blockDim.x) {
        const int64_t at = index % kDecisionFields;
        decisions[index] = at == kRank || at == kDirection ? nanorts::astar::kAbsent : 0;
    }

    uint32_t kept_cost = 0;
    int32_t kept_count = 0;
    int64_t calls = 0;

    for (int64_t row = 0; row < searched; ++row) {
        __syncthreads();

        for (int cell = threadIdx.x; cell < cells; cell += blockDim.x) {
            scratch.clear(cell, (flags[cell] & kOpen) != 0);
        }

        __syncthreads();

        if (threadIdx.x != 0) {
            continue;
        }

        const int32_t slot = order[row];

        if (slot < 0 || slot >= pass.capacity) {
            continue;
        }

        const int32_t* entry = entries + slot * pass.columns;
        const int32_t start = entry[kStart];
        const nanorts::astar::Path path = nanorts::astar::search(scratch, start, entry[kGoal], entry[kRadius], width, height);
        const bool stepped = path.step != nanorts::astar::kAbsent;
        int32_t selector = kFellBack;
        int32_t direction = nanorts::astar::kAbsent;
        Desire desire = listed(entry, pass.fallback);

        if (stepped && moves) {
            selector = kMoved;
            direction = path.step;
            desire = moved(start, direction, width);
        } else if (stepped) {
            selector = kBlocked;
            desire = listed(entry, pass.blocked);
        }

        const bool drew = pass.drawing && entry[kDraws] != 0 && !(stepped && moves);

        if (drew) {
            const int64_t at = calls < pass.draw_count ? calls : pass.draw_count - 1;
            selector = kMoved;
            direction = drawn_direction(drawn[at], entry[kOptions]);
            desire = moved(start, direction, width);
            ++calls;
        }

        const bool on_board = desire.claim >= 0 && desire.claim < cells;
        const bool conflicting = on_board && (flags[desire.claim] & kTaken) != 0;
        const bool overdrawn = desire.cost > 0 && static_cast<int32_t>(reserve + kept_cost + static_cast<uint32_t>(desire.cost)) > held;
        const bool kept = desire.eligible && !conflicting && !overdrawn;

        if (on_board) {
            flags[desire.claim] = (flags[desire.claim] & ~kOpen) | (kept ? kTaken : 0);
        }

        int32_t* decision = decisions + slot * kDecisionFields;
        decision[kRank] = kept ? kept_count : nanorts::astar::kAbsent;
        decision[kSelector] = selector;
        decision[kDirection] = direction;
        decision[kDrew] = drew ? 1 : 0;

        if (kept) {
            kept_cost += static_cast<uint32_t>(desire.cost);
            ++kept_count;
        }
    }

    if (threadIdx.x == 0) {
        int32_t* totals = decisions + pass.capacity * kDecisionFields;

        for (int64_t at = 0; at < kDecisionFields; ++at) {
            totals[at] = 0;
        }

        totals[kKeptCount] = kept_count;
        totals[kCalls] = static_cast<int32_t>(calls);
    }
}

template <typename T>
Rows<T> rows_of(const ffi::AnyBuffer& leaf, int64_t row) {
    return Rows<T>{static_cast<const T*>(leaf.untyped_data()), nanorts::lanes(leaf) == 1 ? 0 : row};
}

bool listed_in(const Columns& columns, int64_t width) {
    for (const int64_t column : {columns.kind, columns.claim, columns.cost}) {
        if (column < kSearchColumns || column >= width) {
            return false;
        }
    }

    return true;
}

ffi::Error Chain(cudaStream_t stream, ffi::AnyBuffer free, ffi::AnyBuffer claimed, ffi::AnyBuffer rows, ffi::AnyBuffer queue, ffi::AnyBuffer count, ffi::AnyBuffer moves_allowed, ffi::AnyBuffer reserved, ffi::AnyBuffer balance, ffi::AnyBuffer draws, ffi::Result<ffi::AnyBuffer> out, std::string_view build_id, int64_t width, int64_t height, int64_t drawing, int64_t fallback_kind, int64_t fallback_claim, int64_t fallback_cost, int64_t blocked_kind, int64_t blocked_claim, int64_t blocked_cost) {
    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (free.element_type() != ffi::PRED || claimed.element_type() != ffi::PRED || rows.element_type() != ffi::S32 || queue.element_type() != ffi::S32 || count.element_type() != ffi::S32 || moves_allowed.element_type() != ffi::PRED || reserved.element_type() != ffi::S32 || balance.element_type() != ffi::S32 || draws.element_type() != ffi::S32 || out->element_type() != ffi::S32) {
        return ffi::Error::InvalidArgument(
            "nanorts_astar_chain takes bool free and claimed cells and moves, int32 rows, "
            "queue, count, reserved, balance and draws, and returns int32");
    }

    const auto decided = out->dimensions();
    const auto entries = rows.dimensions();
    const int64_t cells = width * height;
    const int64_t shared = cells * kChainBytesPerCell;

    if (width <= 0 || height <= 0 || shared > kSharedLimit || decided.size() != 3 || decided[1] < 1 || decided[2] != kDecisionFields || entries.size() != 3 || entries[1] != decided[1] - 1) {
        return ffi::Error::InvalidArgument(
            "the result is one four-field decision per slot and one totals row per lane, "
            "over a field whose chain fits one block's shared memory");
    }

    const int64_t lanes = decided[0];
    const Pass pass{decided[1] - 1, entries[2], Columns{fallback_kind, fallback_claim, fallback_cost}, Columns{blocked_kind, blocked_claim, blocked_cost}, nanorts::lanes(draws) == 0 ? 0 : static_cast<int64_t>(draws.element_count()) / nanorts::lanes(draws), drawing != 0};

    if (pass.columns < kSearchColumns || !listed_in(pass.fallback, pass.columns) || !listed_in(pass.blocked, pass.columns) || (pass.drawing && pass.draw_count == 0)) {
        return ffi::Error::InvalidArgument(
            "each row holds the search columns and both desires' kind, claim and cost, "
            "and a drawing pass has draws");
    }

    if (lanes == 0) {
        return ffi::Error::Success();
    }

    const auto shared_or_per_lane = [lanes](const ffi::AnyBuffer& leaf, int64_t row) {
        const int64_t extent = nanorts::lanes(leaf);
        return (extent == 1 || extent == lanes) && static_cast<int64_t>(leaf.element_count()) == extent * row;
    };

    if (!shared_or_per_lane(free, cells) || !shared_or_per_lane(claimed, cells) || !shared_or_per_lane(rows, pass.capacity * pass.columns) || !shared_or_per_lane(queue, pass.capacity) || !shared_or_per_lane(count, 1) || !shared_or_per_lane(moves_allowed, 1) || !shared_or_per_lane(reserved, 1) || !shared_or_per_lane(balance, 1) || !shared_or_per_lane(draws, pass.draw_count)) {
        return ffi::Error::InvalidArgument(
            "each operand holds one row per lane or one shared row: the free and claimed "
            "cells, the pass's rows and queue, a scalar count, moves flag, reserve and "
            "balance, and the draws");
    }

    chain_kernel<<<static_cast<unsigned>(lanes), kThreads, static_cast<size_t>(shared), stream>>>(rows_of<bool>(free, cells), rows_of<bool>(claimed, cells), rows_of<int32_t>(rows, pass.capacity * pass.columns), rows_of<int32_t>(queue, pass.capacity), rows_of<int32_t>(count, 1), rows_of<bool>(moves_allowed, 1), rows_of<int32_t>(reserved, 1), rows_of<int32_t>(balance, 1), rows_of<int32_t>(draws, pass.draw_count), pass, static_cast<int>(width), static_cast<int>(height), static_cast<int32_t*>(out->untyped_data()));
    return nanorts::launched();
}

const nanorts::Kernels nanorts_astar_chain_kernels("nanorts_astar_chain", {nanorts::kernel(chain_kernel)});

}

NANORTS_HANDLER(nanorts_astar_chain, Chain,
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
                    .Attr<int64_t>("width")
                    .Attr<int64_t>("height")
                    .Attr<int64_t>("drawing")
                    .Attr<int64_t>("fallback_kind")
                    .Attr<int64_t>("fallback_claim")
                    .Attr<int64_t>("fallback_cost")
                    .Attr<int64_t>("blocked_kind")
                    .Attr<int64_t>("blocked_claim")
                    .Attr<int64_t>("blocked_cost"));
