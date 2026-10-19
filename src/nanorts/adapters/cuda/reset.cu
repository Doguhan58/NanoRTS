#include <cstdint>

#include "nanorts_ffi.cuh"

namespace {

namespace ffi = xla::ffi;

constexpr int64_t kMaxLeaves = 32;
constexpr int kWarpsPerBlock = 8;

struct Leaf {
    const char* reset;
    const char* next;
    char* out;
    int64_t row_bytes;
    int64_t reset_stride;
};

struct Leaves {
    Leaf leaf[kMaxLeaves];
    int64_t count;
    bool in_place;
};

template <typename Unit>
__device__ void copy_row(const char* source, char* destination, int64_t bytes, int lane) {
    const Unit* from = reinterpret_cast<const Unit*>(source);

    Unit* to = reinterpret_cast<Unit*>(destination);

    const int64_t units = bytes / static_cast<int64_t>(sizeof(Unit));

    for (int64_t unit_index = lane; unit_index < units; unit_index += 32) {
        to[unit_index] = from[unit_index];
    }
}

__global__ void reset_where_done(const uint8_t* done, int64_t done_stride, Leaves leaves, int64_t lanes) {
    const int64_t environment = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;

    if (environment >= lanes) {
        return;
    }

    const bool finished = done[environment * done_stride] != 0;

    if (!finished && leaves.in_place) {
        return;
    }

    for (int64_t leaf_index = 0; leaf_index < leaves.count; ++leaf_index) {
        const Leaf& leaf = leaves.leaf[leaf_index];

        const char* source = finished ? leaf.reset + environment * leaf.reset_stride : leaf.next + environment * leaf.row_bytes;
        char* destination = leaf.out + environment * leaf.row_bytes;

        if (source == destination) {
            continue;
        }

        const uintptr_t alignment = reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(destination) | static_cast<uintptr_t>(leaf.row_bytes);

        if (alignment % 16 == 0) {
            copy_row<uint4>(source, destination, leaf.row_bytes, lane);
        } else if (alignment % 8 == 0) {
            copy_row<uint2>(source, destination, leaf.row_bytes, lane);
        } else if (alignment % 4 == 0) {
            copy_row<uint32_t>(source, destination, leaf.row_bytes, lane);
        } else if (alignment % 2 == 0) {
            copy_row<uint16_t>(source, destination, leaf.row_bytes, lane);
        } else {
            copy_row<uint8_t>(source, destination, leaf.row_bytes, lane);
        }
    }
}

ffi::Error ResetWhereDone(cudaStream_t stream, ffi::Buffer<ffi::PRED> done, ffi::RemainingArgs operands, ffi::RemainingRets results, std::string_view build_id, int64_t count) {
    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (count < 0 || count > kMaxLeaves || results.size() != static_cast<size_t>(count) || operands.size() < static_cast<size_t>(2 * count)) {
        return ffi::Error::InvalidArgument("nanorts_reset_where_done takes `leaves` reset, next and result buffers");
    }

    Leaves leaves{};
    leaves.count = count;
    leaves.in_place = true;
    int64_t lanes = -1;

    for (int64_t k = 0; k < count; ++k) {
        auto reset = operands.get<ffi::AnyBuffer>(k);
        auto next = operands.get<ffi::AnyBuffer>(count + k);
        auto out = results.get<ffi::AnyBuffer>(k);

        if (!reset.has_value()) {
            return reset.error();
        }

        if (!next.has_value()) {
            return next.error();
        }

        if (!out.has_value()) {
            return out.error();
        }

        ffi::AnyBuffer result = **out;
        const int64_t extent = nanorts::lanes(result);

        if (k == 0) {
            lanes = extent;
        }

        if (extent != lanes || nanorts::lanes(*next) != lanes) {
            return ffi::Error::InvalidArgument("every leaf has the same lane count");
        }

        if (next->size_bytes() != result.size_bytes() || next->element_type() != result.element_type()) {
            return ffi::Error::InvalidArgument("each result has its next leaf's type");
        }

        leaves.in_place &= next->untyped_data() == result.untyped_data();

        if (reset->element_type() != result.element_type()) {
            return ffi::Error::InvalidArgument("a reset leaf has its result's dtype");
        }

        const int64_t row_bytes = lanes == 0 ? 0 : static_cast<int64_t>(result.size_bytes()) / lanes;
        const int64_t shared = nanorts::lanes(*reset);

        if ((shared != 1 && shared != lanes) || static_cast<int64_t>(reset->size_bytes()) != shared * row_bytes) {
            return ffi::Error::InvalidArgument("a reset leaf holds one row for every lane or one shared row");
        }

        leaves.leaf[k] = Leaf{static_cast<const char*>(reset->untyped_data()), static_cast<const char*>(next->untyped_data()), static_cast<char*>(result.untyped_data()), row_bytes, shared == 1 ? 0 : row_bytes};
    }

    const int64_t flags = done.dimensions().size() == 0 ? 0 : done.dimensions()[0];

    if (count > 0 && flags != 1 && flags != lanes) {
        return ffi::Error::InvalidArgument("one done flag for every lane or one shared");
    }

    if (count == 0 || lanes == 0) {
        return ffi::Error::Success();
    }

    const int64_t blocks = (lanes + kWarpsPerBlock - 1) / kWarpsPerBlock;
    reset_where_done<<<static_cast<unsigned>(blocks), kWarpsPerBlock * 32, 0, stream>>>(reinterpret_cast<const uint8_t*>(done.typed_data()), flags == 1 ? 0 : 1, leaves, lanes);
    return nanorts::launched();
}

const nanorts::Kernels nanorts_reset_where_done_kernels("nanorts_reset_where_done", {nanorts::kernel(reset_where_done)});

}

NANORTS_HANDLER(nanorts_reset_where_done, ResetWhereDone,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .Arg<ffi::Buffer<ffi::PRED>>()
                    .RemainingArgs()
                    .RemainingRets()
                    .Attr<std::string_view>("build_id")
                    .Attr<int64_t>("leaves"));
