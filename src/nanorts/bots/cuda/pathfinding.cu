#include <cstdint>

#include "astar.cuh"
#include "nanorts_ffi.cuh"

namespace {

namespace ffi = xla::ffi;

constexpr int kThreads = 32;
constexpr int64_t kSharedLimit = 48 * 1024;
constexpr int64_t kAnswer = 4;

template <typename T>
struct Rows {
    const T* data;
    int64_t stride;
};

__global__ void __launch_bounds__(kThreads) search_kernel(Rows<bool> free, Rows<int32_t> start, Rows<int32_t> target, Rows<int32_t> range, int width, int height, int32_t* out) {
    extern __shared__ uint64_t shared[];
    const int64_t lane = blockIdx.x;
    const int cells = width * height;
    nanorts::astar::Scratch scratch = nanorts::astar::Scratch::carve(shared, cells);
    const bool* searched = free.data + lane * free.stride;

    for (int cell = threadIdx.x; cell < cells; cell += blockDim.x) {
        scratch.clear(cell, searched[cell]);
    }

    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    const nanorts::astar::Path path = nanorts::astar::search(scratch, start.data[lane * start.stride], target.data[lane * target.stride], range.data[lane * range.stride], width, height);
    int32_t* answer = out + lane * kAnswer;
    answer[0] = path.found;
    answer[1] = path.step;
    answer[2] = path.distance;
    answer[3] = path.pops;
}

template <typename T>
Rows<T> rows(const ffi::AnyBuffer& leaf, int64_t row) {
    return Rows<T>{static_cast<const T*>(leaf.untyped_data()), nanorts::lanes(leaf) == 1 ? 0 : row};
}

ffi::Error Search(cudaStream_t stream, ffi::AnyBuffer free, ffi::AnyBuffer start, ffi::AnyBuffer target, ffi::AnyBuffer range, ffi::Result<ffi::AnyBuffer> out, std::string_view build_id, int64_t width, int64_t height) {
    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (free.element_type() != ffi::PRED || start.element_type() != ffi::S32 || target.element_type() != ffi::S32 || range.element_type() != ffi::S32 || out->element_type() != ffi::S32) {
        return ffi::Error::InvalidArgument(
            "nanorts_astar_search takes bool free cells and int32 start, target and "
            "range, and returns int32");
    }

    const int64_t lanes = nanorts::lanes(*out);
    const int64_t cells = width * height;
    const int64_t shared = cells * nanorts::astar::kScratchBytesPerCell;

    if (width <= 0 || height <= 0 || shared > kSharedLimit || static_cast<int64_t>(out->element_count()) != lanes * kAnswer) {
        return ffi::Error::InvalidArgument(
            "the result is one four-field answer per lane, over a field whose search "
            "fits one block's shared memory");
    }

    if (lanes == 0) {
        return ffi::Error::Success();
    }

    const auto shared_or_per_lane = [lanes](const ffi::AnyBuffer& leaf, int64_t row) {
        const int64_t extent = nanorts::lanes(leaf);
        return (extent == 1 || extent == lanes) && static_cast<int64_t>(leaf.element_count()) == extent * row;
    };

    if (!shared_or_per_lane(free, cells) || !shared_or_per_lane(start, 1) || !shared_or_per_lane(target, 1) || !shared_or_per_lane(range, 1)) {
        return ffi::Error::InvalidArgument(
            "each operand holds one row per lane or one shared row: the free cells "
            "and a scalar start, target and range");
    }

    search_kernel<<<static_cast<unsigned>(lanes), kThreads, static_cast<size_t>(shared), stream>>>(rows<bool>(free, cells), rows<int32_t>(start, 1), rows<int32_t>(target, 1), rows<int32_t>(range, 1), static_cast<int>(width), static_cast<int>(height), static_cast<int32_t*>(out->untyped_data()));
    return nanorts::launched();
}

const nanorts::Kernels nanorts_astar_search_kernels("nanorts_astar_search", {nanorts::kernel(search_kernel)});

}

NANORTS_HANDLER(nanorts_astar_search, Search,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Ret<ffi::AnyBuffer>()
                    .Attr<std::string_view>("build_id")
                    .Attr<int64_t>("width")
                    .Attr<int64_t>("height"));
