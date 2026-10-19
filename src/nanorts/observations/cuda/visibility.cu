#include <algorithm>
#include <cstdint>

#include "nanorts_ffi.cuh"

namespace {

namespace ffi = xla::ffi;

constexpr int32_t kAbsent = -1;

constexpr int64_t kMaxCells = 48 * 1024;

template <typename T>
struct Rows {
    const T* data;
    int64_t stride;
};

template <typename Cell>
__global__ void visible_cells(Rows<int32_t> birth, Rows<int8_t> unit_type, Rows<Cell> cell, const int32_t* radius, int64_t type_count, int64_t first, int64_t count, int width, int height, bool* visible_out) {
    extern __shared__ unsigned char seen[];
    const int64_t environment = blockIdx.x;
    const int cells = width * height;

    for (int cell_index = threadIdx.x; cell_index < cells; cell_index += blockDim.x) {
        seen[cell_index] = 0;
    }

    __syncthreads();
    const int32_t* births = birth.data + environment * birth.stride;
    const int8_t* unit_types = unit_type.data + environment * unit_type.stride;
    const Cell* unit_cells = cell.data + environment * cell.stride;

    for (int64_t slot = first + threadIdx.x; slot < first + count; slot += blockDim.x) {
        if (births[slot] == kAbsent) {
            continue;
        }

        const int64_t unit_cell = static_cast<int64_t>(unit_cells[slot]);
        const int64_t unit_type_index = static_cast<int64_t>(unit_types[slot]);

        if (unit_cell < 0 || unit_cell >= cells || unit_type_index < 0 || unit_type_index >= type_count) {
            continue;
        }

        const int x = static_cast<int>(unit_cell % width);
        const int y = static_cast<int>(unit_cell / width);
        const int sight_radius = radius[unit_type_index];
        const int squared_radius = sight_radius * sight_radius;

        for (int seen_y = max(y - sight_radius, 0); seen_y <= min(y + sight_radius, height - 1); ++seen_y) {
            const int dy = seen_y - y;

            for (int seen_x = max(x - sight_radius, 0); seen_x <= min(x + sight_radius, width - 1); ++seen_x) {
                const int dx = seen_x - x;

                if (dx * dx + dy * dy <= squared_radius) {
                    seen[seen_y * width + seen_x] = 1;
                }
            }
        }
    }

    __syncthreads();
    bool* visible = visible_out + environment * cells;

    for (int cell_index = threadIdx.x; cell_index < cells; cell_index += blockDim.x) {
        visible[cell_index] = seen[cell_index] != 0;
    }
}

template <typename T>
Rows<T> rows(const ffi::AnyBuffer& leaf, int64_t slots) {
    return Rows<T>{static_cast<const T*>(leaf.untyped_data()), nanorts::lanes(leaf) == 1 ? 0 : slots};
}

template <typename Cell>
ffi::Error launch(cudaStream_t stream, const ffi::AnyBuffer& birth, const ffi::AnyBuffer& unit_type, const ffi::AnyBuffer& cell, const ffi::AnyBuffer& radius, ffi::Result<ffi::AnyBuffer>& out, int64_t lanes, int64_t slots, int64_t first, int64_t count, int width, int height) {
    const int cells = width * height;
    const int threads = static_cast<int>(std::clamp<int64_t>((std::max<int64_t>(count, cells) + 31) / 32 * 32, 32, 256));
    visible_cells<Cell><<<static_cast<unsigned>(lanes), threads, static_cast<size_t>(cells), stream>>>(rows<int32_t>(birth, slots), rows<int8_t>(unit_type, slots), rows<Cell>(cell, slots), static_cast<const int32_t*>(radius.untyped_data()), static_cast<int64_t>(radius.element_count()), first, count, width, height, static_cast<bool*>(out->untyped_data()));
    return nanorts::launched();
}

ffi::Error VisibleCells(cudaStream_t stream, ffi::AnyBuffer birth, ffi::AnyBuffer unit_type, ffi::AnyBuffer cell, ffi::AnyBuffer radius, ffi::Result<ffi::AnyBuffer> out, std::string_view build_id, int64_t first, int64_t count, int64_t width, int64_t height) {
    if (ffi::Error error = nanorts::check_build_id(build_id); error.failure()) {
        return error;
    }

    if (birth.element_type() != ffi::S32 || unit_type.element_type() != ffi::S8 || radius.element_type() != ffi::S32 || out->element_type() != ffi::PRED) {
        return ffi::Error::InvalidArgument(
            "nanorts_visible_cells takes int32 birth, int8 unit_type, int32 radius "
            "and returns bool");
    }

    const int64_t lanes = nanorts::lanes(*out);
    const int64_t cells = width * height;

    if (width <= 0 || height <= 0 || cells > kMaxCells || static_cast<int64_t>(out->element_count()) != lanes * cells) {
        return ffi::Error::InvalidArgument("the result is one [height, width] field per lane");
    }

    if (lanes == 0) {
        return ffi::Error::Success();
    }

    const auto shared_or_per_lane = [lanes](const ffi::AnyBuffer& leaf) {
        const int64_t extent = nanorts::lanes(leaf);
        return extent == 1 || extent == lanes;
    };

    if (!shared_or_per_lane(birth) || !shared_or_per_lane(unit_type) || !shared_or_per_lane(cell) || nanorts::lanes(radius) != 1) {
        return ffi::Error::InvalidArgument(
            "each per-slot leaf holds one row per lane or one shared row, the radius "
            "table one shared row");
    }

    const auto slots_of = [](const ffi::AnyBuffer& leaf) { return static_cast<int64_t>(leaf.element_count()) / nanorts::lanes(leaf); };
    const int64_t slots = slots_of(birth);

    if (slots_of(unit_type) != slots || slots_of(cell) != slots || first < 0 || count < 0 || first + count > slots || radius.element_count() == 0) {
        return ffi::Error::InvalidArgument("the block lies inside every per-slot leaf");
    }

    const int w = static_cast<int>(width), h = static_cast<int>(height);

    switch (cell.element_type()) {
        case ffi::S8:
            return launch<int8_t>(stream, birth, unit_type, cell, radius, out, lanes, slots, first, count, w, h);
        case ffi::S16:
            return launch<int16_t>(stream, birth, unit_type, cell, radius, out, lanes, slots, first, count, w, h);
        case ffi::S32:
            return launch<int32_t>(stream, birth, unit_type, cell, radius, out, lanes, slots, first, count, w, h);
        default:
            return ffi::Error::InvalidArgument("cell is int8, int16 or int32");
    }
}

const nanorts::Kernels nanorts_visible_cells_kernels("nanorts_visible_cells", {nanorts::kernel(visible_cells<int8_t>), nanorts::kernel(visible_cells<int16_t>), nanorts::kernel(visible_cells<int32_t>)});

}

NANORTS_HANDLER(nanorts_visible_cells, VisibleCells,
                ffi::Ffi::Bind()
                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Arg<ffi::AnyBuffer>()
                    .Ret<ffi::AnyBuffer>()
                    .Attr<std::string_view>("build_id")
                    .Attr<int64_t>("first")
                    .Attr<int64_t>("count")
                    .Attr<int64_t>("width")
                    .Attr<int64_t>("height"));
