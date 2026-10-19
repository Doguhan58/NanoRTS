#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "nanorts_ffi.cuh"

namespace nanorts::whole_act {

namespace ffi = xla::ffi;

constexpr int kPlayers = 2;
constexpr int kCommandFields = 6;
constexpr int kWarp = 32;
constexpr int kMinThreads = 64;
constexpr int kMaxThreads = 256;
constexpr int64_t kSharedLimit = 49152;

template <typename T>
struct Lanes {
    const T* data;
    int64_t stride;

    __device__ const T* at(int64_t lane) const {
        return data + lane * stride;
    }
};

template <typename Slot, typename Cell>
struct StateRows {
    Lanes<int32_t> birth;
    Lanes<int8_t> unit_type;
    Lanes<int8_t> owner;
    Lanes<Cell> cell;
    Lanes<int32_t> carried;
    Lanes<int8_t> pending_kind;
    Lanes<int8_t> pending_direction;
    Lanes<int8_t> pending_produced_type;
    Lanes<Slot> entity_at;
    Lanes<int32_t> resources;
    Lanes<bool> terrain;
};

constexpr int kStateOperands = 11;
constexpr int kCellOperand = 3;
constexpr int kBoardOperand = 8;

struct Extents {
    int32_t width;
    int32_t height;
    int32_t cells;
    int32_t entities;
    int32_t neutral;
    int32_t capacity;
    int32_t types;
};

inline int threads_for(int64_t capacity) {
    const int64_t rounded = (capacity + kWarp - 1) / kWarp * kWarp;
    return rounded < kMinThreads ? kMinThreads : static_cast<int>(rounded);
}

inline bool index_type(ffi::DataType type) {
    return type == ffi::DataType::S16 || type == ffi::DataType::S32;
}

class Operands {
  public:
    Operands(std::string_view target, ffi::RemainingArgs args, int64_t lanes) : target_(target), args_(args), lanes_(lanes) {}

    template <typename T>
    ffi::Error next(int64_t elements, Lanes<T>& into) {
        const int64_t index = taken_++;
        auto buffer = args_.get<ffi::AnyBuffer>(index);

        if (!buffer.has_value()) {
            return buffer.error();
        }

        const int64_t extent = nanorts::lanes(*buffer);

        if ((extent != 1 && extent != lanes_) || ffi::ByteWidth(buffer->element_type()) != sizeof(T) || static_cast<int64_t>(buffer->size_bytes()) != extent * elements * static_cast<int64_t>(sizeof(T))) {
            return refused("operand " + std::to_string(index) + " is not one row per lane or one shared row");
        }

        into = Lanes<T>{static_cast<const T*>(buffer->untyped_data()), extent == 1 ? 0 : elements};
        return ffi::Error::Success();
    }

    ffi::ErrorOr<ffi::AnyBuffer> at(int64_t index) const {
        return args_.get<ffi::AnyBuffer>(index);
    }

    int64_t count() const {
        return static_cast<int64_t>(args_.size());
    }

    ffi::Error refused(const std::string& why) const {
        return ffi::Error::InvalidArgument(std::string(target_) + ": " + why);
    }

  private:
    std::string_view target_;
    ffi::RemainingArgs args_;
    int64_t lanes_;
    int64_t taken_ = 0;
};

inline ffi::Error read_state(const Operands& operands, Extents& size, ffi::DataType& slot, ffi::DataType& cell) {
    using ffi::DataType;
    constexpr DataType kIndex = DataType::INVALID;
    const DataType expected[kStateOperands] = {
        DataType::S32,
        DataType::S8,
        DataType::S8,
        kIndex,
        DataType::S32,
        DataType::S8,
        DataType::S8,
        DataType::S8,
        kIndex,
        DataType::S32,
        DataType::PRED
    };

    if (operands.count() < kStateOperands) {
        return operands.refused("the State leaves are missing");
    }

    for (int k = 0; k < kStateOperands; ++k) {
        auto buffer = operands.at(k);

        if (!buffer.has_value()) {
            return buffer.error();
        }

        const DataType type = buffer->element_type();

        if (expected[k] == kIndex ? !index_type(type) : type != expected[k]) {
            return operands.refused("State leaf " + std::to_string(k) + " has another dtype");
        }
    }

    const auto entities = operands.at(0)->dimensions();
    const auto field = operands.at(kBoardOperand)->dimensions();

    if (entities.size() != 2 || field.size() != 3) {
        return operands.refused("`birth` is `[lanes, E]` and `entity_at` `[lanes, H, W]`");
    }

    size.entities = static_cast<int32_t>(entities[1]);
    size.height = static_cast<int32_t>(field[1]);
    size.width = static_cast<int32_t>(field[2]);
    size.cells = size.height * size.width;
    slot = operands.at(kBoardOperand)->element_type();
    cell = operands.at(kCellOperand)->element_type();
    return ffi::Error::Success();
}

template <typename Slot, typename Cell>
ffi::Error bind_state(Operands& operands, const Extents& size, StateRows<Slot, Cell>& rows) {
    const int64_t entities = size.entities;

    if (auto e = operands.next(entities, rows.birth); e.failure()) {
        return e;
    }

    if (auto e = operands.next(entities, rows.unit_type); e.failure()) {
        return e;
    }

    if (auto e = operands.next(entities, rows.owner); e.failure()) {
        return e;
    }

    if (auto e = operands.next(entities, rows.cell); e.failure()) {
        return e;
    }

    if (auto e = operands.next(entities, rows.carried); e.failure()) {
        return e;
    }

    if (auto e = operands.next(entities, rows.pending_kind); e.failure()) {
        return e;
    }

    if (auto e = operands.next(entities, rows.pending_direction); e.failure()) {
        return e;
    }

    if (auto e = operands.next(entities, rows.pending_produced_type); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{size.cells}, rows.entity_at); e.failure()) {
        return e;
    }

    if (auto e = operands.next(int64_t{kPlayers}, rows.resources); e.failure()) {
        return e;
    }

    return operands.next(int64_t{size.cells}, rows.terrain);
}

template <typename T>
ffi::Error carry_result(const Operands& operands, ffi::RemainingRets results, int64_t result, int64_t operand, int64_t lanes, int64_t elements, T*& into) {
    auto out = results.get<ffi::AnyBuffer>(result);

    if (!out.has_value()) {
        return out.error();
    }

    auto given = operands.at(operand);

    if (!given.has_value()) {
        return given.error();
    }

    ffi::AnyBuffer buffer = **out;

    if (buffer.element_type() != given->element_type() || ffi::ByteWidth(buffer.element_type()) != sizeof(T) || nanorts::lanes(*given) != lanes || static_cast<int64_t>(buffer.size_bytes()) != lanes * elements * static_cast<int64_t>(sizeof(T))) {
        return operands.refused("carry result " + std::to_string(result) + " is not its operand's row on every lane");
    }

    into = static_cast<T*>(buffer.untyped_data());
    return ffi::Error::Success();
}

}
