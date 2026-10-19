#pragma once

#include "action_kind.cuh"

namespace nanorts::core {

__device__ inline int32_t none_duration(int32_t none_duration) {
    return none_duration;
}

__device__ constexpr bool none_is_valid() {
    return true;
}

}
