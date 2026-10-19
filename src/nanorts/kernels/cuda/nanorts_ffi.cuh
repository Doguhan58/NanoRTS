#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace nanorts {

namespace ffi = xla::ffi;

std::string_view build_id();

struct Target {
    explicit Target(const char* name);
};

struct Kernels {
    Kernels(const char* target, std::vector<const void*> functions);
};

template <typename... Parameters>
const void* kernel(void (*function)(Parameters...)) {
    return reinterpret_cast<const void*>(function);
}

inline ffi::Error check_build_id(std::string_view attribute) {
    if (attribute != build_id()) {
        return ffi::Error::InvalidArgument(
            "this executable was traced against another nanoRTS kernel build; "
            "clear the compilation cache or retrace");
    }

    return ffi::Error::Success();
}

inline ffi::Error launched() {
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(error));
    }

    return ffi::Error::Success();
}

inline int64_t lanes(const ffi::AnyBuffer& buffer) {
    auto dimensions = buffer.dimensions();
    return dimensions.size() == 0 ? 0 : dimensions[0];
}

}

#define NANORTS_HANDLER(name, implementation, binding)                                                        \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, implementation, binding, {::xla::ffi::Traits::kCmdBufferCompatible}); \
    static ::nanorts::Target name##_target(#name)
