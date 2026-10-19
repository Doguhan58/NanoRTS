#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "nanorts_build_id.h"
#include "nanorts_ffi.cuh"

namespace nanorts {

namespace {

std::vector<std::string>& registry() {
    static std::vector<std::string> names;
    return names;
}

struct TargetKernels {
    std::string target;
    std::vector<const void*> functions;
};

std::vector<TargetKernels>& kernel_lists() {
    static std::vector<TargetKernels> lists;
    return lists;
}

constexpr uint32_t kProbeLanes = 32;

__host__ __device__ uint32_t probe_value(uint32_t lane) {
    return lane * 2654435761u + 0x9e3779b9u;
}

__global__ void probe(uint32_t* out) {
    const uint32_t lane = threadIdx.x;
    const uint32_t value = probe_value(lane);
    const uint32_t odd = __ballot_sync(0xffffffffu, (value & 1u) != 0);
    out[lane] = __shfl_xor_sync(0xffffffffu, value, 1) ^ odd;
}

uint32_t expected(uint32_t lane) {
    uint32_t odd = 0;

    for (uint32_t other = 0; other < kProbeLanes; ++other) {
        odd |= (probe_value(other) & 1u) << other;
    }

    return probe_value(lane ^ 1u) ^ odd;
}

std::string failed(const std::string& what, cudaError_t error) {
    return what + ": " + cudaGetErrorString(error);
}

std::string run_probe() {
    cudaStream_t stream = nullptr;
    cudaError_t error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    if (error != cudaSuccess) {
        return failed("the probe's stream", error);
    }

    uint32_t* answer = nullptr;
    uint32_t host[kProbeLanes] = {};
    error = cudaMalloc(&answer, sizeof host);

    if (error == cudaSuccess) {
        probe<<<1, kProbeLanes, 0, stream>>>(answer);
        error = cudaGetLastError();
    }

    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(host, answer, sizeof host, cudaMemcpyDeviceToHost, stream);
    }

    if (error == cudaSuccess) {
        error = cudaStreamSynchronize(stream);
    }

    if (answer != nullptr) {
        cudaFree(answer);
    }

    cudaStreamDestroy(stream);

    if (error != cudaSuccess) {
        return failed("the probe launch", error);
    }

    for (uint32_t lane = 0; lane < kProbeLanes; ++lane) {
        if (host[lane] != expected(lane)) {
            return "the probe returned " + std::to_string(host[lane]) + " on lane " + std::to_string(lane) + ", not " + std::to_string(expected(lane));
        }
    }

    return "";
}

std::string unlisted_target() {
    for (const std::string& name : registry()) {
        bool listed = false;

        for (const TargetKernels& list : kernel_lists()) {
            listed |= list.target == name;
        }

        if (!listed) {
            return name + " lists no kernels";
        }
    }

    return "";
}

std::string check_device(int device) {
    const std::string where = "device " + std::to_string(device) + ": ";
    cudaError_t error = cudaSetDevice(device);

    if (error != cudaSuccess) {
        return where + failed("cudaSetDevice", error);
    }

    for (const TargetKernels& list : kernel_lists()) {
        for (const void* function : list.functions) {
            cudaFuncAttributes attributes;
            error = cudaFuncGetAttributes(&attributes, function);

            if (error != cudaSuccess) {
                return where + failed(list.target + " has no image it loads", error);
            }
        }
    }

    const std::string probe = run_probe();
    return probe.empty() ? probe : where + probe;
}

std::string check_devices(const int32_t* devices, int64_t count) {
    int original = 0;
    cudaError_t error = cudaGetDevice(&original);

    if (error != cudaSuccess) {
        return failed("cudaGetDevice", error);
    }

    std::string failure;

    for (int64_t index = 0; index < count && failure.empty(); ++index) {
        failure = check_device(devices[index]);
    }

    cudaSetDevice(original);
    return failure;
}

std::string self_check(const int32_t* devices, int64_t count) {
    std::string failure = unlisted_target();

    if (failure.empty()) {
        failure = check_devices(devices, count);
    }

    cudaGetLastError();
    return failure;
}

}

std::string_view build_id() {
    return NANORTS_BUILD_ID;
}

Target::Target(const char* name) {
    registry().emplace_back(name);
}

Kernels::Kernels(const char* target, std::vector<const void*> functions) {
    kernel_lists().push_back(TargetKernels{target, std::move(functions)});
}

}

extern "C" const char* nanorts_build_id() {
    return NANORTS_BUILD_ID;
}

extern "C" const char* nanorts_targets() {
    static const std::string joined = [] {
        std::string names;

        for (const auto& name : nanorts::registry()) {
            if (!names.empty()) {
                names += ",";
            }

            names += name;
        }

        return names;
    }();
    return joined.c_str();
}

extern "C" const char* nanorts_self_check(const int32_t* devices, int64_t count) {
    static std::string failure;
    failure = nanorts::self_check(devices, count);
    return failure.c_str();
}
