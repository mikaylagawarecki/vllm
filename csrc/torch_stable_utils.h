#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>

#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace {

// Device properties cache for stable ABI compatibility
// Uses raw CUDA/HIP APIs instead of ATen functions
std::deque<std::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

inline void initDeviceVectors() {
  static bool init_flag [[maybe_unused]] = []() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
      STD_TORCH_CHECK(false, "cudaGetDeviceCount failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    device_flags.resize(device_count);
    device_properties.resize(device_count);
    return true;
  }();
}

inline void initDeviceProperty(int device_index) {
  cudaDeviceProp device_prop{};
  cudaError_t err = cudaGetDeviceProperties(&device_prop, device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDeviceProperties failed: " +
                               std::string(cudaGetErrorString(err)));
  }
  device_properties[device_index] = device_prop;
}

}  // anonymous namespace

// Get device properties using raw CUDA/HIP APIs (stable ABI compatible)
inline cudaDeviceProp* get_device_prop() {
  initDeviceVectors();
  int device_index;
  cudaError_t err = cudaGetDevice(&device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(
        false, "cudaGetDevice failed: " + std::string(cudaGetErrorString(err)));
  }

  std::call_once(device_flags[device_index], initDeviceProperty, device_index);
  return &device_properties[device_index];
}

// Determines the preferred FP8 type for the current platform.
// Returns true for OCP format (Float8_e4m3fn), false for FNUZ format
// (Float8_e4m3fnuz). On CUDA this always returns true. On ROCm it checks
// device properties to determine the format.
inline bool is_fp8_ocp() {
#ifndef USE_ROCM
  return true;
#else
  auto* dprops = get_device_prop();
  std::string device_arch = dprops->gcnArchName;
  // gfx94x devices use FNUZ format, others use OCP format
  size_t substring = device_arch.find("gfx94");
  return substring == std::string::npos;
#endif
}

// Utility to get the current CUDA stream for a given device using stable APIs.
// Returns a cudaStream_t for use in kernel launches.
inline cudaStream_t get_current_cuda_stream(int32_t device_index) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}
