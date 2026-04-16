# Stable ABI Migration Best Practices

## Lessons learned from migrating vLLM CUDA kernels to torch stable ABI

### Copy comments exactly when migrating registrations

When moving op registrations from `csrc/torch_bindings.cpp` to `csrc/libtorch_stable/torch_bindings.cpp`, copy the comments verbatim. Don't truncate or paraphrase.

### Use `torch::stable::empty_like` when possible

`torch::stable::empty_like(tensor)` exists. Use it instead of manually reconstructing `torch::stable::empty(t.sizes(), t.scalar_type(), std::nullopt, t.device())`.

### `torch::stable::empty` accepts scalars for size

`IntHeaderOnlyArrayRef` has an implicit single-element constructor, so `torch::stable::empty(workspace_size, ...)` works — no need for `{static_cast<int64_t>(workspace_size)}`. `size_t` converts implicitly to `int64_t`.

### Preserve `const` qualifiers exactly

See `feedback_const_qualifiers.md` for the full rule. In short: match the original qualifier exactly — don't add `const` where it wasn't, and don't remove it where it was.

### Avoid gratuitous reformatting

When migrating shared headers, only change what's necessary (types, macros). Don't let clang-format reformat unchanged logic — it creates noisy diffs and obscures the actual changes. If clang-format reformats untouched code, revert the formatting to match the original.

### `#ifndef USE_ROCM` guards

When migrating ops.h / torch_bindings.cpp, ensure every op that was guarded with `#ifndef USE_ROCM` in the original remains guarded in the stable version. Structural differences (e.g. consolidating multiple small guard blocks into one) are fine as long as correctness matches.

### Avoid `using Tensor = torch::stable::Tensor;`

Don't create a `Tensor` alias — it conflicts with `cute::Tensor` (a template) when `using namespace cute;` is in scope, causing ambiguity during nvcc template instantiation. Always use fully qualified `torch::stable::Tensor` in stable ABI files. Similarly, use `TensorType` (not `Tensor`) for aliases in shared headers.

### `.data_ptr()` conversions

- Non-templated `.data_ptr()` (returns `void*`) exists on `torch::stable::Tensor` — no conversion needed. `static_cast<T*>(t.data_ptr())` can stay as-is. Don't change to `.mutable_data_ptr()` or `.const_data_ptr()` unless the original used a templated version.
- Templated `.data_ptr<T>()` does NOT exist on `torch::stable::Tensor`. Use `.mutable_data_ptr<T>()` or `.const_data_ptr<T>()` instead, depending on whether the pointer is written to or read from.

### Use dispatch macros, not manual if/else

`VLLM_DISPATCH_HALF_TYPES` and other `AT_DISPATCH_*` macros don't work with stable ABI. Use the stable equivalents in `libtorch_stable/dispatch_utils.h` (e.g. `VLLM_STABLE_DISPATCH_HALF_TYPES`). Don't manually expand dispatch macros into if/else chains — add a new stable dispatch macro if one doesn't exist yet.

### Keep op registrations in source files

The original pattern uses `TORCH_LIBRARY_IMPL_EXPAND` inside the namespace to register ops directly. The stable equivalent is `STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m)` inside the namespace, referencing functions by their unqualified names (e.g. `&mm`). Note: register under `_C` (the op namespace), not `_C_stable_libtorch` (the CMake target/extension name). Schema defs (`m.def(...)`) stay in `torch_bindings.cpp`. Don't create global wrapper functions outside the namespace.

### `torch::stable::Device` supports CPU

`torch::stable::Device(torch::stable::DeviceType::CPU)` works. Don't replace CPU tensor creation + `.to(device)` with `std::vector` + `cudaMemcpy` workarounds. Use `torch::stable::to(tensor, device)` for device transfers.

### Use `torch::headeronly::` for types in shared headers

In shared headers (used by both stable and unstable targets), prefer `torch::headeronly::Half`, `torch::headeronly::BFloat16`, `torch::headeronly::ScalarType`, `torch::headeronly::CppTypeToScalarType` over `c10::` equivalents. The headeronly namespace re-exports these from c10.

### Don't add comments that weren't in the original

When migrating, don't add explanatory comments like `// moved from _C to _C_stable_libtorch` or `// W4A8 ops are registered directly in source files`. Keep the migrated code structurally matching the original.

### Don't remove dead includes unnecessarily

If the original file included a header that wasn't used, it's fine to remove it during migration — but verify that no transitive includers depended on it.

### `TORCH_BOX` requires trivially copyable types

`TORCH_BOX` uses `stableivalue_conversions.h` which requires all parameter types to be trivially copyable. `std::tuple<int64_t, int64_t>` is NOT trivially copyable, so `std::optional<std::tuple<int64_t, int64_t>>` fails with a static assertion. **Workaround**: Use `int[]?` in the schema with `std::optional<torch::headeronly::IntHeaderOnlyArrayRef>` in C++, and add a runtime `STD_TORCH_CHECK(arr->size() == 2, ...)`. Python callers can pass tuples or lists unchanged since `int[]` accepts both.

### `AT_CUDA_CHECK` → `STD_CUDA_CHECK`

`AT_CUDA_CHECK(expr)` from `<ATen/cuda/Exceptions.h>` is not available in stable ABI. Use `STD_CUDA_CHECK(expr)` from `<torch/csrc/stable/macros.h>` (included transitively via `torch_utils.h`). Also available: `STD_CUDA_KERNEL_LAUNCH_CHECK()` which wraps `STD_CUDA_CHECK(cudaGetLastError())`.

### `TORCH_CHECK` / `TORCH_CHECK_EQ` / `TORCH_CHECK_LE` → `STD_TORCH_CHECK`

The ATen `TORCH_CHECK*` macros are not available. Use `STD_TORCH_CHECK(cond, ...)` for assertions. For equality/comparison checks, write them out manually: `STD_TORCH_CHECK(a == b, "...")`.

### Zero-initialized tensor creation

`torch::zeros(sizes, options)` doesn't exist in stable ABI. Use `torch::stable::empty(...)` + `torch::stable::fill_(tensor, 0.0)`. Or use `torch::stable::new_zeros(ref_tensor, sizes)` if inheriting dtype/device from a reference tensor is acceptable (but this loses the ability to set dtype and device independently).

### `c10::cuda::current_device()` → `cudaGetDevice`

Not available in stable ABI. Use raw CUDA API: `int device; cudaGetDevice(&device);`

### `at::DeviceGuard` / `at::cuda::OptionalCUDAGuard` → `torch::stable::accelerator::DeviceGuard`

Replace with `const torch::stable::accelerator::DeviceGuard device_guard(tensor.get_device_index())`.

### Build command

```bash
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vllm-dev && export CUDA_HOME=/usr/local/cuda-12.9 && export PATH=/usr/local/cuda-12.9/bin:$PATH && uv pip install --no-build-isolation -U vllm -e . --excludes excludes.txt &> out.txt
```

Note: Use CUDA 12.9 (not 12.8) because 12.8 on this machine has an incomplete installation (missing libcudart).
