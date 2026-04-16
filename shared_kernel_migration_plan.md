# Plan: Migrate Shared CUDA/ROCm Kernels to Torch Stable ABI

## Context

The vLLM stable ABI migration moves CUDA/C++ kernels from the legacy `_C` extension (using ATen headers) to `_C_stable_libtorch` (using `torch/csrc/stable/` headers). This enables binary compatibility across PyTorch versions. CUDA-only kernels are largely done on `temp8`. The activation kernel migration is in progress. This plan covers the remaining **shared CUDA/ROCm kernels** — those that compile for both NVIDIA and AMD GPUs.

The ROCm enablement for `_C_stable_libtorch` is already done (commit on temp8), so each migration is 2 commits: a pure `git mv` move, then the code conversion.

## Proposed Migration Order

Ordered by complexity (easiest first), to build momentum and catch infrastructure issues early.

---

### Phase 1: Easy wins (straightforward type conversions, few shared headers)

#### 1. Non-CUTLASS w8a8 INT8 quant

- **Files**: `csrc/quantization/w8a8/int8/scaled_quant.cu` (328 lines)
- **Ops**: `static_scaled_int8_quant`, `dynamic_scaled_int8_quant` (2 ops)
- **Why easy**: Self-contained, already uses `libtorch_stable/quantization/vectorization_utils.cuh` (migrated header). ROCm guards are internal (PTX vs `std::nearbyint`) and don't affect the migration pattern. Rebase plan says "likely cherry-pickable with minor fixup".
- **Shared headers**: `dispatch_utils.h`, `cub_helpers.h`, `vectorization_utils.cuh` (already migrated)
- **Test**: `tests/kernels/quantization/test_int8_quant.py`

#### 2. Non-CUTLASS w8a8 FP8 quant

- **Files**: `csrc/quantization/w8a8/fp8/common.cu` (404 lines)
- **Ops**: `static_scaled_fp8_quant`, `dynamic_scaled_fp8_quant`, `dynamic_per_token_scaled_fp8_quant` (3 ops)
- **Why easy**: Similar to INT8 — already uses migrated `vectorization_utils.cuh`. ROCm guards are in `common.cuh` header (NVIDIA vs AMD quant_utils), not in the entry-point code. The `std::optional<at::Tensor>` → `std::optional<torch::stable::Tensor>` and `std::optional<std::tuple<int64_t, int64_t>>` types need care.
- **Shared headers**: `common.cuh` (has `#ifndef USE_ROCM` for nvidia/amd quant_utils), `dispatch_utils.h`, `cub_helpers.h`, `vectorization_utils.cuh`
- **Complication**: `common.cuh` is included by both this file and `layernorm_quant_kernels.cu` (not yet migrated). Will need the `#ifdef TORCH_TARGET_VERSION` / `TensorType` pattern in `common.cuh` if both targets include it, or move the file to `libtorch_stable/` and update the remaining `_C` consumer.
- **Test**: `tests/kernels/quantization/test_fp8_quant.py`

#### 3. GPTQ kernels

- **Files**: `csrc/quantization/gptq/q_gemm.cu` (1862 lines) + sub-headers (`compat.cuh`, `matrix_view.cuh`, `qdq_*.cuh`)
- **Ops**: `gptq_gemm`, `gptq_shuffle` (2 ops)
- **Why relatively easy**: No temp3 reference needed — straightforward `torch::Tensor` → `torch::stable::Tensor`. ROCm guards are internal (hipBLAS wrapper, `half2` init). Large file but the entry points are simple. Sub-headers are GPTQ-only (no sharing with other kernel groups).
- **Note**: Uses `ATen/cuda/CUDAContext.h` for cuBLAS handle → needs `get_current_cuda_blas_handle()` from `torch_utils.h`. Also uses `torch::empty` for output tensor allocation → needs `torch::stable::empty`.
- **Test**: `tests/kernels/quantization/test_gptq.py`

#### 4. GGML kernels

- **Files**: `csrc/quantization/gguf/gguf_kernel.cu` (543 lines) + sub-headers (`ggml-common.h`, `vecdotq.cuh`, `dequantize.cuh`, `mmvq.cuh`, `mmq.cuh`, `moe.cuh`, `moe_vec.cuh`)
- **Ops**: `ggml_dequantize`, `ggml_mul_mat_vec_a8`, `ggml_mul_mat_a8`, `ggml_moe_a8`, `ggml_moe_a8_vec`, `ggml_moe_get_block_size` (6 ops)
- **Why moderate**: No temp3 reference, but straightforward conversions. Several ops return `torch::Tensor` (not void) — need `torch::stable::empty` for output allocation. Sub-headers are GGML-only. ROCm compatibility comes via `cuda_compat.h` macros (already shared infrastructure).
- **Note**: `ggml_moe_get_block_size` takes/returns only `int64_t` — register as `CompositeExplicitAutograd` (no tensor dispatch key).
- **Test**: `tests/kernels/quantization/test_ggml.py`, `test_gguf.py` (note: `ggml_moe_get_block_size` has no dedicated test coverage)

#### 5. Custom all reduce

- **Files**: `csrc/custom_all_reduce.cu` (190 lines) + `csrc/custom_all_reduce.cuh` (632 lines)
- **Ops**: `dispose`, `meta_size`, `register_buffer`, `get_graph_buffer_ipc_meta`, `allocate_shared_buffer_and_handle`, `open_mem_handle`, `free_shared_buffer`, `init_custom_ar`, `all_reduce`, `register_graph_buffers` (10 ops)
- **Why moderate**: Rebase plan says "likely cherry-pickable". ROCm guards are in the `.cuh` (barrier primitives) and `.cu` (memory alloc) — these are internal and don't affect the migration pattern. However, uses `fptr_t` (opaque pointer type), `std::vector<int64_t>`, and `std::tuple` in signatures — need to verify these work with `TORCH_BOX`.
- **Complication**: These ops use `std::vector<int64_t>` and `fptr_t` parameters which may not be supported by `TORCH_BOX` / stable ABI op schema. Need to investigate.
- **Test**: `tests/distributed/test_custom_all_reduce.py` (requires >= 2 GPUs)

---

### Phase 2: Medium complexity (more shared headers, some ROCm-specific handling)

#### 6. Pos encoding: `rotary_embedding`

- **Files**: `csrc/pos_encoding_kernels.cu` (185 lines)
- **Ops**: `rotary_embedding` (1 op)
- **Why medium**: Small file, no direct ROCm guards (uses `cuda_compat.h` for `VLLM_LDG`). Standard type conversions. Uses `dispatch_utils.h` → need `VLLM_STABLE_DISPATCH_FLOATING_TYPES`.
- **Test**: `tests/kernels/core/test_pos_encoding.py`

#### 7. Pos encoding: `fused_qk_norm_rope`

- **Files**: `csrc/fused_qknorm_rope_kernel.cu` (436 lines)
- **Ops**: `fused_qk_norm_rope` (1 op)
- **Why medium**: Has ROCm guard for `FINAL_MASK` and `__syncwarp` shim. Uses `type_convert.cuh` (shared header). Standard conversions otherwise.
- **Test**: `tests/kernels/core/test_fused_qk_norm_rope.py`

#### 8. Norm kernels (basic)

- **Files**: `csrc/layernorm_kernels.cu` (287 lines)
- **Ops**: `rms_norm`, `fused_add_rms_norm` (2 ops)
- **Why medium**: Uses migrated `vectorization_utils.cuh`, `type_convert.cuh`, `cub_helpers.h`, `core/batch_invariant.hpp`. No direct ROCm guards. But `type_convert.cuh` and `cub_helpers.h` are shared with non-migrated code — may need `#ifdef TORCH_TARGET_VERSION` pattern.
- **Test**: `tests/kernels/core/test_layernorm.py`

#### 9. Norm kernels (quantized)

- **Files**: `csrc/layernorm_quant_kernels.cu` (282 lines)
- **Ops**: `rms_norm_static_fp8_quant`, `fused_add_rms_norm_static_fp8_quant` (2 ops)
- **Depends on**: FP8 quant migration (#2 above) — shares `common.cuh`
- **Test**: `tests/kernels/core/test_layernorm.py`

#### 10. Norm kernels (dynamic per-token quant)

- **Files**: `csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu` (270+ lines)
- **Ops**: `rms_norm_dynamic_per_token_quant`, `rms_norm_per_block_quant` (2 ops)
- **Depends on**: Norm basics (#8). Uses `layernorm_utils.cuh` and `quant_conversions.cuh` (unique to this file).
- **Test**: `tests/kernels/core/test_fused_quant_layernorm.py`

#### 11. Sampler ops

- **Files**: `csrc/sampler.cu` (728 lines), `csrc/topk.cu` (373 lines)
- **Ops**: `apply_repetition_penalties_`, `top_k_per_row_prefill`, `top_k_per_row_decode`, `large_context_topk` (4 ops)
- **Why medium**: ROCm guards for cub/hipcub includes and shared memory budgets. `topk.cu` has ROCm-specific function pointer handling. Standard type conversions otherwise.
- **Test**: `tests/kernels/test_top_k_per_row.py`, `tests/kernels/test_apply_repetition_penalties.py`

---

### Phase 3: Complex (deep header chains, extensive ROCm divergence)

#### 12. Merge attn states

- **Files**: `csrc/attention/merge_attn_states.cu` (239 lines)
- **Ops**: `merge_attn_states` (1 op)
- **Why first in phase 3**: Simpler than paged attention — no `attention_kernels.cuh` dependency. Uses `attention_dtypes.h` and `attention_utils.cuh` (shared with paged attention). Good stepping stone.
- **Test**: `tests/kernels/attention/test_merge_attn_states.py`

#### 13. Paged attention

- **Files**: `csrc/attention/paged_attention_v1.cu` (187 lines), `csrc/attention/paged_attention_v2.cu` (197 lines)
- **Ops**: `paged_attention_v1`, `paged_attention_v2` (2 ops)
- **Why complex**: Deep dependency on `attention_kernels.cuh` which pulls in dtype headers, FP8 headers (with ROCm guards), and `cuda_compat.h`. The `.cuh` is a shared header — needs `TensorType` pattern or to be moved wholesale.
- **Depends on**: Merge attn states (#12) for shared attention headers
- **Test**: `tests/kernels/attention/test_attention.py`

#### 14. Cache kernels

- **Files**: `csrc/cache_kernels.cu` (1416 lines), `csrc/cache_kernels_fused.cu` (280 lines)
- **Ops**: 12+ cache ops (swap_blocks, reshape_and_cache, concat_and_cache_mla, convert_fp8, gather_and_maybe_dequant_cache, etc.)
- **Why complex**: Largest kernel file, extensive ROCm guards (quant_utils, hip_bf16, __shfl_xor_sync,__syncwarp, fp8 conversions), many FP8 quant header dependencies. Also uses `concat_mla_q.cuh`.
- **Note**: Currently in `_C_cache_ops` sub-library — needs to move to `_C` namespace in stable target.
- **Test**: `tests/kernels/attention/test_cache.py`, `tests/kernels/test_cache_kernels.py` (note: `copy_blocks` has no dedicated test coverage)

#### 15. Mamba

- **Files**: `csrc/mamba/mamba_ssm/selective_scan_fwd.cu` (832 lines) + `selective_scan.h`, `static_switch.h`
- **Ops**: `selective_scan_fwd` (1 op)
- **Why complex**: Many ROCm guards (exception headers, cub/hipcub, cudaFuncSetAttribute vs hipFuncSetAttribute, different kernel launch configs). Complex template metaprogramming in `selective_scan.h`.
- **Test**: `tests/kernels/mamba/test_mamba_ssm.py`

---

### Blocked on torch 2.11+ (`torch::from_blob` with lambda deleter)

#### `get_cuda_view_from_cpu_tensor`

- **Files**: `csrc/cuda_view.cu` (59 lines)
- **Ops**: `get_cuda_view_from_cpu_tensor` (1 op)
- **Blocker**: Uses `torch::from_blob` with a lambda deleter, which is not available in stable ABI headers until torch 2.11.
- **Test**: No dedicated test file.

#### `weak_ref_tensor`

- **Files**: Defined inline in `csrc/ops.h` (lines 11-31)
- **Ops**: `weak_ref_tensor` (1 op)
- **Blocker**: Same — uses `torch::from_blob` and `torch::Tensor` storage APIs.
- **Test**: No dedicated test file.

---

## PR Grouping Strategy

Each PR should be self-contained and testable. Suggested grouping:

| PR | Kernels | Commits | Why grouped |
|---|---|---|---|
| PR A | INT8 quant (#1) | 2 (move + migrate) | Self-contained, single file |
| PR B | FP8 quant (#2) | 2 | Self-contained, same pattern as INT8 |
| PR C | GPTQ (#3) | 2 | Self-contained, isolated sub-headers |
| PR D | GGML (#4) | 2 | Self-contained, isolated sub-headers |
| PR E | Custom all reduce (#5) | 2 | Self-contained, needs `fptr_t`/vector investigation |
| PR F | Pos encoding (#6 + #7) | 4 (2 moves + 2 migrates) | Same test file, related functionality |
| PR G | Norm basic + quant (#8 + #9) | 4 | Same test file, shared `type_convert.cuh` |
| PR H | Norm dynamic quant (#10) | 2 | Depends on PR G headers |
| PR I | Sampler (#11) | 4 (2 moves + 2 migrates) | sampler.cu + topk.cu, same dispatch pattern |
| PR J | Merge attn states (#12) | 2 | Stepping stone for paged attention |
| PR K | Paged attention (#13) | 4 | Two files, shared attention headers with PR J |
| PR L | Cache (#14) | 4 | Two files, depends on attention header patterns |
| PR M | Mamba (#15) | 2 | Self-contained |

`get_cuda_view_from_cpu_tensor` and `weak_ref_tensor` are blocked on torch 2.11+ (`torch::from_blob` with lambda deleter).

## Per-Migration Checklist (2 commits each)

1. **Move commit**: `git mv csrc/<path> csrc/libtorch_stable/<path>` — zero code changes
2. **Migration commit**:
   - Convert includes (see conversion table in rebase_plan.md)
   - Convert `torch::Tensor` → `torch::stable::Tensor` in function signatures
   - Convert dispatch macros (`VLLM_DISPATCH_*` → `VLLM_STABLE_DISPATCH_*`)
   - Convert CUDA context APIs (`at::cuda::*` → `get_current_cuda_stream()` etc.)
   - Convert `data_ptr<T>()` → `mutable_data_ptr<T>()` where needed
   - Convert `TORCH_CHECK` → `STD_TORCH_CHECK`
   - Add declarations to `csrc/libtorch_stable/ops.h` (outside `#ifndef USE_ROCM` for shared ops)
   - Add `ops.def` + `ops.impl` to `csrc/libtorch_stable/torch_bindings.cpp`
   - Remove declarations from `csrc/ops.h`
   - Remove registrations from `csrc/torch_bindings.cpp`
   - Update `CMakeLists.txt`: move from `VLLM_EXT_SRC` → `VLLM_STABLE_EXT_SRC`
   - For shared headers used by both targets: add `#ifdef TORCH_TARGET_VERSION` / `TensorType` pattern
3. **Build**: `uv pip install --no-build-isolation --verbose -U vllm -e . --excludes excludes.txt`
4. **Test**: `python -m pytest <test_file> -v`

## Migration Best Practices

### Golden rule

Changes must be **strictly limited** to what is necessary for the stable ABI migration. No formatting changes, no code improvements, no refactoring, no adding or removing `const`, no comment edits. The migrated code must behave identically to the original. **If unsure about any conversion, ask explicitly instead of making a decision.**

### Source of truth for conversions

The conversion tables in `rebase_plan.md` capture known patterns, but the **actual source of truth** is the installed torch stable ABI headers:

- `torch/csrc/stable/tensor.h` — `torch::stable::Tensor` methods
- `torch/csrc/stable/ops.h` — tensor creation ops (`empty`, `new_zeros`, `empty_like`, etc.)
- `torch/csrc/stable/c/shim.h` — low-level C shim functions (stream, device, etc.)
- `torch/csrc/stable/accelerator.h` — `DeviceGuard`
- `torch/headeronly/core/ScalarType.h` — `ScalarType` enum values
- `torch/headeronly/util/shim_utils.h` — `STD_TORCH_CHECK` and related macros

### `const` qualifiers

Do **not** add or remove `const` qualifiers during migration. Match the original exactly. This applies to all variables and types, not just `DeviceGuard`. (Past mistake: `const` was added to DeviceGuard in MLA/Hadamard and removed in AWQ/AllSpark.)

### Comments and formatting

- Copy comments verbatim when moving registrations — don't truncate or paraphrase.
- Don't add explanatory comments like `// moved from _C to _C_stable_libtorch`.
- Avoid gratuitous reformatting. If clang-format reformats unchanged logic, revert the formatting to match the original. Noisy diffs obscure the actual migration changes.

### `data_ptr()` conversions

- Non-templated `.data_ptr()` (returns `void*`) exists on `torch::stable::Tensor` — no change needed. `static_cast<T*>(t.data_ptr())` stays as-is.
- Templated `.data_ptr<T>()` does **not** exist on `torch::stable::Tensor`. Use `.mutable_data_ptr<T>()` (writes) or `.const_data_ptr<T>()` (reads). Which one you choose should depend on the signature of the function that this data ptr is being passed to (whether the relevant argument is `const` or not). If the function signature is not available, use `.mutable_data_ptr<T>()`.

### Tensor creation

- Use `torch::stable::empty_like(tensor)` when possible — don't manually reconstruct `torch::stable::empty(t.sizes(), t.scalar_type(), std::nullopt, t.device())`.
- `torch::stable::empty` accepts scalars for size — `IntHeaderOnlyArrayRef` has an implicit single-element constructor, so `torch::stable::empty(workspace_size, ...)` works without wrapping in braces.

### Dispatch macros

`AT_DISPATCH_*` macros don't work with stable ABI. Use stable equivalents from `libtorch_stable/dispatch_utils.h`:

- `VLLM_STABLE_DISPATCH_FLOATING_TYPES` (Float, Half, BFloat16)
- `VLLM_STABLE_DISPATCH_FP8_TYPES` (Float8_e4m3fn; also Float8_e4m3fnuz on ROCm)
- `VLLM_STABLE_DISPATCH_HALF_TYPES` (Half, BFloat16)

Don't manually expand dispatch macros into if/else chains — add a new stable dispatch macro if one doesn't exist yet.

### Tensor type aliases

- **Never** create `using Tensor = torch::stable::Tensor;` — it conflicts with `cute::Tensor` (a template) when `using namespace cute;` is in scope. Always use fully qualified `torch::stable::Tensor`.
- In shared headers (used by both `_C` and `_C_stable_libtorch`), use `TensorType`:
  ```cpp
  #ifdef TORCH_TARGET_VERSION
  using TensorType = torch::stable::Tensor;
  #else
  using TensorType = torch::Tensor;
  #endif
  ```

### Types in shared headers

Prefer `torch::headeronly::Half`, `torch::headeronly::BFloat16`, `torch::headeronly::ScalarType`, `torch::headeronly::CppTypeToScalarType` over `c10::` equivalents. The `headeronly` namespace re-exports these from c10 and works in both stable and unstable targets.

### Op registration

- Schema defs (`ops.def(...)`) go in `csrc/libtorch_stable/torch_bindings.cpp`.
- Impl registrations (`ops.impl(...)`) can go either in `torch_bindings.cpp` or directly in the kernel source file via `STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m)`.
- Register under `_C` (the op namespace), **not** `_C_stable_libtorch` (the CMake target name).
- Preserve `#ifndef USE_ROCM` guards — every op guarded in the original must remain guarded in the stable version.

### `torch::stable::Device` supports CPU

`torch::stable::Device(torch::stable::DeviceType::CPU)` works. Use `torch::stable::to(tensor, device)` for device transfers. Don't replace with `std::vector` + `cudaMemcpy` workarounds.

# Hipify Include Path Best Practices

When adding source files to HIP-compatible extension targets (e.g. `_C_stable_libtorch`), header includes must use **same-directory relative paths** so that the hipify preprocessor can trace and convert them.

## Why

vLLM's hipify step (`cmake/hipify.py`) uses PyTorch's `hipify()` with `hipify_extra_files_only=True`. This means hipify processes:

1. The listed source files (`.cu` → `.hip`)
2. Headers those source files transitively include

Hipify resolves `#include "foo.h"` **relative to the source file's directory**. It does NOT use `-I` include paths from the build system. So if a source file at `csrc/libtorch_stable/activation_kernels.cu` includes `"libtorch_stable/torch_utils.h"` (resolved via `-I csrc/`), hipify cannot find it and the header won't be hipified — leaving CUDA-only symbols like `cuda_runtime.h`, `cudaDeviceProp`, etc. unconverted.

## Rule

For any header in the **same directory** as the source file, use a direct relative include:

```cpp
// WRONG — hipify can't trace this
#include "libtorch_stable/torch_utils.h"

// CORRECT — hipify resolves this relative to the source file
#include "torch_utils.h"
```

For headers in a **parent directory** (e.g. shared headers in `csrc/`), use `../`:

```cpp
// CORRECT — hipify traces this to csrc/cuda_compat.h and hipifies it
#include "../cuda_compat.h"
```

## Summary

| Include style | Hipified? | Why |
|---|---|---|
| `"torch_utils.h"` (same dir) | Yes | Resolved relative to source file |
| `"../cuda_compat.h"` (parent dir) | Yes | Resolved relative to source file |
| `"libtorch_stable/torch_utils.h"` (via `-I`) | **No** | Hipify doesn't use build system include paths |

### Build and verify

```bash
conda activate vllm-dev
export CUDA_HOME=/usr/local/cuda-12.9/
export PATH=${CUDA_HOME}/bin:${PATH}
export CUDA_NVCC_EXECUTABLE=${CUDA_HOME}/bin/nvcc
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/:${LD_LIBRARY_PATH}
export CC="ccache gcc" && export CXX="ccache g++" && export CCACHE_NOHASHDIR=true
uv pip install --no-build-isolation --verbose -U vllm -e . --excludes excludes.txt &> out.txt
```

Every commit must build and pass relevant tests before moving on.

## Key Risks and Mitigations

- **`fptr_t` / `std::vector` in custom all reduce**: May not work with `TORCH_BOX`. Mitigation: investigate early, may need wrapper functions.
- **Shared header conflicts**: `common.cuh`, `type_convert.cuh`, `dispatch_utils.h` used by both `_C` and `_C_stable_libtorch` consumers. Mitigation: use `#ifdef TORCH_TARGET_VERSION` / `TensorType` typedef pattern.
- **`std::optional<at::Tensor>` → `std::optional<torch::stable::Tensor>`**: Verify this works with the stable ABI boxing. Check existing FP8 quant migration for precedent.
- **cub/hipcub conditional includes**: These are internal to kernel code and don't affect the stable ABI interface — should be transparent to the migration.

## Recommended Starting Point

Start with **PR A (INT8 quant)** — it's the smallest, most self-contained, and has the clearest cherry-pick path from temp3. It validates the migration pattern for quant kernels before tackling FP8 (which has the shared `common.cuh` complication).

## Size Estimates

| # | Kernel group | Files moved | Files modified | Total |
|---|---|---|---|---|
| **Phase 1** | | | | |
| 1 | INT8 quant | 1 | 6 | **7** |
| 2 | FP8 quant | 1 | 7 (incl. `common.cuh` shared header) | **8** |
| 3 | GPTQ | 7 (`.cu` + 6 sub-headers) | 6 | **13** |
| 4 | GGML | 8 (`.cu` + 7 sub-headers) | 6 | **14** |
| 5 | Custom all reduce | 2 (`.cu` + `.cuh`) | 6 | **8** |
| **Phase 2** | | | | |
| 6 | `rotary_embedding` | 1 | 6 | **7** |
| 7 | `fused_qk_norm_rope` | 1 | 6 | **7** |
| 8 | Norm basic | 1 | 8 (incl. shared headers) | **9** |
| 9 | Norm quant | 1 | 6 | **7** |
| 10 | Norm dynamic quant | 3 (`.cu` + 2 sub-headers) | 6 | **9** |
| 11 | Sampler | 2 (`sampler.cu` + `topk.cu`) | 7 | **9** |
| **Phase 3** | | | | |
| 12 | Merge attn states | 1 | 8 (incl. shared attention headers) | **9** |
| 13 | Paged attention | 2 (`v1.cu` + `v2.cu`) | 8 (incl. `attention_kernels.cuh`) | **10** |
| 14 | Cache kernels | 3 (2 `.cu` + `concat_mla_q.cuh`) | 9 (incl. shared quant headers) | **12** |
| 15 | Mamba | 3 (`.cu` + 2 headers) | 6 | **9** |
| **Blocked** | | | | |
| — | `get_cuda_view_from_cpu_tensor` | 1 | 6 | **7** |
| — | `weak_ref_tensor` | 0 (inline) | 4 | **4** |

"Files modified" always includes 5 plumbing files (`csrc/libtorch_stable/ops.h`, `csrc/libtorch_stable/torch_bindings.cpp`, `csrc/ops.h`, `csrc/torch_bindings.cpp`, `CMakeLists.txt`) plus the kernel source file(s) and any shared headers needing `#ifdef TORCH_TARGET_VERSION` updates. Moved sub-headers (GPTQ's `qdq_*.cuh`, GGML's `vecdotq.cuh`, etc.) are internal and generally need no code changes.
