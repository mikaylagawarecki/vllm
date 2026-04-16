# Stable ABI Migration Plan

## Landed on main

- **[1/n]** Migrate `permute_cols` to libtorch stable ABI (#31509)
- **[2/n]** Migrate `per_token_group_quant` to torch stable ABI (#36058)

## On temp8 (pending PR)

- **[3/n]** Migrate CUTLASS `scaled_mm` + MoE to torch stable ABI (move build config + registrations + convert types)
- **[4/n]** Migrate FP4/W4A8 CUTLASS kernels to torch stable ABI

## Still to migrate (ROCm-shared — needs ROCm CI coordination)

These kernels are in the shared `VLLM_EXT_SRC` build list and compile for both CUDA and ROCm (`torch::kCUDA` maps to HIP on ROCm).

Note: Every commit below will conflict on `CMakeLists.txt`, `csrc/ops.h`, and `csrc/torch_bindings.cpp` because main has diverged significantly in those files (landed [1/n], [2/n], [3/n], [4/n], Sparse24 removal, DGX Spark, MXFP8, DSV3, MLA concat, topK decode, Mamba chunk alignment, etc.). These conflicts are mechanical — the resolution is always: remove the migrated ops from the old location, add them to `csrc/libtorch_stable/torch_bindings.cpp`, and adjust `VLLM_STABLE_EXT_SRC` in `CMakeLists.txt`. Conflicts specific to each commit's **kernel source files** are called out below.

- **Activation kernels** — `silu_and_mul`, `mul_and_silu`, `gelu_and_mul`, `gelu_tanh_and_mul`, `fatrelu_and_mul`, `gelu_new`, `gelu_fast`, `gelu_quick`, `silu_and_mul_quant`, `swigluoai_and_mul`, `persistent_masked_m_silu_mul_quant`
    - ref: temp3 `[1/n]` `996b0bb2a`
    - **Cannot cherry-pick directly.** `csrc/activation_kernels.cu` was modified by 256-bit LDG/STG (#33022), CUDA 12.9 gating (#34791), and vectorized mem ops refactor (#35105). Also `setup.py` and `vllm/platforms/cuda.py` have heavy churn. Must re-do the migration from scratch on current main, using temp3 as a guide for the conversion pattern (`torch::Tensor` → `torch::stable::Tensor`, etc.).

- **Norm kernels** — `rms_norm`, `fused_add_rms_norm`, `rms_norm_static_fp8_quant`, `fused_add_rms_norm_static_fp8_quant`, `rms_norm_dynamic_per_token_quant`, `rms_norm_per_block_quant`
    - ref: temp3 `[3/n]` `af7aaffb9`
    - **Cannot cherry-pick directly.** Norm code shares files with per_token_group_quant (already migrated as [2/n] on main): `layernorm_utils.cuh`, `quant_conversions.cuh`, `vectorization.cuh`, `vectorization_utils.cuh`, `fp8/common.cu`, `fp8/per_token_group_quant.cu` were all modified by [2/n] (`bf4cc9ed2`). Additionally `layernorm_utils.cuh` was changed by non-contiguous fused RMSNorm (#36551) and TMA-aligned scales fix (#33255). ROCm skinny_gemms also diverged heavily (#34709, #34304, #33762, etc.).

- **pos_encoding + sampler** — `rotary_embedding`, `fused_qk_norm_rope`, `apply_repetition_penalties_`, `top_k_per_row_prefill`, `top_k_per_row_decode`, `large_context_topk`
    - ref: temp3 `[4/n]` `06f7f3633` (does not cover `large_context_topk`)
    - **Cannot cherry-pick directly.** `csrc/sampler.cu` was modified by native MTP indexer support (#36982) and faster topKperRow decode (#33680). `large_context_topk` was added after temp3 branched and must be migrated fresh.

- **Non-CUTLASS w8a8 quant** — `static_scaled_fp8_quant`, `dynamic_scaled_fp8_quant`, `dynamic_per_token_scaled_fp8_quant`, `static_scaled_int8_quant`, `dynamic_scaled_int8_quant`
    - ref: temp3 `[5/n]` `d2355c809`
    - **Likely cherry-pickable with minor fixup.** Only kernel-level conflict is `per_token_group_quant_8bit.h` modified by [2/n] (`bf4cc9ed2`), but that was a clean split — this commit's changes to the int8/fp8 quant files should apply cleanly after accounting for the [2/n] migration.

- **Paged attention + merge** — `paged_attention_v1`, `paged_attention_v2`, `merge_attn_states`, `convert_vertical_slash_indexes`, `convert_vertical_slash_indexes_mergehead`
    - ref: temp3 `[6/n]` `12d21d114`
    - **Cannot cherry-pick directly.** `csrc/cache_kernels.cu` was modified by MLA 320-dim support (#36161), MLA query concat (#34917), fp8 kvcache cast fix (#33884), cp_gather_and_upconvert optimization (#35290), and indexer_k_quant_and_cache build fix (#34653). The `quant_utils.cuh` (nvidia) file also diverged.

- **Cache kernels** — all `cache_ops` (`swap_blocks`, `reshape_and_cache`, `reshape_and_cache_flash`, `concat_and_cache_mla`, `concat_and_cache_mla_rope_fused`, `convert_fp8`, `gather_and_maybe_dequant_cache`, `cp_gather_cache`, `cp_gather_and_upconvert_fp8_kv_cache`, `indexer_k_quant_and_cache`, `concat_mla_q`, `cp_gather_indexer_k_quant_cache`)
    - ref: temp3 `[7/n]` `064bf5016` (does not cover `concat_mla_q`)
    - **Cannot cherry-pick directly.** Same `cache_kernels.cu` conflicts as [6/n] above. Also `csrc/cpu/torch_bindings.cpp` diverged (#37987, #35466, #34321). `concat_mla_q` (#34917) was added after temp3 branched and must be migrated fresh. Blockwise SM120 dispatch also changed (#37970, #33517).

- **GGML kernels** — `ggml_dequantize`, `ggml_mul_mat_vec_a8`, `ggml_mul_mat_a8`, `ggml_moe_a8`, `ggml_moe_a8_vec`, `ggml_moe_get_block_size`
    - **No temp3 reference.** Must be migrated from scratch. These are straightforward `torch::Tensor` → `torch::stable::Tensor` conversions. No known conflicts beyond the standard files.

- **GPTQ kernels** — `gptq_gemm`, `gptq_shuffle`
    - **No temp3 reference.** Must be migrated from scratch. Simple ops, straightforward conversion. Registrations and build are unconditional (not `#ifndef USE_ROCM` guarded); in base `VLLM_EXT_SRC`.

- **Mamba** — `selective_scan_fwd`
    - ref: temp3 `[12/n]` `b524a5c97` (mamba portion only)
    - **Cannot cherry-pick directly.** `selective_scan_fwd.cu` and `selective_scan.h` were modified by Mamba1 chunk alignment for prefix caching (#34798) and uint32 overflow fix (#35275). No `#ifndef USE_ROCM` guard in registrations; in base `VLLM_EXT_SRC` (compiles on both CUDA and ROCm); `.cu` file has internal `#ifdef USE_ROCM` guards.

- **Custom all reduce** — `dispose`, `meta_size`, `register_buffer`, `get_graph_buffer_ipc_meta`, `allocate_shared_buffer_and_handle`, `open_mem_handle`, `free_shared_buffer`
    - ref: temp3 `[13/n]` `91d71213a`
    - **Likely cherry-pickable with minor fixup.** `csrc/custom_all_reduce.cu` and `csrc/custom_all_reduce.cuh` had no conflicting changes on main. Path rename needed. Core ops are registered unconditionally; ROCm adds additional Quick Reduce ops (`qr_all_reduce`, etc.) inside `#ifdef USE_ROCM`.

- **Misc (ROCm-shared)** — `weak_ref_tensor`, `get_cuda_view_from_cpu_tensor`
    - ref: temp3 `[2/n]` `6591295b9` for `get_cuda_view_from_cpu_tensor`
    - **Cannot cherry-pick directly.** temp3 [2/n] also included `permute_cols` (already landed as [1/n] on main). `weak_ref_tensor` is new and must be migrated fresh.

## Still to migrate (CUDA-only — no ROCm dependency)

These are guarded by `#ifndef USE_ROCM` or only built when `VLLM_GPU_LANG STREQUAL "CUDA"`.

- **~~CUTLASS MLA~~** — migrated on temp8
- **~~Hadamard~~** — migrated on temp8
- **~~AWQ kernels~~** — migrated on temp8

- **~~DSV3 fused A GEMM~~** — migrated on temp8
- **~~AllSpark kernels~~** — migrated on temp8

- **Machete kernels** — `machete_supported_schedules`, `machete_mm`, `machete_prepack_B`
    - ref: temp3 `[11/n]` `bc390b914`
    - **Likely cherry-pickable with minor fixup.** Machete kernel source files had no conflicting changes on main. Path rename needed. Note: temp3 [11/n] also included Marlin ops — those should be split out due to `Float8_e8m0fnu` blocker on `marlin_gemm`.

- **Marlin kernels** — `marlin_gemm`†, `gptq_marlin_repack`, `awq_marlin_repack`, `marlin_int4_fp8_preprocess`
    - ref: temp3 `[11/n]` `bc390b914`
    - **Cannot cherry-pick directly.** `marlin_gemm` is blocked on `Float8_e8m0fnu` (see below). Marlin kernel/template files (`kernel.h`, `marlin_template.h`) were modified by float16 NaN/Inf fix (#33972). `gptq_marlin_repack`, `awq_marlin_repack`, and `marlin_int4_fp8_preprocess` could potentially be split out and migrated independently if they don't touch `Float8_e8m0fnu`.

- **`_moe_C` extension** — MXFP8 MoE (`mxfp8_experts_quant`, `cutlass_mxfp8_grouped_mm`), Marlin MoE (`moe_wna16_marlin_gemm`†), and related ops (separate `_moe_C` library)
    - ref: temp3 `[14/n]` `a4e4834d7`
    - **Cannot cherry-pick directly.** `csrc/moe/torch_bindings.cpp` diverged with Router GEMM kernel (#37205), Cublas BF16 gate (#35121), DSV3 arch fix (#35123), DSV3 Router GEMM (#34302), and moe_permute refactor (#33449). Also `setup.py` has heavy churn. Marlin MoE kernel files (`kernel.h`, `marlin_template.h`) also modified by float16 NaN/Inf fix (#33972). `moe_wna16_marlin_gemm` is blocked on `Float8_e8m0fnu`.

## Blocked on torch 2.11+

†`marlin_gemm` and `moe_wna16_marlin_gemm` use `at::ScalarType::Float8_e8m0fnu`, which is not available in the torch stable ABI headers in torch 2.10. These ops cannot be migrated until torch 2.11+ exposes `Float8_e8m0fnu` in `torch/csrc/stable/`. No other unmigrated ops use `Float8_e8m0fnu` or `Float4_e2m1fn_x2`.

## Reference

- temp3 branch has the original full migration ([1/n]–[14/n]) but uses `csrc/stable/` paths (old convention); does not cover GGML, AWQ, AllSpark, DSV3, GPTQ, `concat_mla_q`, `large_context_topk`, or `weak_ref_tensor` (added to main after temp3 was branched)
- temp8 branch uses `csrc/libtorch_stable/` paths (current convention)
- The DGX Spark commit (`97d19197b` — `[NVIDIA] Fix DGX Spark logic (#38126)`) adds `12.1a` arch support; must be carried forward when rebasing
- All temp3 commits will conflict on `CMakeLists.txt`, `csrc/ops.h`, `csrc/torch_bindings.cpp` — these are always mechanical (remove from old location, add to stable). The notes above focus on **kernel source file** conflicts that require more careful resolution.

## Migration Best Practices

### Golden rule

Changes must be **strictly limited** to what is necessary for the stable ABI migration. No formatting changes, no code improvements, no refactoring, no adding or removing `const`, no comment edits. The migrated code must behave identically to the original. **If unsure about any conversion, ask explicitly instead of making a decision.**

### Commit structure

Two commits per migration:

1. A pure **move commit** (`git mv` only, zero code changes) moving files from `csrc/` to `csrc/libtorch_stable/`
2. The **migration commit**: type conversions, include changes, build config, and op registrations

**Do not** add `Co-authored-by: Claude` or any AI attribution trailers to commit messages.

### Source of truth for conversions

The conversion table below captures known patterns, but the **actual source of truth** is the installed torch stable ABI headers. For any conversion not listed, look up the equivalent in:

- `torch/csrc/stable/tensor.h` — available `torch::stable::Tensor` methods
- `torch/csrc/stable/ops.h` — tensor creation ops (`empty`, `new_zeros`, etc.)
- `torch/csrc/stable/c/shim.h` — low-level C shim functions (stream, device, etc.)
- `torch/csrc/stable/accelerator.h` — `DeviceGuard`
- `torch/headeronly/core/ScalarType.h` — `ScalarType` enum values
- `torch/headeronly/util/shim_utils.h` — `STD_TORCH_CHECK` and related macros

### Known type conversions

| Old (non-stable) | New (stable ABI) |
|---|---|
| `torch::Tensor` | `torch::stable::Tensor` |
| `torch::Device` | `torch::stable::Device` |
| `std::optional<torch::Tensor>` | `std::optional<torch::stable::Tensor>` |
| `torch::empty(size, options)` | `torch::stable::empty(size, ScalarType, std::nullopt, device)` |
| `torch::zeros(size, options)` | `torch::stable::new_zeros({size}, ScalarType, std::nullopt, device)` |
| `torch::TensorOptions().dtype(...).device(...)` | Pass dtype and device directly to `torch::stable::empty`/`new_zeros` |
| `tensor.dtype() == torch::kBFloat16` | `tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16` |
| `tensor.dtype() == torch::kFloat16` | `tensor.scalar_type() == torch::headeronly::ScalarType::Half` |
| `tensor.dtype() == torch::kFloat32` | `tensor.scalar_type() == torch::headeronly::ScalarType::Float` |
| `tensor.dtype() == torch::kFloat8_e4m3fn` | `tensor.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn` |
| `tensor.dtype() == torch::kInt8` | `tensor.scalar_type() == torch::headeronly::ScalarType::Char` |
| `tensor.dtype() == torch::kInt32` | `tensor.scalar_type() == torch::headeronly::ScalarType::Int` |
| `tensor.dtype() == torch::kInt64` | `tensor.scalar_type() == torch::headeronly::ScalarType::Long` |
| `tensor.dtype() == torch::kUInt8` | `tensor.scalar_type() == torch::headeronly::ScalarType::Byte` |
| `TORCH_CHECK(...)` | `STD_TORCH_CHECK(...)` |
| `C10_CUDA_CHECK(...)` | `STD_CUDA_CHECK(...)` (from `torch/csrc/stable/macros.h`) |
| `C10_CUDA_KERNEL_LAUNCH_CHECK()` | `STD_CUDA_KERNEL_LAUNCH_CHECK()` (from `torch/csrc/stable/macros.h`) |
| `TORCH_CHECK_NOT_IMPLEMENTED(...)` | `STD_TORCH_CHECK_NOT_IMPLEMENTED(...)` (defined in `torch_utils.h`) |
| `at::cuda::getCurrentCUDAStream(idx)` | `get_current_cuda_stream(idx)` (defined in `torch_utils.h`) |
| `at::cuda::OptionalCUDAGuard device_guard(device_of(t))` | `torch::stable::accelerator::DeviceGuard device_guard(t.get_device_index())` |
| `tensor.device().index()` | `tensor.get_device_index()` or `tensor.device().index()` (both work) |
| `tensor.get_device()` | `tensor.get_device_index()` |
| `tensor.sum(dim)` | `torch::stable::sum(tensor, dim)` (see note below) |

**Note on tensor methods vs stable ops:** Some `torch::Tensor` methods (e.g. `.sum()`, `.reshape()`) are not available on `torch::stable::Tensor`. Their stable equivalents are free functions in `torch/csrc/stable/ops.h` (e.g. `torch::stable::sum`, `torch::stable::reshape`). When a tensor method is missing, check `torch/csrc/stable/ops.h` for a corresponding free function.

### Include conversions

| Old | New |
|---|---|
| `#include <torch/all.h>` | `#include <torch/csrc/stable/tensor.h>` |
| `#include <ATen/cuda/CUDAContext.h>` | `#include "libtorch_stable/torch_utils.h"` |
| `#include <c10/cuda/CUDAGuard.h>` | `#include "libtorch_stable/torch_utils.h"` |
| `#include <c10/cuda/CUDAStream.h>` | `#include "libtorch_stable/torch_utils.h"` |
| `#include <c10/cuda/CUDAException.h>` | `#include <torch/csrc/stable/macros.h>` |

Additional includes as needed:

- `<torch/csrc/stable/ops.h>` — for `torch::stable::empty`, `torch::stable::new_zeros`
- `<torch/headeronly/util/shim_utils.h>` — for `STD_TORCH_CHECK`
- `<torch/headeronly/core/ScalarType.h>` — for `torch::headeronly::ScalarType::*`

### Op registration (in `csrc/libtorch_stable/torch_bindings.cpp`)

- **Op definitions**: `STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops)` with `ops.def("op_name(...) -> ...")`
- **Op implementations (CUDA)**: `STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops)` with `ops.impl("op_name", TORCH_BOX(&func))`
- **Non-tensor capability checks**: `STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, ops)` — no device to dispatch on
- **Extension registration**: `REGISTER_EXTENSION(_C_stable_libtorch)` at bottom of file

### Shared headers

When a header is included by both `_C` (non-stable) and `_C_stable_libtorch` (stable) targets, use `#ifdef TORCH_TARGET_VERSION` to conditionally typedef:

```cpp
#ifdef TORCH_TARGET_VERSION
using TensorType = torch::stable::Tensor;
#else
using TensorType = torch::Tensor;
#endif
```

### Build config (CMakeLists.txt)

- Move source files from `VLLM_EXT_SRC` to `VLLM_STABLE_EXT_SRC`
- Move arch blocks from the `_C` target section to the `_C_stable_libtorch` section
- Paths change from `csrc/...` to `csrc/libtorch_stable/...`

### Op declarations (in `csrc/libtorch_stable/ops.h`)

Function signatures must **exactly match** the original declarations in `csrc/ops.h`, including const-qualification, except with `torch::Tensor` replaced by `torch::stable::Tensor`.

### Verification after each commit

Every commit must build and pass relevant tests before moving on.

**Build:**

```bash
conda activate vllm-dev
export CUDA_HOME=/usr/local/cuda-12.9/
export PATH=${CUDA_HOME}/bin:${PATH}
export CUDA_NVCC_EXECUTABLE=${CUDA_HOME}/bin/nvcc
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/:${LD_LIBRARY_PATH}
export CC="ccache gcc" && export CXX="ccache g++" && export CCACHE_NOHASHDIR=true
uv pip install --no-build-isolation --verbose -U vllm -e . --excludes excludes.txt &> out.txt
```

**Tests by kernel group:**

| Kernel group | Test files | Required arch |
|---|---|---|
| **CUDA-only (migrated on temp8)** | | |
| CUTLASS scaled_mm | `tests/kernels/quantization/test_cutlass_scaled_mm.py` | Any (INT8), SM89+ (FP8), SM90+ (blockwise/group) |
| FP4/NVFP4 | `tests/kernels/quantization/test_nvfp4_scaled_mm.py`, `test_nvfp4_quant.py`, `test_silu_mul_nvfp4_quant.py` | SM100+ |
| FP4/NVFP4 qutlass | `tests/kernels/quantization/test_nvfp4_qutlass.py`, `test_mxfp4_qutlass.py` | SM100 or SM120 |
| W4A8 | `tests/kernels/quantization/test_cutlass_w4a8.py`, `test_cutlass_w4a8_moe.py` | SM90+ |
| CUTLASS MoE | `tests/kernels/moe/test_cutlass_moe.py` | SM90–SM109 |
| NVFP4 MoE | `tests/kernels/moe/test_nvfp4_moe.py` | SM100+ |
| CUTLASS MLA | `tests/kernels/attention/test_cutlass_mla_decode.py` | SM100+ |
| Hadamard | `tests/kernels/quantization/test_hadacore.py` | Any CUDA (no ROCm) |
| DSV3 fused A GEMM | No dedicated test file | Any CUDA (no ROCm) |
| AWQ | `tests/kernels/quantization/test_awq.py` | Any CUDA (no ROCm) |
| AllSpark | `tests/kernels/quantization/test_allspark_gemm.py` | SM80–SM89 only |
| **CUDA-only (not yet migrated)** | | |
| Machete | `tests/kernels/quantization/test_machete_mm.py` | SM90+ |
| Marlin | `tests/kernels/quantization/test_marlin_gemm.py` | SM75+ |
| `_moe_C` (MXFP8) | `tests/kernels/moe/test_cutlass_mxfp8_grouped_mm.py` | SM100+ |
| `_moe_C` (Marlin MoE) | `tests/kernels/moe/test_moe.py` (`test_fused_marlin_moe`) | SM75+ |
| **Shared CUDA/ROCm (not yet migrated)** | | |
| Activation | `tests/kernels/core/test_activation.py` | Any (SM70+) |
| Norm | `tests/kernels/core/test_layernorm.py` | Any (SM70+) |
| pos_encoding | `tests/kernels/core/test_pos_encoding.py` | Any (SM70+) |
| Sampler (topk) | `tests/kernels/test_top_k_per_row.py` | Any (SM70+) |
| Non-CUTLASS w8a8 FP8 | `tests/kernels/quantization/test_fp8_quant.py` | SM80+ (bf16→fp8 asserts on SM<80) |
| Non-CUTLASS w8a8 Int8 | `tests/kernels/quantization/test_int8_quant.py` | Any (SM70+) |
| Paged attention | `tests/kernels/attention/test_attention.py` | Any (SM70+); SM80+ if FP8 KV cache |
| Merge attn states | `tests/kernels/attention/test_merge_attn_states.py` | Any (SM70+) |
| Cache | `tests/kernels/attention/test_cache.py`, `tests/kernels/test_cache_kernels.py` | Any (SM70+); SM80+ if FP8 cache |
| GGML | `tests/kernels/quantization/test_ggml.py`, `test_gguf.py` | Any (SM70+; uses `__dp4a` which needs SM61+, satisfied by build floor) |
| GPTQ | `tests/kernels/quantization/test_gptq.py` | Any (SM70+) |
| Mamba | `tests/kernels/mamba/test_mamba_ssm.py` | Any (SM70+) |
| Custom all reduce | `tests/distributed/test_custom_all_reduce.py` | Any (SM70+, >=2 GPUs) |
| Misc (`weak_ref_tensor`, `get_cuda_view_from_cpu_tensor`) | No dedicated test file | N/A (not GPU kernels) |

Run with:

```bash
conda activate vllm-dev
mkdir -p test_logs
python -m pytest <test_file> -v 2>&1 | tee test_logs/<kernel_group>.log
```
