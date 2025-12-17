// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

// Stable ABI compatible header - only include stable headers
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

// All activation functions migrated to use torch::stable::Tensor
// These are defined in activation_kernels.cu

// Gated activation functions (input: [..., 2*d] -> output: [..., d])
void silu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);
void mul_and_silu(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_tanh_and_mul(torch::stable::Tensor& out,
                       torch::stable::Tensor& input);
void fatrelu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input,
                     double threshold);
void swigluoai_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input,
                       double alpha, double limit);

// Element-wise activation functions (input: [..., d] -> output: [..., d])
void gelu_new(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_fast(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_quick(torch::stable::Tensor& out, torch::stable::Tensor& input);
