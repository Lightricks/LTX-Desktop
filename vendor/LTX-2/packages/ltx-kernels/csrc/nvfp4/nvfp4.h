#pragma once

#include <torch/extension.h>

#include <optional>

namespace ltx_nvfp4 {

// FP4 E2M1 / FP8 E4M3 range constants (see docs/NVFP4.md).
constexpr float kE2M1Max = 6.0f;
constexpr float kE4M3Max = 448.0f;
constexpr int kBlockSize = 16;

// Fused NVFP4 quantize: packs `x` (2-D, bf16/fp16/fp32, contiguous) into `out_packed`
// (uint8, two E2M1 values per byte) and writes E4M3 block scales straight into the
// cuBLAS 128x4 tiled layout of `out_scales`.
//
// `per_tensor_scale` is the *decode* scale (amax / (448 * 6)) as a device fp32 scalar;
// it is read on the GPU so callers never sync.
// `variant`: 1 = one thread per 16-element block, 2 = CTA per 128x4 scale tile with the
// scale bytes staged in shared memory (default; see docs/NVFP4.md optimization log).
void quantize_nvfp4_cuda(const torch::Tensor &x, const torch::Tensor &per_tensor_scale, torch::Tensor &out_packed,
                         torch::Tensor &out_scales, bool hi_first, int64_t variant = 2);

// Reference dequantize (tests / debugging), reads the same swizzled scale layout.
void dequantize_nvfp4_cuda(const torch::Tensor &packed, const torch::Tensor &per_tensor_scale,
                           const torch::Tensor &scales, torch::Tensor &out, bool hi_first);

// alpha[0] = a[0] * b[0] on device (fp32 scalars), so the GEMM's per-tensor product
// never round-trips through the host.
void mul_scalars_cuda(const torch::Tensor &a, const torch::Tensor &b, torch::Tensor &out);

// out[0] = max(|x|) / divisor in a single pass. NaN is ignored and +-inf saturates to
// FLT_MAX, matching `x.float().abs().nan_to_num().amax()`.
void amax_scale_cuda(const torch::Tensor &x, torch::Tensor &out, double divisor);

// D = alpha * (A @ B^T) + bias, with A (M,K) / B (N,K) NVFP4-packed uint8 and
// 1x16 E4M3 block scales in cuBLAS tiled layout. `alpha` is a device fp32 scalar.
torch::Tensor scaled_mm_nvfp4_cublaslt(const torch::Tensor &a, const torch::Tensor &b,
                                       const torch::Tensor &block_scale_a, const torch::Tensor &block_scale_b,
                                       const torch::Tensor &alpha, const std::optional<torch::Tensor> &bias,
                                       torch::Tensor &out);

// Diagnostic: which alpha delivery mode the cuBLASLt build supports.
std::string gemm_alpha_mode();
void set_gemm_alpha_on_device(bool on_device);
void set_gemm_autotune(bool enabled);
std::string probe_gemm_support(int64_t m, int64_t n, int64_t k, int64_t scale_ptr_value);

} // namespace ltx_nvfp4
