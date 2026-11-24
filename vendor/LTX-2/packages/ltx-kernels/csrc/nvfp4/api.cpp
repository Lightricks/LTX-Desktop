// pybind surface for the in-house NVFP4 quantize + GEMM (see docs/NVFP4.md).

#include <torch/extension.h>

#include "nvfp4.h"

namespace ltx_nvfp4 {
namespace {

int64_t round_up(int64_t x, int64_t multiple) { return (x + multiple - 1) / multiple * multiple; }

std::tuple<int64_t, int64_t> scale_shape(int64_t rows, int64_t k) {
  return {round_up(rows, 128), round_up(k / kBlockSize, 4)};
}

std::tuple<torch::Tensor, torch::Tensor> quantize(const torch::Tensor &x, const torch::Tensor &per_tensor_scale,
                                                  bool hi_first, int64_t variant) {
  TORCH_CHECK(x.dim() == 2, "quantize_nvfp4 expects a 2-D input, got ", x.dim(), "-D");
  const int64_t rows = x.size(0);
  const int64_t k = x.size(1);
  auto [scale_rows, scale_cols] = scale_shape(rows, k);

  auto byte_opts = x.options().dtype(torch::kUInt8);
  torch::Tensor packed = torch::empty({rows, k / 2}, byte_opts);
  // The tiled kernel writes every byte of every 128x4 tile (zeros for padded rows/cols);
  // the elementwise one only writes live entries, so it needs a zeroed buffer.
  const bool needs_zero = variant == 1 && ((rows % 128 != 0) || ((k / kBlockSize) % 4 != 0));
  torch::Tensor scales = needs_zero ? torch::zeros({scale_rows, scale_cols}, byte_opts)
                                    : torch::empty({scale_rows, scale_cols}, byte_opts);

  quantize_nvfp4_cuda(x.contiguous(), per_tensor_scale, packed, scales, hi_first, variant);
  return {packed, scales};
}

torch::Tensor dequantize(const torch::Tensor &packed, const torch::Tensor &per_tensor_scale,
                         const torch::Tensor &scales, torch::ScalarType out_dtype, bool hi_first) {
  TORCH_CHECK(packed.dim() == 2, "dequantize_nvfp4 expects a 2-D input");
  torch::Tensor out = torch::empty({packed.size(0), packed.size(1) * 2}, packed.options().dtype(out_dtype));
  dequantize_nvfp4_cuda(packed, per_tensor_scale, scales, out, hi_first);
  return out;
}

torch::Tensor scaled_mm(const torch::Tensor &a, const torch::Tensor &b, const torch::Tensor &block_scale_a,
                        const torch::Tensor &block_scale_b, const torch::Tensor &alpha,
                        const std::optional<torch::Tensor> &bias, torch::ScalarType out_dtype) {
  torch::Tensor out = torch::empty({a.size(0), b.size(0)}, a.options().dtype(out_dtype));
  return scaled_mm_nvfp4_cublaslt(a, b, block_scale_a, block_scale_b, alpha, bias, out);
}

torch::Tensor mul_scalars(const torch::Tensor &a, const torch::Tensor &b) {
  torch::Tensor out = torch::empty({1}, a.options().dtype(torch::kFloat32));
  mul_scalars_cuda(a.reshape({1}).contiguous(), b.reshape({1}).contiguous(), out);
  return out;
}

torch::Tensor amax_scale(const torch::Tensor &x, double divisor) {
  torch::Tensor out = torch::empty({}, x.options().dtype(torch::kFloat32));
  amax_scale_cuda(x.contiguous(), out, divisor);
  return out;
}

} // namespace
} // namespace ltx_nvfp4

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_nvfp4", &ltx_nvfp4::quantize, py::arg("x"), py::arg("per_tensor_scale"), py::arg("hi_first") = true,
        py::arg("variant") = 2, "Fused NVFP4 quantize -> (packed uint8, E4M3 block scales in cuBLAS 128x4 layout)");
  m.def("dequantize_nvfp4", &ltx_nvfp4::dequantize, py::arg("packed"), py::arg("per_tensor_scale"), py::arg("scales"),
        py::arg("out_dtype") = torch::kBFloat16, py::arg("hi_first") = true,
        "Reference NVFP4 dequantize (tests / debugging)");
  m.def("scaled_mm_nvfp4", &ltx_nvfp4::scaled_mm, py::arg("a"), py::arg("b"), py::arg("block_scale_a"),
        py::arg("block_scale_b"), py::arg("alpha"), py::arg("bias") = std::nullopt,
        py::arg("out_dtype") = torch::kBFloat16, "alpha * (a @ b^T) + bias with NVFP4 operands (cuBLASLt)");
  m.def("mul_scalars", &ltx_nvfp4::mul_scalars, py::arg("a"), py::arg("b"),
        "Device-side product of two fp32 scalars (per-tensor scale product for the GEMM)");
  m.def("amax_scale", &ltx_nvfp4::amax_scale, py::arg("x"), py::arg("divisor") = 1.0,
        "Single-pass max(|x|) / divisor as an fp32 device scalar (NaN-suppressing)");
  m.def("gemm_alpha_mode", &ltx_nvfp4::gemm_alpha_mode, "How alpha is passed to cuBLASLt");
  m.def("set_gemm_alpha_on_device", &ltx_nvfp4::set_gemm_alpha_on_device, py::arg("on_device"),
        "Toggle device-scalar alpha (CUBLASLT_POINTER_MODE_DEVICE)");
  m.def("set_gemm_autotune", &ltx_nvfp4::set_gemm_autotune, py::arg("enabled"),
        "Toggle timing-based selection among cuBLASLt heuristic candidates");
  m.def("probe_gemm_support", &ltx_nvfp4::probe_gemm_support, py::arg("m"), py::arg("n"), py::arg("k"),
        py::arg("scale_ptr_value") = 0, "Diagnostic sweep of cuBLASLt FP4 heuristic support");
}
