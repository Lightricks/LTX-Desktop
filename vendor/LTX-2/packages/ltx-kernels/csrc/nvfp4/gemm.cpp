// NVFP4 block-scaled GEMM on cuBLASLt (SM >= 10.0).
//
// Row-major D(M,N) = alpha * A(M,K) @ B(N,K)^T + bias, all in FP4 E2M1 with 1x16 E4M3
// block scales. cuBLASLt is column-major, so we compute the transposed problem
//     D_col(N,M) = op_T(Bw)(N,K) @ op_N(Ax)(K,M)
// which is the TN layout FP4 requires: cuBLAS "A" is our weight, cuBLAS "B" our
// activation, and D_col(N,M) with ld=N *is* row-major D(M,N).
//
// Descriptors + heuristic are cached per (M,N,K,bias,out dtype); pointers are bound per
// call. `alpha` is a device fp32 scalar (CUBLASLT_POINTER_MODE_DEVICE) so a dynamic
// activation scale costs neither a host sync nor an extra pass over D.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>

#include <cstdlib>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "nvfp4.h"

namespace ltx_nvfp4 {
namespace {

#define LTX_CUBLAS_CHECK(expr)                                                                                         \
  do {                                                                                                                 \
    cublasStatus_t status_ = (expr);                                                                                   \
    TORCH_CHECK(status_ == CUBLAS_STATUS_SUCCESS, "cuBLASLt call failed (", #expr, ") with status ",                   \
                static_cast<int>(status_));                                                                            \
  } while (0)

// 32 MiB matches torch's Lt workspace default. Override with LTX_NVFP4_WORKSPACE_MB to
// let the heuristic consider split-k algos with bigger scratch needs.
size_t workspace_bytes() {
  static const size_t bytes = [] {
    const char *env = std::getenv("LTX_NVFP4_WORKSPACE_MB");
    const long mb = env != nullptr ? std::strtol(env, nullptr, 10) : 32;
    return static_cast<size_t>(mb > 0 ? mb : 32) << 20;
  }();
  return bytes;
}

struct PlanKey {
  int64_t m, n, k;
  int out_dtype;
  bool has_bias;

  bool operator==(const PlanKey &o) const {
    return m == o.m && n == o.n && k == o.k && out_dtype == o.out_dtype && has_bias == o.has_bias;
  }
};

struct PlanKeyHash {
  size_t operator()(const PlanKey &p) const {
    size_t h = std::hash<int64_t>()(p.m);
    h = h * 1000003 ^ std::hash<int64_t>()(p.n);
    h = h * 1000003 ^ std::hash<int64_t>()(p.k);
    h = h * 1000003 ^ std::hash<int>()(p.out_dtype);
    h = h * 1000003 ^ std::hash<bool>()(p.has_bias);
    return h;
  }
};

constexpr int kMaxAlgos = 8;

struct Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr; // weight, (K, N) col-major
  cublasLtMatrixLayout_t b = nullptr; // activation, (K, M) col-major
  cublasLtMatrixLayout_t d = nullptr; // output, (N, M) col-major
  cublasLtMatmulAlgo_t algo{};
  size_t workspace = 0;
  // Remaining heuristic candidates, timed on the first eager call (cuBLASLt's own ranking
  // is not always right for these FP4 shapes; picking by measurement is worth a few %).
  cublasLtMatmulAlgo_t candidates[kMaxAlgos]{};
  int num_candidates = 0;
  bool tuned = false;
};

std::mutex g_mutex;
std::unordered_map<PlanKey, Plan, PlanKeyHash> g_plans;

// Persistent per-device scratch: [0] = beta (always 0).
torch::Tensor &beta_zero(const torch::Device &device) {
  static std::unordered_map<int, torch::Tensor> cache;
  auto it = cache.find(device.index());
  if (it == cache.end()) {
    it = cache.emplace(device.index(), torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(device)))
             .first;
  }
  return it->second;
}

torch::Tensor &workspace_for(const torch::Device &device) {
  static std::unordered_map<int, torch::Tensor> cache;
  auto it = cache.find(device.index());
  if (it == cache.end()) {
    it = cache
             .emplace(device.index(), torch::empty({static_cast<int64_t>(workspace_bytes())},
                                                   torch::TensorOptions().dtype(torch::kUInt8).device(device)))
             .first;
  }
  return it->second;
}

cudaDataType_t cuda_dtype(torch::ScalarType t) {
  switch (t) {
  case torch::kBFloat16:
    return CUDA_R_16BF;
  case torch::kHalf:
    return CUDA_R_16F;
  case torch::kFloat32:
    return CUDA_R_32F;
  default:
    TORCH_CHECK(false, "unsupported NVFP4 GEMM output dtype ", t);
  }
}

// Pointers cuBLASLt inspects at heuristic time (it derives alignment from them), rebound
// on every call.
struct Bindings {
  const void *a_scale = nullptr; // cuBLAS A == our weight
  const void *b_scale = nullptr;
  const void *bias = nullptr;
};

bool g_alpha_on_device = true;
bool g_autotune = false;

// Times every heuristic candidate once and keeps the fastest. Called at most once per
// cached plan, never during cudagraph capture.
template <typename RunFn> void tune_plan(Plan &plan, const RunFn &run, cudaStream_t stream) {
  cudaEvent_t start, stop;
  if (cudaEventCreate(&start) != cudaSuccess)
    return;
  if (cudaEventCreate(&stop) != cudaSuccess) {
    cudaEventDestroy(start);
    return;
  }
  float best_ms = -1.0f;
  int best = 0;
  for (int i = 0; i < plan.num_candidates; ++i) {
    if (run(plan.candidates[i], stream) != CUBLAS_STATUS_SUCCESS)
      continue; // warmup
    if (cudaEventRecord(start, stream) != cudaSuccess)
      break;
    bool ok = true;
    for (int rep = 0; rep < 3; ++rep) {
      ok = ok && run(plan.candidates[i], stream) == CUBLAS_STATUS_SUCCESS;
    }
    cudaEventRecord(stop, stream);
    if (cudaEventSynchronize(stop) != cudaSuccess || !ok)
      continue;
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, start, stop) != cudaSuccess)
      continue;
    if (best_ms < 0.0f || ms < best_ms) {
      best_ms = ms;
      best = i;
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  plan.algo = plan.candidates[best];
}

Plan &get_plan(const PlanKey &key, torch::ScalarType out_dtype, cudaDataType_t bias_dtype, const Bindings &binds) {
  auto it = g_plans.find(key);
  if (it != g_plans.end())
    return it->second;

  Plan plan;
  const cudaDataType_t d_type = cuda_dtype(out_dtype);
  LTX_CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  const cublasOperation_t op_t = CUBLAS_OP_T;
  const cublasOperation_t op_n = CUBLAS_OP_N;
  LTX_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
  LTX_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  const int32_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  LTX_CUBLAS_CHECK(
      cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
  LTX_CUBLAS_CHECK(
      cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));

  LTX_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &binds.a_scale,
                                                  sizeof(binds.a_scale)));
  LTX_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &binds.b_scale,
                                                  sizeof(binds.b_scale)));

  if (g_alpha_on_device) {
    const int32_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    LTX_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode,
                                                    sizeof(pointer_mode)));
  }

  if (key.has_bias) {
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    LTX_CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    LTX_CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_dtype, sizeof(bias_dtype)));
    LTX_CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &binds.bias, sizeof(binds.bias)));
  }

  LTX_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_4F_E2M1, key.k, key.n, key.k));
  LTX_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_4F_E2M1, key.k, key.m, key.k));
  LTX_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.d, d_type, key.n, key.m, key.n));

  cublasLtMatmulPreference_t pref = nullptr;
  LTX_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = workspace_bytes();
  LTX_CUBLAS_CHECK(
      cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

  cublasLtMatmulHeuristicResult_t heuristics[kMaxAlgos]{};
  int returned = 0;
  const cublasStatus_t heur_status =
      cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(), plan.op, plan.a, plan.b, plan.d, plan.d,
                                     pref, kMaxAlgos, heuristics, &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(heur_status == CUBLAS_STATUS_SUCCESS && returned > 0, "cuBLASLt found no NVFP4 GEMM algo for M=", key.m,
              " N=", key.n, " K=", key.k, " (status ", static_cast<int>(heur_status),
              "). NVFP4 needs SM >= 10.0, K % 32 == 0", " and N % 8 == 0.");
  plan.algo = heuristics[0].algo;
  plan.workspace = heuristics[0].workspaceSize;
  plan.num_candidates = returned;
  for (int i = 0; i < returned; ++i)
    plan.candidates[i] = heuristics[i].algo;

  return g_plans.emplace(key, plan).first->second;
}

} // namespace

torch::Tensor scaled_mm_nvfp4_cublaslt(const torch::Tensor &a, const torch::Tensor &b,
                                       const torch::Tensor &block_scale_a, const torch::Tensor &block_scale_b,
                                       const torch::Tensor &alpha, const std::optional<torch::Tensor> &bias,
                                       torch::Tensor &out) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "NVFP4 GEMM operands must be CUDA tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "NVFP4 GEMM operands must be 2-D");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "NVFP4 GEMM operands must be contiguous");
  TORCH_CHECK(a.size(1) == b.size(1), "packed K mismatch: ", a.size(1), " vs ", b.size(1));
  TORCH_CHECK(alpha.is_cuda() && alpha.scalar_type() == torch::kFloat32 && alpha.numel() == 1,
              "alpha must be a CUDA fp32 scalar");

  const int64_t m = a.size(0);
  const int64_t n = b.size(0);
  const int64_t k = a.size(1) * 2;
  TORCH_CHECK(k % 32 == 0, "NVFP4 GEMM needs K % 32 == 0, got K=", k);
  TORCH_CHECK(n % 8 == 0, "NVFP4 GEMM needs N % 8 == 0, got N=", n);
  TORCH_CHECK(out.size(0) == m && out.size(1) == n, "output shape mismatch");

  const c10::cuda::CUDAGuard guard(a.device());
  const bool has_bias = bias.has_value() && bias->numel() > 0;
  const cudaDataType_t bias_dtype = has_bias ? cuda_dtype(bias->scalar_type()) : CUDA_R_16BF;
  const PlanKey key{m, n, k, static_cast<int>(out.scalar_type()), has_bias};

  Bindings binds;
  binds.a_scale = block_scale_b.data_ptr(); // cuBLAS A == our weight
  binds.b_scale = block_scale_a.data_ptr();
  binds.bias = has_bias ? bias->data_ptr() : nullptr;

  std::lock_guard<std::mutex> lock(g_mutex);
  Plan &plan = get_plan(key, out.scalar_type(), bias_dtype, binds);

  LTX_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &binds.a_scale,
                                                  sizeof(binds.a_scale)));
  LTX_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &binds.b_scale,
                                                  sizeof(binds.b_scale)));
  if (has_bias) {
    LTX_CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &binds.bias, sizeof(binds.bias)));
  }

  // Host alpha fallback: cuBLASLt without device pointer mode gets alpha = 1 and the
  // per-tensor product is folded in afterwards by the caller.
  const float host_alpha = 1.0f;
  const float host_beta = 0.0f;
  const void *alpha_ptr = g_alpha_on_device ? alpha.data_ptr() : static_cast<const void *>(&host_alpha);
  const void *beta_ptr = g_alpha_on_device ? beta_zero(a.device()).data_ptr() : static_cast<const void *>(&host_beta);

  torch::Tensor &ws = workspace_for(a.device());
  auto run = [&](const cublasLtMatmulAlgo_t &algo, cudaStream_t stream) {
    return cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), plan.op, alpha_ptr,
                          b.data_ptr(), // cuBLAS A: weight, (K, N) col-major
                          plan.a,
                          a.data_ptr(), // cuBLAS B: activation, (K, M) col-major
                          plan.b, beta_ptr, out.data_ptr(), plan.d, out.data_ptr(), plan.d, &algo, ws.data_ptr(),
                          workspace_bytes(), stream);
  };

  auto stream = at::cuda::getCurrentCUDAStream();
  if (g_autotune && !plan.tuned && plan.num_candidates > 1 &&
      at::cuda::currentStreamCaptureStatusMayInitCtx() == at::cuda::CaptureStatus::None) {
    plan.tuned = true;
    tune_plan(plan, run, stream);
  }
  LTX_CUBLAS_CHECK(run(plan.algo, stream));
  return out;
}

std::string gemm_alpha_mode() { return g_alpha_on_device ? "device_scalar" : "host_scalar"; }

void set_gemm_autotune(bool enabled) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_autotune = enabled;
  g_plans.clear();
}

void set_gemm_alpha_on_device(bool on_device) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (on_device != g_alpha_on_device) {
    g_plans.clear(); // pointer mode is baked into the cached descriptors
    g_alpha_on_device = on_device;
  }
}

// Diagnostic sweep: which cuBLASLt FP4 configurations this driver/library accepts.
std::string probe_gemm_support(int64_t m, int64_t n, int64_t k, int64_t scale_ptr_value) {
  std::ostringstream report;
  const void *fake_scale = reinterpret_cast<const void *>(scale_ptr_value);
  for (int pointer_dev = 0; pointer_dev < 2; ++pointer_dev) {
    for (int with_scales = 0; with_scales < 2; ++with_scales) {
      for (int with_scale_ptr = 0; with_scale_ptr < 2; ++with_scale_ptr) {
        cublasLtMatmulDesc_t op = nullptr;
        cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
        cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
        if (with_scales) {
          const int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
          cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
          cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
        }
        if (with_scale_ptr) {
          cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &fake_scale, sizeof(fake_scale));
          cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &fake_scale, sizeof(fake_scale));
        }
        if (pointer_dev) {
          const int32_t pm = CUBLASLT_POINTER_MODE_DEVICE;
          cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm));
        }
        cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, k, n, k);
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, k, m, k);
        cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, n, m, n);

        cublasLtMatmulPreference_t pref = nullptr;
        cublasLtMatmulPreferenceCreate(&pref);
        size_t ws = workspace_bytes();
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
        cublasLtMatmulHeuristicResult_t heur{};
        int returned = 0;
        const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(), op, la, lb, ld,
                                                                 ld, pref, 1, &heur, &returned);
        report << "alpha_device=" << pointer_dev << " scale_mode=" << with_scales << " scale_ptr=" << with_scale_ptr
               << " -> status=" << static_cast<int>(st) << " algos=" << returned << "\n";
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld);
        cublasLtMatmulDescDestroy(op);
      }
    }
  }
  return report.str();
}

} // namespace ltx_nvfp4
