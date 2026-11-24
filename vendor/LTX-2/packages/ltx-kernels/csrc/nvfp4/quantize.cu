// Fused NVFP4 quantize / dequantize for Blackwell (SM >= 10.0).
//
// One thread owns `kBlocksPerThread` 16-element blocks: it reads them with 16-byte
// vector loads, reduces the amax in registers, rounds the block scale to E4M3, converts
// the payload with the hardware `cvt.rn.satfinite.e2m1x2.f32` path, and stores the scale
// directly at its swizzled (cuBLAS 128x4 tiled) address -- no separate `to_blocked` pass.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "nvfp4.h"

namespace ltx_nvfp4 {
namespace {

constexpr int kBlocksPerThread = 2;
constexpr int kThreads = 256;

// Offset of scale element (row, col) in the cuBLAS 128x4 tiled layout.
// padded_cols == roundup(K / 16, 4).
__device__ __forceinline__ int64_t swizzled_offset(int row, int col, int padded_cols) {
  const int n_col_blocks = padded_cols >> 2;
  const int tile = (row >> 7) * n_col_blocks + (col >> 2);
  const int r = row & 127;
  return static_cast<int64_t>(tile) * 512 + (r & 31) * 16 + (r >> 5) * 4 + (col & 3);
}

__device__ __forceinline__ float half_bits_to_float(unsigned short bits) {
  __half_raw hr;
  hr.x = bits;
  return __half2float(__half(hr));
}

__device__ __forceinline__ float e4m3_decode(__nv_fp8_storage_t s) {
  return half_bits_to_float(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3).x);
}

// E4M3 block scale for one 16-element block, given the block amax and the per-tensor
// decode scale.
//
// The two divisions are deliberate. The layout contract is `(amax / 6) / per_tensor_scale`
// as two correctly-rounded divides; folding them into `amax * (1/6) * __frcp_rn(pts)` is
// 1 ulp off, which is invisible almost everywhere but tips the E4M3 rounding the other way
// for block amaxes that land exactly on a boundary. At 2M blocks per tensor that happened
// on roughly a third of random 8160x4096 draws and broke bit-exact golden / checkpoint
// compatibility. Two divides per 16 loaded elements is free next to the loads.
__device__ __forceinline__ __nv_fp8_storage_t block_scale_e4m3(float amax, float pts) {
  const float scaled = pts > 0.0f ? fminf((amax / kE2M1Max) / pts, kE4M3Max) : 0.0f;
  return __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
}

template <typename T> struct Vec16B; // 16 bytes worth of T

template <> struct Vec16B<__nv_bfloat16> {
  static constexpr int kElems = 8;
  union U {
    uint4 raw;
    __nv_bfloat16 v[8];
  };
  __device__ static void unpack(const U &u, float *dst) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      dst[i] = __bfloat162float(u.v[i]);
  }
};

template <> struct Vec16B<__half> {
  static constexpr int kElems = 8;
  union U {
    uint4 raw;
    __half v[8];
  };
  __device__ static void unpack(const U &u, float *dst) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      dst[i] = __half2float(u.v[i]);
  }
};

template <> struct Vec16B<float> {
  static constexpr int kElems = 4;
  union U {
    uint4 raw;
    float v[4];
  };
  __device__ static void unpack(const U &u, float *dst) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      dst[i] = u.v[i];
  }
};

// Reads kBlockSize (16) values starting at `src` into `dst` using 16-byte loads.
template <typename T> __device__ __forceinline__ void load_block(const T *__restrict__ src, float *dst) {
  using V = Vec16B<T>;
  constexpr int kLoads = kBlockSize / V::kElems;
  const uint4 *p = reinterpret_cast<const uint4 *>(src);
#pragma unroll
  for (int l = 0; l < kLoads; ++l) {
    typename V::U u;
    u.raw = p[l];
    V::unpack(u, dst + l * V::kElems);
  }
}

template <typename T, bool kHiFirst>
__global__ void quantize_kernel(const T *__restrict__ x, const float *__restrict__ per_tensor_scale,
                                uint8_t *__restrict__ out, uint8_t *__restrict__ scales, int rows, int blocks_per_row,
                                int padded_scale_cols) {
  const float pts = *per_tensor_scale;

  const int64_t total_blocks = static_cast<int64_t>(rows) * blocks_per_row;
  int64_t blk = (static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x) * kBlocksPerThread;
  if (blk >= total_blocks)
    return;

#pragma unroll
  for (int b = 0; b < kBlocksPerThread; ++b, ++blk) {
    if (kBlocksPerThread > 1 && blk >= total_blocks)
      return;
    const int row = static_cast<int>(blk / blocks_per_row);
    const int col = static_cast<int>(blk - static_cast<int64_t>(row) * blocks_per_row);

    float v[kBlockSize];
    load_block<T>(x + blk * kBlockSize, v);

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kBlockSize; ++i)
      amax = fmaxf(amax, fabsf(v[i]));

    // Block scale in E4M3, expressed relative to the per-tensor decode scale.
    const __nv_fp8_storage_t sf = block_scale_e4m3(amax, pts);
    scales[swizzled_offset(row, col, padded_scale_cols)] = static_cast<uint8_t>(sf);

    const float total = pts * e4m3_decode(sf);
    const float inv = total > 0.0f ? __frcp_rn(total) : 0.0f;

    // 16 values -> 8 packed bytes -> one 8-byte store.
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float a = v[2 * j] * inv;
      const float c = v[2 * j + 1] * inv;
      // hi_first: element 2j lands in the high nibble (Core42 PTQ order).
      const float2 pair = kHiFirst ? make_float2(c, a) : make_float2(a, c);
      const uint32_t byte = __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
      if (j < 4) {
        lo |= byte << (8 * j);
      } else {
        hi |= byte << (8 * (j - 4));
      }
    }
    reinterpret_cast<uint2 *>(out)[blk] = make_uint2(lo, hi);
  }
}

// Tiled variant: one CTA owns a full 128x4 scale tile (128 rows x 64 elements), stages the
// 512 scale bytes in shared memory and flushes them with 16-byte stores. The elementwise
// variant above writes single scale bytes at 512-byte strides, which costs ~8x write
// amplification on the scale array; staging removes it and also zero-fills the padded
// rows/cols for free (no torch::zeros needed).
constexpr int kTileRows = 128;
constexpr int kTileCols = 4; // scale columns == 64 elements
// One scale column (16 elements) per thread, so a CTA covering one 128x4 tile has 512
// threads. Two columns per thread splits each thread's payload into two 8-byte stores
// that land in alternating halves of a 32-byte sector, so every store instruction writes
// partial sectors. Packed bytes are identical either way; GB200 CI goldens are the
// layout contract if they ever drift.
constexpr int kColsPerThread = 1;
constexpr int kTileThreads = kTileRows * kTileCols / kColsPerThread;

template <typename T, bool kHiFirst>
__global__ void quantize_tiled_kernel(const T *__restrict__ x, const float *__restrict__ per_tensor_scale,
                                      uint8_t *__restrict__ out, uint8_t *__restrict__ scales, int rows,
                                      int blocks_per_row, int padded_scale_cols) {
  __shared__ uint8_t tile[kTileRows * kTileCols];

  const float pts = *per_tensor_scale;

  const int tile_col = blockIdx.x * kTileCols;
  const int tile_row = blockIdx.y * kTileRows;
  const int local_row = threadIdx.x / (kTileCols / kColsPerThread);
  const int local_col = (threadIdx.x % (kTileCols / kColsPerThread)) * kColsPerThread;
  const int row = tile_row + local_row;

  // In-tile scale offsets for this thread's kColsPerThread adjacent columns.
  const int tile_off = (local_row & 31) * 16 + (local_row >> 5) * 4 + local_col;

  uint8_t sf_bytes[kColsPerThread] = {};
  if (row < rows) {
#pragma unroll
    for (int c = 0; c < kColsPerThread; ++c) {
      const int col = tile_col + local_col + c;
      if (col >= blocks_per_row)
        continue;
      const int64_t blk = static_cast<int64_t>(row) * blocks_per_row + col;

      float v[kBlockSize];
      load_block<T>(x + blk * kBlockSize, v);

      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < kBlockSize; ++i)
        amax = fmaxf(amax, fabsf(v[i]));

      const __nv_fp8_storage_t sf = block_scale_e4m3(amax, pts);
      sf_bytes[c] = static_cast<uint8_t>(sf);

      const float total = pts * e4m3_decode(sf);
      const float inv = total > 0.0f ? __frcp_rn(total) : 0.0f;

      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float a = v[2 * j] * inv;
        const float b = v[2 * j + 1] * inv;
        const float2 pair = kHiFirst ? make_float2(b, a) : make_float2(a, b);
        const uint32_t byte = __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
        if (j < 4) {
          lo |= byte << (8 * j);
        } else {
          hi |= byte << (8 * (j - 4));
        }
      }
      reinterpret_cast<uint2 *>(out)[blk] = make_uint2(lo, hi);
    }
  }

#pragma unroll
  for (int c = 0; c < kColsPerThread; ++c)
    tile[tile_off + c] = sf_bytes[c];
  __syncthreads();

  // 32 threads x 16 bytes == the whole 512-byte tile, fully coalesced.
  if (threadIdx.x < 32) {
    const int64_t tile_index = static_cast<int64_t>(blockIdx.y) * (padded_scale_cols / kTileCols) + blockIdx.x;
    uint4 *dst = reinterpret_cast<uint4 *>(scales + tile_index * (kTileRows * kTileCols));
    dst[threadIdx.x] = reinterpret_cast<const uint4 *>(tile)[threadIdx.x];
  }
}

template <typename T, bool kHiFirst>
__global__ void dequantize_kernel(const uint8_t *__restrict__ packed, const float *__restrict__ per_tensor_scale,
                                  const uint8_t *__restrict__ scales, T *__restrict__ out, int rows, int blocks_per_row,
                                  int padded_scale_cols) {
  const float pts = *per_tensor_scale;
  const int64_t total_blocks = static_cast<int64_t>(rows) * blocks_per_row;
  const int64_t blk = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (blk >= total_blocks)
    return;

  const int row = static_cast<int>(blk / blocks_per_row);
  const int col = static_cast<int>(blk - static_cast<int64_t>(row) * blocks_per_row);
  const float scale = pts * e4m3_decode(scales[swizzled_offset(row, col, padded_scale_cols)]);

  const uint2 bytes = reinterpret_cast<const uint2 *>(packed)[blk];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t word = j < 4 ? bytes.x : bytes.y;
    const uint8_t byte = static_cast<uint8_t>(word >> (8 * (j & 3)));
    const __half2_raw h2 = __nv_cvt_fp4x2_to_halfraw2(byte, __NV_E2M1);
    // halfraw2 lane .x is the low nibble.
    const float lo = half_bits_to_float(h2.x);
    const float hi = half_bits_to_float(h2.y);
    const float first = kHiFirst ? hi : lo;
    const float second = kHiFirst ? lo : hi;
    T *dst = out + blk * kBlockSize + 2 * j;
    dst[0] = static_cast<T>(first * scale);
    dst[1] = static_cast<T>(second * scale);
  }
}

__global__ void mul_scalars_kernel(const float *a, const float *b, float *out) { out[0] = a[0] * b[0]; }

// ---------------------------------------------------------------------------
// Fused |amax| / divisor -- the per-tensor decode scale in one pass.
//
// `x.float().abs().nan_to_num().amax() / divisor` in ATen is three passes over the
// activation plus five scalar launches. This is one pass with 16-byte loads.
//
// NaN/inf handling matches `nan_to_num`: NaN contributes nothing (CUDA `fmaxf` returns
// the non-NaN operand) and +-inf saturates to FLT_MAX. Dividing by a positive `divisor`
// is monotonic on non-negative reals, so we can fold it in before the atomic and skip a
// follow-up kernel.
// ---------------------------------------------------------------------------
constexpr int kAmaxThreads = 256;
constexpr int kAmaxMaxBlocks = 1024;

__device__ __forceinline__ float sanitize_abs(float v) {
  const float a = fabsf(v);
  // NaN fails both comparisons, so it survives here and is dropped by the fmaxf below.
  return a > FLT_MAX ? FLT_MAX : a;
}

template <typename T>
__global__ void amax_scale_kernel(const T *__restrict__ x, int64_t numel, float inv_divisor, float *__restrict__ out) {
  using V = Vec16B<T>;
  constexpr int kElems = V::kElems;

  float amax = 0.0f;
  const int64_t nvec = numel / kElems;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const uint4 *p = reinterpret_cast<const uint4 *>(x);

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < nvec; i += stride) {
    typename V::U u;
    u.raw = p[i];
    float vals[kElems];
    V::unpack(u, vals);
#pragma unroll
    for (int j = 0; j < kElems; ++j)
      amax = fmaxf(amax, sanitize_abs(vals[j]));
  }
  // Ragged tail (numel is a multiple of 16 for quantizable inputs, but keep this general).
  for (int64_t i = nvec * kElems + blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += stride) {
    amax = fmaxf(amax, sanitize_abs(static_cast<float>(x[i])));
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, off));
  }

  __shared__ float warp_max[kAmaxThreads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0)
    warp_max[warp] = amax;
  __syncthreads();

  if (threadIdx.x == 0) {
    float block_max = 0.0f;
#pragma unroll
    for (int w = 0; w < kAmaxThreads / 32; ++w)
      block_max = fmaxf(block_max, warp_max[w]);
    const float scaled = block_max * inv_divisor;
    // Monotonic bit trick: for non-negative floats, IEEE bit patterns order like the values.
    atomicMax(reinterpret_cast<unsigned int *>(out), __float_as_uint(scaled));
  }
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

} // namespace

void quantize_nvfp4_cuda(const torch::Tensor &x, const torch::Tensor &per_tensor_scale, torch::Tensor &out_packed,
                         torch::Tensor &out_scales, bool hi_first, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be 2-D contiguous CUDA");
  TORCH_CHECK(x.size(1) % kBlockSize == 0, "K must be a multiple of 16, got ", x.size(1));
  TORCH_CHECK(per_tensor_scale.is_cuda() && per_tensor_scale.scalar_type() == torch::kFloat32,
              "per_tensor_scale must be a CUDA fp32 scalar");
  TORCH_CHECK(per_tensor_scale.numel() == 1, "per_tensor_scale must hold one element");

  const int rows = static_cast<int>(x.size(0));
  const int blocks_per_row = static_cast<int>(x.size(1) / kBlockSize);
  const int padded_scale_cols = static_cast<int>(out_scales.size(1));
  TORCH_CHECK(padded_scale_cols % 4 == 0, "scale columns must be padded to a multiple of 4");

  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t total_blocks = static_cast<int64_t>(rows) * blocks_per_row;
  const int64_t threads_needed = ceil_div(total_blocks, kBlocksPerThread);
  const dim3 grid(static_cast<unsigned>(ceil_div(threads_needed, kThreads)));
  const dim3 tiled_grid(static_cast<unsigned>(padded_scale_cols / kTileCols),
                        static_cast<unsigned>(ceil_div(rows, kTileRows)));
  auto stream = at::cuda::getCurrentCUDAStream();

  const float *pts = per_tensor_scale.data_ptr<float>();
  uint8_t *out = out_packed.data_ptr<uint8_t>();
  uint8_t *sc = out_scales.data_ptr<uint8_t>();

#define LTX_LAUNCH_QUANT(T, HI)                                                                                        \
  (variant == 1                                                                                                        \
       ? quantize_kernel<T, HI><<<grid, kThreads, 0, stream>>>(reinterpret_cast<const T *>(x.data_ptr()), pts, out,    \
                                                               sc, rows, blocks_per_row, padded_scale_cols)            \
       : quantize_tiled_kernel<T, HI><<<tiled_grid, kTileThreads, 0, stream>>>(                                        \
             reinterpret_cast<const T *>(x.data_ptr()), pts, out, sc, rows, blocks_per_row, padded_scale_cols))

  switch (x.scalar_type()) {
  case torch::kBFloat16:
    hi_first ? LTX_LAUNCH_QUANT(__nv_bfloat16, true) : LTX_LAUNCH_QUANT(__nv_bfloat16, false);
    break;
  case torch::kHalf:
    hi_first ? LTX_LAUNCH_QUANT(__half, true) : LTX_LAUNCH_QUANT(__half, false);
    break;
  case torch::kFloat32:
    hi_first ? LTX_LAUNCH_QUANT(float, true) : LTX_LAUNCH_QUANT(float, false);
    break;
  default:
    TORCH_CHECK(false, "quantize_nvfp4 supports bf16/fp16/fp32, got ", x.scalar_type());
  }
#undef LTX_LAUNCH_QUANT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dequantize_nvfp4_cuda(const torch::Tensor &packed, const torch::Tensor &per_tensor_scale,
                           const torch::Tensor &scales, torch::Tensor &out, bool hi_first) {
  TORCH_CHECK(packed.is_cuda() && packed.dim() == 2 && packed.is_contiguous(), "packed must be 2-D contiguous CUDA");
  const int rows = static_cast<int>(packed.size(0));
  const int blocks_per_row = static_cast<int>(packed.size(1) * 2 / kBlockSize);
  const int padded_scale_cols = static_cast<int>(scales.size(1));

  const c10::cuda::CUDAGuard guard(packed.device());
  const int64_t total_blocks = static_cast<int64_t>(rows) * blocks_per_row;
  const dim3 grid(static_cast<unsigned>(ceil_div(total_blocks, kThreads)));
  auto stream = at::cuda::getCurrentCUDAStream();

#define LTX_LAUNCH_DEQUANT(T, HI)                                                                                      \
  dequantize_kernel<T, HI><<<grid, kThreads, 0, stream>>>(                                                             \
      packed.data_ptr<uint8_t>(), per_tensor_scale.data_ptr<float>(), scales.data_ptr<uint8_t>(),                      \
      reinterpret_cast<T *>(out.data_ptr()), rows, blocks_per_row, padded_scale_cols)

  switch (out.scalar_type()) {
  case torch::kBFloat16:
    hi_first ? LTX_LAUNCH_DEQUANT(__nv_bfloat16, true) : LTX_LAUNCH_DEQUANT(__nv_bfloat16, false);
    break;
  case torch::kFloat32:
    hi_first ? LTX_LAUNCH_DEQUANT(float, true) : LTX_LAUNCH_DEQUANT(float, false);
    break;
  default:
    TORCH_CHECK(false, "dequantize_nvfp4 supports bf16/fp32 out, got ", out.scalar_type());
  }
#undef LTX_LAUNCH_DEQUANT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void amax_scale_cuda(const torch::Tensor &x, torch::Tensor &out, double divisor) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "amax input must be contiguous CUDA");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32 && out.numel() == 1,
              "amax out must be a CUDA fp32 scalar");
  TORCH_CHECK(divisor > 0.0, "divisor must be positive, got ", divisor);

  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  float *out_ptr = out.data_ptr<float>();

  // atomicMax accumulates, so the target must start at zero. memsetAsync is stream-ordered
  // and graph-capturable, and beats a torch::zeros fill launch.
  C10_CUDA_CHECK(cudaMemsetAsync(out_ptr, 0, sizeof(float), stream));

  const int64_t numel = x.numel();
  if (numel == 0)
    return;

  const float inv_divisor = static_cast<float>(1.0 / divisor);

#define LTX_LAUNCH_AMAX(T)                                                                                             \
  do {                                                                                                                 \
    const int64_t nvec = numel / Vec16B<T>::kElems;                                                                    \
    const int64_t want = ceil_div(nvec > 0 ? nvec : 1, kAmaxThreads);                                                  \
    const unsigned blocks = static_cast<unsigned>(want < kAmaxMaxBlocks ? (want > 0 ? want : 1) : kAmaxMaxBlocks);     \
    amax_scale_kernel<T>                                                                                               \
        <<<blocks, kAmaxThreads, 0, stream>>>(reinterpret_cast<const T *>(x.data_ptr()), numel, inv_divisor, out_ptr); \
  } while (0)

  switch (x.scalar_type()) {
  case torch::kBFloat16:
    LTX_LAUNCH_AMAX(__nv_bfloat16);
    break;
  case torch::kHalf:
    LTX_LAUNCH_AMAX(__half);
    break;
  case torch::kFloat32:
    LTX_LAUNCH_AMAX(float);
    break;
  default:
    TORCH_CHECK(false, "amax_scale supports bf16/fp16/fp32, got ", x.scalar_type());
  }
#undef LTX_LAUNCH_AMAX
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mul_scalars_cuda(const torch::Tensor &a, const torch::Tensor &b, torch::Tensor &out) {
  const c10::cuda::CUDAGuard guard(a.device());
  mul_scalars_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                                                    out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace ltx_nvfp4
