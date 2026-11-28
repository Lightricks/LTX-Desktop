#pragma once

/**
 * @file rms_norm_split_rope.cuh
 * @brief Per-thread core of fused RMS-norm + split RoPE, shared by the kernels that apply it.
 *
 * A row is one token's hidden vector, ``h = heads * d``. The split RoPE rotates pair ``(i, i + d/2)``
 * of every head, and each thread owns PAIRS consecutive pairs of a single head -- both halves of each,
 * so rotating needs no exchange with another thread and every cos/sin value is read once.
 *
 * The kernels differ in how they walk rows and where the result goes (in place, fp8, or scattered
 * into all2all buffers); they share this function so their rounding is identical by construction.
 */

#include <cuda_bf16.h>

namespace ltx_kernels {

/// Loads N consecutive bf16 values as N / 8 16-byte vectors; `source` must be 16-byte aligned.
template <int N> __device__ __forceinline__ void load_run(const __nv_bfloat16 *source, __nv_bfloat16 values[N]) {
#pragma unroll
  for (int i = 0; i < N / 8; i++) {
    reinterpret_cast<int4 *>(values)[i] = reinterpret_cast<const int4 *>(source)[i];
  }
}

/**
 * @brief One thread's PAIRS rotation pairs of a row, RMS-normalized, weighted and rotated.
 *
 * Every thread of the row must call it (it contains a __syncthreads()), including threads of an
 * inactive row, which pass any valid addresses and discard the result. Returns in fp32 so the
 * caller picks the output type.
 *
 * @param x_first First-half run of this thread's pairs in the input row
 * @param x_second Second-half run (x_first + d/2)
 * @param cos_run, sin_run This thread's PAIRS frequency values
 * @param first_weight, second_weight RMS-norm weights of the two runs
 * @param eps Added to the mean square before the rsqrt
 * @param h Row length
 * @param lane_in_row This thread's index among the row's WARPS_PER_ROW * 32 threads
 * @param row_smem WARPS_PER_ROW floats of shared memory owned by this row
 * @param first_out, second_out The rotated runs
 */
template <int PAIRS, int WARPS_PER_ROW>
__device__ __forceinline__ void
rms_norm_split_rope_pairs(const __nv_bfloat16 *x_first, const __nv_bfloat16 *x_second, const __nv_bfloat16 *cos_run,
                          const __nv_bfloat16 *sin_run, const __nv_bfloat16 (&first_weight)[PAIRS],
                          const __nv_bfloat16 (&second_weight)[PAIRS], float eps, int h, int lane_in_row,
                          float *row_smem, float (&first_out)[PAIRS], float (&second_out)[PAIRS]) {
  __nv_bfloat16 first_raw[PAIRS], second_raw[PAIRS], cos_raw[PAIRS], sin_raw[PAIRS];
  load_run<PAIRS>(x_first, first_raw);
  load_run<PAIRS>(x_second, second_raw);
  load_run<PAIRS>(cos_run, cos_raw);
  load_run<PAIRS>(sin_run, sin_raw);

  float first[PAIRS], second[PAIRS], sum_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < PAIRS; i++) {
    first[i] = __bfloat162float(first_raw[i]);
    second[i] = __bfloat162float(second_raw[i]);
    sum_sq += first[i] * first[i] + second[i] * second[i];
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
  }

  // One barrier: every thread re-sums the per-warp partials itself. Electing one thread to sum them
  // and publish the result back needs a second barrier.
  if ((lane_in_row & 31) == 0) {
    row_smem[lane_in_row / 32] = sum_sq;
  }
  __syncthreads();
  float total_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < WARPS_PER_ROW; i++) {
    total_sum += row_smem[i];
  }
  const float inv_rms = rsqrtf(total_sum / h + eps);

#pragma unroll
  for (int i = 0; i < PAIRS; i++) {
    const float a = first[i] * inv_rms * __bfloat162float(first_weight[i]);
    const float c = second[i] * inv_rms * __bfloat162float(second_weight[i]);
    const float cs = __bfloat162float(cos_raw[i]);
    const float sn = __bfloat162float(sin_raw[i]);
    first_out[i] = a * cs - c * sn;
    second_out[i] = c * cs + a * sn;
  }
}

} // namespace ltx_kernels
