/**
 * Producers that write straight into the All2All head-exchange layout.
 *
 * ``send_recv_heads`` reads a finished ``[batch, tokens, heads, head_size]`` tensor and copies each
 * rank's head slice into that rank's IPC buffer. The kernels here fold that copy into the kernel that
 * produces the tensor, so the tensor is never materialized locally and its peer slice travels over
 * NVLink while the producer runs:
 *
 * - ``rope_scatter_heads``: fused RMS-norm + split RoPE (the math of ops/rms_norm_split_rope) whose
 *   output rows land in the owning rank's buffer at ``[batch, rank * tokens + token, head % hpr, :]``.
 * - ``fused_barrier``: signals every rank that this rank's writes into its buffer are complete, then
 *   waits for the same from every rank. Used after producers that are not kernels of ours (cuBLAS).
 *
 * Both complete only once every rank's writes into the local buffer are visible, as
 * ``send_recv_heads`` does.
 */

#include <algorithm>
#include <cstdio>

#include "../../include/cuda/configs.cuh"
#include "../../include/cuda/rms_norm_split_rope.cuh"
#include "../../include/cuda/utils.cuh"
#include "api.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

namespace ltx_kernels {
namespace all2all {
namespace all2all_cuda {

using bf16 = __nv_bfloat16;

// Rotation pairs per thread, as in ops/rms_norm_split_rope_cuda.cu.
constexpr int kScatterPairsPerThread = 16;
constexpr int kScatterCtasPerSm = 8;

template <int PAIRS, int THREADS_PER_ROW>
__global__ void __launch_bounds__(THREADS_PER_ROW)
    rope_scatter_heads(const bf16 *__restrict__ x, const bf16 *__restrict__ sin_freqs,
                       const bf16 *__restrict__ cos_freqs, const bf16 *__restrict__ weights, float eps, int b, int s,
                       int n, int h, long cos_sb, long cos_sn, long cos_ss, long sin_sb, long sin_sn, long sin_ss,
                       void **buffer_ptrs, int **barrier_signal_ptrs, int rank, int world_size, uint64_t timeout_cycles,
                       bool barrier) {
  constexpr int kWarpsPerRow = THREADS_PER_ROW / 32;
  constexpr int kRowElems = PAIRS * THREADS_PER_ROW * 2;
  __shared__ float smem[kWarpsPerRow];
  // The finished row, staged so it leaves as contiguous 16-byte-per-lane runs: the RoPE pair layout
  // alone would store 16-byte chunks 32 bytes apart, half-filling every sector it sends over NVLink.
  __shared__ __align__(16) bf16 row_out[kRowElems];

  const int lane_in_row = threadIdx.x;
  const int d = h / n;
  const int half = d / 2;
  const int pair0 = lane_in_row * PAIRS;
  const int head = pair0 / half;
  const int freq0 = pair0 - head * half;
  const int first_channel = head * d + freq0;
  const int second_channel = first_channel + half;
  const int hpr = n / world_size;
  const int slice = hpr * d; // elements of one row that go to one rank
  const int64_t total_tokens = int64_t(s) * world_size;

  bf16 first_weight[PAIRS], second_weight[PAIRS];
  load_run<PAIRS>(weights + first_channel, first_weight);
  load_run<PAIRS>(weights + second_channel, second_weight);

  for (int token_idx = blockIdx.x; token_idx < b * s; token_idx += gridDim.x) {
    const int batch = token_idx / s;
    const int token = token_idx % s;
    const size_t row = static_cast<size_t>(token_idx) * static_cast<size_t>(h);
    const long cos_offset = static_cast<long>(batch) * cos_sb + static_cast<long>(head) * cos_sn +
                            static_cast<long>(token) * cos_ss + freq0;
    const long sin_offset = static_cast<long>(batch) * sin_sb + static_cast<long>(head) * sin_sn +
                            static_cast<long>(token) * sin_ss + freq0;

    float first_rot[PAIRS], second_rot[PAIRS];
    rms_norm_split_rope_pairs<PAIRS, kWarpsPerRow>(x + row + first_channel, x + row + second_channel,
                                                   cos_freqs + cos_offset, sin_freqs + sin_offset, first_weight,
                                                   second_weight, eps, h, lane_in_row, smem, first_rot, second_rot);
    bf16 first_out[PAIRS], second_out[PAIRS];
#pragma unroll
    for (int i = 0; i < PAIRS; i++) {
      first_out[i] = __float2bfloat16(first_rot[i]);
      second_out[i] = __float2bfloat16(second_rot[i]);
    }
#pragma unroll
    for (int i = 0; i < PAIRS / 8; i++) {
      reinterpret_cast<int4 *>(row_out + first_channel)[i] = reinterpret_cast<int4 *>(first_out)[i];
      reinterpret_cast<int4 *>(row_out + second_channel)[i] = reinterpret_cast<int4 *>(second_out)[i];
    }
    __syncthreads();

    // Rank `target` owns heads [target * hpr, (target + 1) * hpr), i.e. row elements
    // [target * slice, (target + 1) * slice); its buffer is [batch, world_size * s, hpr, d] with this
    // rank's tokens at [rank * s, (rank + 1) * s).
    const int64_t dst_row = int64_t(batch) * total_tokens + int64_t(rank) * s + token;
    for (int chunk = lane_in_row; chunk < kRowElems / 8; chunk += THREADS_PER_ROW) {
      const int elem = chunk * 8;
      const int target = elem / slice;
      bf16 *dst = static_cast<bf16 *>(buffer_ptrs[target]) + dst_row * slice + (elem - target * slice);
      st_na_global(reinterpret_cast<int4 *>(dst), reinterpret_cast<int4 *>(row_out)[chunk]);
    }
    // row_out and smem are rewritten by the next row.
    __syncthreads();
  }

  // Every block wrote to every rank, so completion is counted across blocks before anyone signals.
  // Without the barrier a later fused_barrier on the stream covers these writes.
  if (barrier) {
    barrier_last_block_wait_and_reset(barrier_signal_ptrs, barrier_signal_ptrs[rank] + kBlocksDoneSlot, rank,
                                      world_size, threadIdx.x, timeout_cycles);
  }
}

// One block per target rank: the writes it covers are complete already (they came from earlier
// kernels on this stream), so each block only has to signal its rank.
__global__ void fused_barrier(int **barrier_signal_ptrs, int rank, int world_size, uint64_t timeout_cycles) {
  barrier_wait_and_reset(barrier_signal_ptrs, blockIdx.x, rank, world_size, 1, blockIdx.x, threadIdx.x, timeout_cycles);
}

void rope_scatter_heads_launch(void *x, void *sin_freqs, void *cos_freqs, void *weights, float eps, int b, int s, int n,
                               int h, long cos_sb, long cos_sn, long cos_ss, long sin_sb, long sin_sn, long sin_ss,
                               void **buffer_ptrs, int **barrier_signal_ptrs, int rank, int world_size,
                               uint64_t timeout_cycles, bool barrier, cudaStream_t stream) {
  TORCH_CHECK(n > 0 && h % n == 0 && n % world_size == 0, "heads must divide the hidden size and the world size");
  TORCH_CHECK((h / n) % (2 * kScatterPairsPerThread) == 0, "head_dim/2 must be a multiple of ", kScatterPairsPerThread);
  // Persistent CTAs walk the rows, so the completion counter sees one atomic per CTA, not per row.
  int device = 0;
  int num_sms = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  const int grid = std::min(b * s, num_sms * kScatterCtasPerSm);
#define LAUNCH_ROPE_SCATTER(THREADS_PER_ROW)                                                                           \
  rope_scatter_heads<kScatterPairsPerThread, (THREADS_PER_ROW)><<<grid, (THREADS_PER_ROW), 0, stream>>>(               \
      static_cast<bf16 *>(x), static_cast<bf16 *>(sin_freqs), static_cast<bf16 *>(cos_freqs),                          \
      static_cast<bf16 *>(weights), eps, b, s, n, h, cos_sb, cos_sn, cos_ss, sin_sb, sin_sn, sin_ss, buffer_ptrs,      \
      barrier_signal_ptrs, rank, world_size, timeout_cycles, barrier)
  if (h == 2048) {
    LAUNCH_ROPE_SCATTER(64);
  } else if (h == 4096) {
    LAUNCH_ROPE_SCATTER(128);
  } else {
    TORCH_CHECK(h == 8192, "Hidden dimension must be 2048, 4096, or 8192");
    LAUNCH_ROPE_SCATTER(256);
  }
#undef LAUNCH_ROPE_SCATTER
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_barrier_launch(int **barrier_signal_ptrs, int rank, int world_size, uint64_t timeout_cycles,
                          cudaStream_t stream) {
  fused_barrier<<<world_size, 32, 0, stream>>>(barrier_signal_ptrs, rank, world_size, timeout_cycles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace all2all_cuda
} // namespace all2all
} // namespace ltx_kernels
