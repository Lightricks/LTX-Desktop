#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cuda/rms_norm_split_rope.cuh"

// CUDA kernel template for RMS norm + split RoPE
// out_t can be at::Float8_e4m3fn or at::BFloat16

using bf16 = __nv_bfloat16;
using fp8 = __nv_fp8_e4m3;

// One thread owns kPairsPerThread rotation pairs of a single head -- both halves of
// every pair it owns. The alternative, a contiguous channel run per thread, puts a
// pair's two halves in different threads: it then needs a shuffle to reach the partner
// and reads each cos/sin cache line twice, once per half. This layout has neither, and
// measures 1.6x the channel-run layout at h=4096 (6.8 vs 4.3 TB/s on GB200).
//
// kPairsPerThread must divide d/2, so a thread never straddles two heads, and
// kPairsPerThread * threads_per_row * 2 must equal h.
constexpr int kPairsPerThread = 16;

// Frequency tables are laid out logically as [b, n, s, d/2] (what
// apply_split_rotary_emb produces -- a swapaxes view whose physical layout is
// [b, s, n, d/2]). The strides (sb, sn, ss; inner d/2 stride is 1) are forwarded from
// the host so the read is correct for both that non-contiguous view and a genuinely
// contiguous [b, n, s, d/2] tensor.
template<typename out_t, int PAIRS, int THREADS_PER_ROW, int ROWS_PER_CTA>
__global__ void _rms_norm_split_rope_kernel(
    const bf16* __restrict__ x,
    const bf16* __restrict__ sin_freqs,
    const bf16* __restrict__ cos_freqs,
    void* __restrict__ out,
    const bf16* __restrict__ weights,
    float eps,
    int b, int s, int n, int h,
    long cos_sb, long cos_sn, long cos_ss, long sin_sb, long sin_sn, long sin_ss
){
    constexpr int kWarpsPerRow = THREADS_PER_ROW / 32;
    __shared__ float smem[ROWS_PER_CTA * kWarpsPerRow];

    const int row_in_block = threadIdx.x / THREADS_PER_ROW;
    const int lane_in_row = threadIdx.x % THREADS_PER_ROW;
    const int token_idx = blockIdx.x * ROWS_PER_CTA + row_in_block;
    const bool active = token_idx < b * s;
    const int safe_token = active ? token_idx : 0;

    const int d = h / n;
    const int half = d / 2;
    const int pair0 = lane_in_row * PAIRS;          // first rotation pair of this thread
    const int head = pair0 / half;
    const int freq0 = pair0 - head * half;
    const int first_channel = head * d + freq0;
    const int second_channel = first_channel + half;

    const size_t row = static_cast<size_t>(safe_token) * static_cast<size_t>(h);
    const long freq_offset = static_cast<long>(safe_token / s) * cos_sb
        + static_cast<long>(head) * cos_sn
        + static_cast<long>(safe_token % s) * cos_ss + freq0;
    const long sin_offset = static_cast<long>(safe_token / s) * sin_sb
        + static_cast<long>(head) * sin_sn
        + static_cast<long>(safe_token % s) * sin_ss + freq0;

    bf16 first_weight[PAIRS], second_weight[PAIRS];
    ltx_kernels::load_run<PAIRS>(weights + first_channel, first_weight);
    ltx_kernels::load_run<PAIRS>(weights + second_channel, second_weight);
    float first_rot[PAIRS], second_rot[PAIRS];
    ltx_kernels::rms_norm_split_rope_pairs<PAIRS, kWarpsPerRow>(
        x + row + first_channel, x + row + second_channel, cos_freqs + freq_offset, sin_freqs + sin_offset,
        first_weight, second_weight, eps, h, lane_in_row, smem + row_in_block * kWarpsPerRow, first_rot,
        second_rot);

    // at::Float8_e4m3fn / at::BFloat16 do not convert from the native CUDA types, so
    // the staging arrays and the output pointer both use the native one.
    using store_t = std::conditional_t<std::is_same_v<out_t, at::Float8_e4m3fn>, fp8, bf16>;
    store_t first_out[PAIRS], second_out[PAIRS];
    #pragma unroll
    for (int i = 0; i < PAIRS; i++) {
        if constexpr (std::is_same_v<out_t, at::Float8_e4m3fn>) {
            first_out[i] = fp8(first_rot[i]);
            second_out[i] = fp8(second_rot[i]);
        } else {
            first_out[i] = __float2bfloat16(first_rot[i]);
            second_out[i] = __float2bfloat16(second_rot[i]);
        }
    }
    if (active) {
        constexpr int kBytes = PAIRS * sizeof(store_t);
        static_assert(kBytes % 16 == 0, "a pair run must be a whole number of 16B stores");
        auto* base = static_cast<store_t*>(out) + row;
        #pragma unroll
        for (int i = 0; i < kBytes / 16; i++) {
            reinterpret_cast<int4*>(base + first_channel)[i] = reinterpret_cast<int4*>(first_out)[i];
            reinterpret_cast<int4*>(base + second_channel)[i] = reinterpret_cast<int4*>(second_out)[i];
        }
    }
}

template<typename out_t>
void rms_norm_split_rope_cuda(
    void* x,           // Input: [b, s, h]
    void* sin_freqs,   // Sin frequencies: [b, n, s, d]
    void* cos_freqs,   // Cos frequencies: [b, n, s, d]
    void* weights,
    float eps,                  // Added to the mean square before the rsqrt
    int b,                      // Batch size
    int s,                      // Sequence length
    int n,                      // Number of heads (32)
    int h,                      // Hidden dimension (2048, 4096, or 8192)
    long cos_sb, long cos_sn, long cos_ss,  // cos_freqs strides (b, n, s)
    long sin_sb, long sin_sn, long sin_ss,  // sin_freqs strides (b, n, s)
    void* out,                // Output: [b, s, h]
    cudaStream_t stream
) {
    const int num_tokens = b * s;
    TORCH_CHECK(n > 0 && h % n == 0, "hidden dimension must divide by the head count");
    TORCH_CHECK((h / n) % (2 * kPairsPerThread) == 0,
                "head_dim/2 must be a multiple of ", kPairsPerThread);

    // One row per CTA. Sharing a CTA across rows only widens the reduction and was
    // measured slower at every hidden size.
    constexpr int kRowsPerCta = 1;
#define LAUNCH_SPLIT_ROPE(THREADS_PER_ROW)                                            \
    do {                                                                              \
        static_assert(kPairsPerThread * (THREADS_PER_ROW) * 2 == kHidden,             \
                      "threads_per_row must cover exactly half the row");             \
        _rms_norm_split_rope_kernel<out_t, kPairsPerThread, (THREADS_PER_ROW), kRowsPerCta> \
            <<<(num_tokens + kRowsPerCta - 1) / kRowsPerCta,                          \
               (THREADS_PER_ROW) * kRowsPerCta, 0, stream>>>(                         \
                reinterpret_cast<bf16*>(x),                                           \
                reinterpret_cast<bf16*>(sin_freqs),                                   \
                reinterpret_cast<bf16*>(cos_freqs),                                   \
                out,                                                                  \
                reinterpret_cast<bf16*>(weights),                                     \
                eps,                                                                  \
                b, s, n, h,                                                           \
                cos_sb, cos_sn, cos_ss,                                               \
                sin_sb, sin_sn, sin_ss);                                              \
    } while (0)

    if (h == 2048) {
        constexpr int kHidden = 2048;
        LAUNCH_SPLIT_ROPE(64);
    } else if (h == 4096) {
        constexpr int kHidden = 4096;
        LAUNCH_SPLIT_ROPE(128);
    } else {
        TORCH_CHECK(h == 8192, "Hidden dimension must be 2048, 4096, or 8192");
        constexpr int kHidden = 8192;
        LAUNCH_SPLIT_ROPE(256);
    }
#undef LAUNCH_SPLIT_ROPE

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Explicit template instantiations
template void rms_norm_split_rope_cuda<at::BFloat16>(
    void*, void*, void*, void*, float, int, int, int, int, long, long, long, long, long, long, void*, cudaStream_t
);

template void rms_norm_split_rope_cuda<at::Float8_e4m3fn>(
    void*, void*, void*, void*, float, int, int, int, int, long, long, long, long, long, long, void*, cudaStream_t
);
