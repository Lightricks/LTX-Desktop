// M5 W8A8 fragment layout and TensorOps register tiling adapted from
// Mininglamp-AI/cider (cider/kernels/w8a8_matmul.metal) via
// pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8 at cfb04c305ee52225a75014df79b2c2c5e2ee4a8d.
// MIT License
// Copyright (c) 2026 Mininglamp contributors
// Copyright (c) 2026 Pawel Mazurkiewicz
// Copyright (c) 2026 Illuminus7 (ported patches, see upstream README Credits)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Each 32-lane SIMD group computes a 32-by-32 output tile. The fragment
// register layout below is specific to the M5; validate on device before use.
template <typename OutT>
kernel void h3_gemm(
    const device int8_t *a [[buffer(0)]], const device int8_t *w [[buffer(1)]],
    device OutT *out [[buffer(2)]], const device float *row_scale [[buffer(3)]],
    const device float *weight_scale [[buffer(4)]], const device OutT *bias [[buffer(5)]],
    constant uint &m [[buffer(6)]], constant uint &n [[buffer(7)]],
    constant uint &k [[buffer(8)]], constant uint &num_scales [[buffer(9)]],
    constant uint &has_bias [[buffer(10)]],
    uint2 tile [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
    // Four SIMD groups per threadgroup, contiguous across N. Output-channel
    // tiles are whole (host checks N % 128 == 0). M is allowed a partial tile.
    const uint row = tile.y * 32;
    const uint col = tile.x * 128 + sg * 32;
    const ushort lane = ushort(lid);
    const ushort quad = lane >> 2;
    const ushort fr = (quad & 4) | ((lane >> 1) & 3);
    const ushort fc = ((quad & 2) | (lane & 1)) * 4;
    constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true,
                                               matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto left = op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto right = op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    using Left = __tensor_ops_detail::__remove_addrspace_t<decltype(left)>;
    using Right = __tensor_ops_detail::__remove_addrspace_t<decltype(right)>;
    auto acc0 = op.get_destination_cooperative_tensor<Left, Right, int32_t>();
    auto acc1 = op.get_destination_cooperative_tensor<Left, Right, int32_t>();
    for (ushort i = 0; i < 16; ++i) {
        acc0[i] = 0;
        acc1[i] = 0;
    }

    // Stored weights are [N,K]. K is divisible by 16; only the last group
    // of activation rows can be partial. Guard its reads as well as writes.
    for (uint offset = 0; offset < k; offset += 16) {
        for (ushort i = 0; i < 8; ++i) {
            uint r = fr + (i / 4) * 8;
            uint c = fc + (i % 4) + offset;
            right[i] = w[ulong(col + r) * k + c];
            right[i + 8] = w[ulong(col + 16 + r) * k + c];
            left[i] = row + r < m ? a[ulong(row + r) * k + c] : int8_t(0);
        }
        op.run(left, right, acc0);
        for (ushort i = 0; i < 8; ++i) {
            uint r = row + 16 + fr + (i / 4) * 8;
            uint c = fc + (i % 4) + offset;
            left[i] = r < m ? a[ulong(r) * k + c] : int8_t(0);
        }
        op.run(left, right, acc1);
    }

    for (ushort mm = 0; mm < 2; ++mm) {
        for (ushort nn = 0; nn < 2; ++nn) {
            for (ushort i = 0; i < 8; ++i) {
                uint r = row + mm * 16 + fr + (i / 4) * 8;
                uint c = col + nn * 16 + fc + i % 4;
                if (r < m) {
                    int product = mm == 0 ? acc0[nn * 8 + i] : acc1[nn * 8 + i];
                    float scale = row_scale[r] * weight_scale[num_scales == 1 ? 0 : c];
                    OutT value = OutT(float(product) * scale);
                    out[ulong(r) * n + c] = has_bias ? OutT(value + bias[c]) : value;
                }
            }
        }
    }
}

#define H3_VARIANT(T, NAME) template [[host_name(NAME)]] [[kernel]] decltype(h3_gemm<T>) h3_gemm<T>;
H3_VARIANT(half, "h3_fp16")
H3_VARIANT(bfloat, "h3_bf16")
H3_VARIANT(float, "h3_fp32")
