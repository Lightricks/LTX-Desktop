"""Blockwise FP8/FP6 quantization ops.
Per-op modules: ``quantize`` (activation quantize/dequantize + torch weight
quantizers), ``fp6_pack`` (FP6 pack/unpack), and ``gemm`` (the FP8 GEMM).
"""
