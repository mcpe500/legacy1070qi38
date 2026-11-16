// iso3 (IsoQuant 3-bit) KV cache format, host reference. Same arithmetic as quantize_row_iso3_0_ref and
// dequantize_row_iso3_0 in PR #38 (docs/ref_pr38_iso3_mtp.diff). One block holds 128 values in 50 bytes:
//   bytes  0..1   norm  (fp16, block L2 norm divided by the L2 norm of the selected codebook values)
//   bytes  2..33  qs    (low 2 bits of each 3-bit code, 4 codes per byte)
//   bytes 34..49  signs (high bit of each code, 8 codes per byte)
// Every group of 4 values is rotated by a fixed unit quaternion (left product) before the 3-bit codebook lookup.
#pragma once
#include <cstdint>

constexpr int ISO3_QK = 128;
constexpr int ISO3_BYTES = 50;

void iso3_quantize_block_ref(const float * x, uint8_t * out);       // 128 floats -> 50 bytes
void iso3_dequantize_block_ref(const uint8_t * in, float * y);      // 50 bytes -> 128 floats
