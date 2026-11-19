// Ternary weight formats of Ternary Bonsai 2 27B (PrismML packings), shared by the CPU and the CUDA paths.
// Reverse-engineered from the PrismML release `prism-b12128` (libggml-base: quantize_row_ptq1_0_ref /
// dequantize_row_ptq1_0, ggml type id 143) and verified block by block against that quantizer, and against
// the Q2_0 decode of stock llama.cpp (type 42, fc9437a).
//
// Q2_0 (type 42), 2.25 bpw: block of 64 values = 18 bytes = fp16 scale d + 16 bytes of 2-bit codes
// (element j at bits 2j of the block payload, little-endian words; value = (code - 1) * d, codes 0..2).
//
// PTQ1_0 (type 143), 1.75 bpw: group of 128 values = 28 bytes. One fp16 scale d (little-endian at offset 26,
// the max |x| of the group) and 26 payload bytes of base-3 packed digits. Each payload byte carries a
// substitution of the base-3 code c of 5 (or 4, for the two tail bytes) ternary digits:
//   c = sum_s 3^(n-1-s) * (t_s + 1),  t_s in {-1,0,1}.
// The byte->code substitution itself is a fixed bijection [0,242] -> 256 minus the 13 bytes b == 1 (mod 19)
// (the released files never contain those); it lives in the empirical kPtq1Code table of src/cpu/ternary_cpu.cpp
// (pinned against the PrismML decoder over 409,600 real groups, zero conflicts) instead of a closed formula,
// because the release reference quantizer and the shipped files use two different roundings of it.
// The values a byte covers:
//   bytes  0..15 (stage 1): byte j  -> values {j, j+16, j+32, j+48, j+64}                (5 digits)
//   bytes 16..23 (stage 2): byte 16+k -> values {80+k, 88+k, 96+k, 104+k, 112+k}          (5 digits)
//   bytes 24..25 (tail):    byte 24 -> {120,122,124,126}, byte 25 -> {121,123,125,127}   (4 digits;
//                          their code word is 3 * c4, so c4 = code / 3 in the inverse table)
// Decoded value = (digit - 1) * d.
#pragma once
#include <cstdint>

namespace ternary {

constexpr int kTypeQ2_0 = 42;    // ggml type id
constexpr int kTypePtq1_0 = 143; // ggml type id
constexpr int kQ2_0Block = 64;   // values per Q2_0 block
constexpr int kQ2_0BlockBytes = 18;
constexpr int kPtqGroup = 128;   // values per PTQ1_0 group
constexpr int kPtqGroupBytes = 28;
constexpr int kPtqPayload = 26;  // payload bytes per group
constexpr int kStage1 = 16;      // stage-1 payload bytes
constexpr int kStage2 = 8;        // stage-2 payload bytes

inline bool is_ternary(int type) { return type == kTypeQ2_0 || type == kTypePtq1_0; }

// Byte -> base-3 code word of the PTQ1_0 payload, 255 = byte never occurs. Pinned empirically against the
// PrismML release decoder (dequantize_row_ptq1_0) over 409,600 real groups of blk.0.attn_qkv.weight: zero
// conflicts, 243 distinct code words. The unused bytes are exactly b == 1 (mod 19) and the code word of a
// byte is its rank among the used bytes, so the substitution is bijective onto [0, 242].
constexpr uint8_t kPtq1Code[256] = {
     0, 255, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
     15, 16, 17, 18, 255, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
     30, 31, 32, 33, 34, 35, 36, 37, 255, 38, 39, 40, 41, 42, 43, 44,
     45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 255, 57, 58, 59,
     60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 255,
     75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
     91, 92, 93, 255, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
     106, 107, 108, 109, 110, 111, 112, 255, 113, 114, 115, 116, 117, 118, 119, 120,
     121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 255, 131, 132, 133, 134, 135,
     136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 255, 150,
     151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
     167, 168, 255, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
     182, 183, 184, 185, 186, 255, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196,
     197, 198, 199, 200, 201, 202, 203, 204, 205, 255, 206, 207, 208, 209, 210, 211,
     212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 255, 225, 226,
     227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
};

// byte -> five base-3 digits (0..2), most significant first (built once, thread-safe).
struct Digits5 {
    int8_t d[5];
};
struct Digits4 {
    int8_t d[4];
};
const Digits5 * digit_table5();  // 256 entries; invalid bytes keep all digits at -1
const Digits4 * digit_table4();  // 256 entries

}  // namespace