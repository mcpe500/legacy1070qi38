// Quantized matrix-vector products on the CPU. The weight stays in its GGUF block format (a pointer into the
// mmap), the activation is quantized to the type libggml-cpu expects (Q8_K or Q8_0), and the AVX2 vec_dot of
// ggml-cpu does the integer dot products (decision D1 in docs/AFK_STATUS.md). No weight is dequantized to FP32.
#pragma once
#include <cstdint>
#include <string>

#include "gguf_index.hpp"
#include "pool.hpp"
#include "ternary_spec.hpp"

// True for the Ternary Bonsai packings (Q2_0 type 42, PTQ1_0 type 143): these have no ggml-cpu vec_dot, the
// engine decodes them itself (src/cpu/ternary_cpu.cpp) and their activations are Hadamard-rotated first.
inline bool qmat_is_ternary(int type) { return ternary::is_ternary(type); }

// Row bytes of a ggml type without calling ggml_row_size (type 143 is unknown to stock ggml).
inline size_t qmat_row_bytes(int type, int64_t in) {
    if (type == ternary::kTypeQ2_0) return static_cast<size_t>(in / 64) * ternary::kQ2_0BlockBytes;
    if (type == ternary::kTypePtq1_0) return static_cast<size_t>(in / 128) * ternary::kPtqGroupBytes;
    return 0;  // caller falls back to ggml_row_size
}

struct QMat {
    const uint8_t * p = nullptr;  // out rows of row_bytes each
    int type = 0;                 // ggml_type
    int64_t in = 0, out = 0;
    size_t row_bytes = 0;
};

// Initializes the ggml CPU backend tables once (fp16 tables, iq grids). Safe to call repeatedly.
void qmat_init();

// Looks up a 2-D weight tensor "name" (ne[0] = in, ne[1] = out). Throws if missing or not 2-D.
QMat make_qmat(const GgufIndex & ix, const std::string & name);

// Bytes of activation scratch needed to quantize T rows of length `in` for weight type `wtype`.
size_t qmat_act_bytes(int wtype, int64_t in, int64_t T);

// y[t][o] = sum_i W[o][i] * x[t][i] for t in [0, T). x is [T][in] FP32, y is [T][out] FP32.
// `act` points to at least qmat_act_bytes(W.type, W.in, T) bytes of scratch (from the arena).
void qmatmul(Pool & pool, const QMat & W, const float * x, int64_t T, float * y, uint8_t * act);

// Dequantizes one row of W into out[W.in] (used for the token embedding lookup).
void qmat_row_to_f32(const QMat & W, int64_t row, float * out);
