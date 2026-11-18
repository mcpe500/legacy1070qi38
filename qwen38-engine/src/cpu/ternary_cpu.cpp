#include "ternary_cpu.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "f16bits.hpp"

namespace ternary {

const Digits5 * digit_table5() {
    static const Digits5 * t = [] {
        static Digits5 tab[256];
        for (int b = 0; b < 256; ++b) {
            const int c = kPtq1Code[b];
            if (c > 242) { for (int d = 0; d < 5; ++d) tab[b].d[d] = -1; continue; }  // invalid byte marker
            int v = c;
            for (int d = 4; d >= 0; --d) { tab[b].d[d] = static_cast<int8_t>(v % 3); v /= 3; }
        }
        return tab;
    }();
    return t;
}

const Digits4 * digit_table4() {
    static const Digits4 * t = [] {
        static Digits4 tab[256];
        for (int b = 0; b < 256; ++b) {
            const int t3 = kPtq1Code[b];
            if (t3 > 242 || t3 % 3 != 0) { for (int d = 0; d < 4; ++d) tab[b].d[d] = -1; continue; }
            int v = t3 / 3;  // the tail code word is 3 * c4
            for (int d = 3; d >= 0; --d) { tab[b].d[d] = static_cast<int8_t>(v % 3); v /= 3; }
        }
        return tab;
    }();
    return t;
}

}  // namespace ternary

namespace {

inline uint16_t rd16(const uint8_t * p) { return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8); }

// One Q2_0 block (18 bytes) dotted with 64 activations: acc += d * sum (code-1) * x.
// The payload is 16 bytes; payload byte b holds elements 4b..4b+3 at bits 0-1, 2-3, 4-5, 6-7.
inline float q2_0_dot(const uint8_t * blk, const float * x) {
    const float d = f16_to_f32(rd16(blk));
    float acc = 0.0f;
    for (int j = 0; j < 64; j += 4) {
        const uint32_t w = blk[2 + j / 4];
        for (int k = 0; k < 4; ++k) {
            const int code = static_cast<int>((w >> (2 * k)) & 3u) - 1;
            acc += static_cast<float>(code) * x[j + k];
        }
    }
    return d * acc;
}

// One PTQ1_0 group (28 bytes) dotted with 128 activations.
inline float ptq1_0_dot(const uint8_t * g, const float * x) {
    const float d = f16_to_f32(rd16(g + 26));
    const ternary::Digits5 * t5 = ternary::digit_table5();
    const ternary::Digits4 * t4 = ternary::digit_table4();
    float acc = 0.0f;
    for (int j = 0; j < 16; ++j) {
        const ternary::Digits5 & dg = t5[g[j]];
        acc += (dg.d[0] - 1) * x[j] + (dg.d[1] - 1) * x[j + 16] + (dg.d[2] - 1) * x[j + 32] +
               (dg.d[3] - 1) * x[j + 48] + (dg.d[4] - 1) * x[j + 64];
    }
    for (int k = 0; k < 8; ++k) {
        const ternary::Digits5 & dg = t5[g[16 + k]];
        acc += (dg.d[0] - 1) * x[80 + k] + (dg.d[1] - 1) * x[88 + k] + (dg.d[2] - 1) * x[96 + k] +
               (dg.d[3] - 1) * x[104 + k] + (dg.d[4] - 1) * x[112 + k];
    }
    acc += (t4[g[24]].d[0] - 1) * x[120] + (t4[g[24]].d[1] - 1) * x[122] + (t4[g[24]].d[2] - 1) * x[124] +
           (t4[g[24]].d[3] - 1) * x[126];
    acc += (t4[g[25]].d[0] - 1) * x[121] + (t4[g[25]].d[1] - 1) * x[123] + (t4[g[25]].d[2] - 1) * x[125] +
           (t4[g[25]].d[3] - 1) * x[127];
    return d * acc;
}

}  // namespace

void ternary_matmul(Pool & pool, const QMat & W, const float * x, int64_t T, float * y) {
    if (!ternary::is_ternary(W.type)) throw std::runtime_error("ternary_matmul: type is not ternary");
    if (W.type == ternary::kTypeQ2_0) {
        if (W.in % 64 != 0) throw std::runtime_error("ternary_matmul: Q2_0 in not a multiple of 64");
    } else {
        if (W.in % 128 != 0) throw std::runtime_error("ternary_matmul: PTQ1_0 in not a multiple of 128");
    }
    auto work = [&](int64_t b, int64_t e) {
        for (int64_t o = b; o < e; ++o) {
            const uint8_t * row = W.p + static_cast<size_t>(o) * W.row_bytes;
            for (int64_t t = 0; t < T; ++t) {
                const float * xr = x + t * W.in;
                float acc = 0.0f;
                if (W.type == ternary::kTypeQ2_0) {
                    for (int64_t u = 0; u < W.in; u += 64) acc += q2_0_dot(row + (u / 64) * ternary::kQ2_0BlockBytes, xr + u);
                } else {
                    for (int64_t u = 0; u < W.in; u += 128) acc += ptq1_0_dot(row + (u / 128) * ternary::kPtqGroupBytes, xr + u);
                }
                y[t * W.out + o] = acc;
            }
        }
    };
    pool.parallel_for(W.out, 64, work);
}

void ternary_row_to_f32(const QMat & W, int64_t row, float * out) {
    if (!ternary::is_ternary(W.type)) throw std::runtime_error("ternary_row_to_f32: type is not ternary");
    const uint8_t * p = W.p + static_cast<size_t>(row) * W.row_bytes;
    if (W.type == ternary::kTypeQ2_0) {
        for (int64_t u = 0; u < W.in; u += 64) {
            const uint8_t * blk = p + (u / 64) * ternary::kQ2_0BlockBytes;
            const float d = f16_to_f32(rd16(blk));
            for (int j = 0; j < 64; ++j) {
                const int code = static_cast<int>((blk[2 + j / 4] >> (2 * (j % 4))) & 3u) - 1;
                out[u + j] = static_cast<float>(code) * d;
            }
        }
        return;
    }
    const ternary::Digits5 * t5 = ternary::digit_table5();
    const ternary::Digits4 * t4 = ternary::digit_table4();
    for (int64_t u = 0; u < W.in; u += 128) {
        const uint8_t * g = p + (u / 128) * ternary::kPtqGroupBytes;
        const float d = f16_to_f32(rd16(g + 26));
        for (int j = 0; j < 16; ++j) {
            const ternary::Digits5 & dg = t5[g[j]];
            out[u + j] = (dg.d[0] - 1) * d;
            out[u + j + 16] = (dg.d[1] - 1) * d;
            out[u + j + 32] = (dg.d[2] - 1) * d;
            out[u + j + 48] = (dg.d[3] - 1) * d;
            out[u + j + 64] = (dg.d[4] - 1) * d;
        }
        for (int k = 0; k < 8; ++k) {
            const ternary::Digits5 & dg = t5[g[16 + k]];
            out[u + 80 + k] = (dg.d[0] - 1) * d;
            out[u + 88 + k] = (dg.d[1] - 1) * d;
            out[u + 96 + k] = (dg.d[2] - 1) * d;
            out[u + 104 + k] = (dg.d[3] - 1) * d;
            out[u + 112 + k] = (dg.d[4] - 1) * d;
        }
        const ternary::Digits4 & da = t4[g[24]], & db = t4[g[25]];
        out[u + 120] = (da.d[0] - 1) * d; out[u + 122] = (da.d[1] - 1) * d; out[u + 124] = (da.d[2] - 1) * d; out[u + 126] = (da.d[3] - 1) * d;
        out[u + 121] = (db.d[0] - 1) * d; out[u + 123] = (db.d[1] - 1) * d; out[u + 125] = (db.d[2] - 1) * d; out[u + 127] = (db.d[3] - 1) * d;
    }
}

void hadamard_rotate(float * x, const float * signs, int64_t width, int64_t T) {
    if (width % 1024 != 0) throw std::runtime_error("hadamard_rotate: width not a multiple of 1024");
    float v[1024];  // stack scratch: no allocation in the token loop (memory-budget-24gb rule 1)
    for (int64_t t = 0; t < T; ++t) {
        float * row = x + t * width;
        for (int64_t b = 0; b < width; b += 1024) {
            for (int k = 0; k < 1024; ++k) v[k] = row[b + k] * signs[b + k];
            for (int h = 1; h < 1024; h <<= 1)
                for (int i = 0; i < 1024; i += 2 * h)
                    for (int j = i; j < i + h; ++j) { const float a = v[j], c = v[j + h]; v[j] = a + c; v[j + h] = a - c; }
            for (int k = 0; k < 1024; ++k) row[b + k] = v[k] * 0.03125f;
        }
    }
}

void hadamard_unrotate(float * x, const float * signs, int64_t width, int64_t T) {
    if (width % 1024 != 0) throw std::runtime_error("hadamard_unrotate: width not a multiple of 1024");
    float v[1024];
    for (int64_t t = 0; t < T; ++t) {
        float * row = x + t * width;
        for (int64_t b = 0; b < width; b += 1024) {
            for (int k = 0; k < 1024; ++k) v[k] = row[b + k];
            for (int h = 1; h < 1024; h <<= 1)
                for (int i = 0; i < 1024; i += 2 * h)
                    for (int j = i; j < i + h; ++j) { const float a = v[j], c = v[j + h]; v[j] = a + c; v[j + h] = a - c; }
            for (int k = 0; k < 1024; ++k) row[b + k] = v[k] * 0.03125f * signs[b + k];
        }
    }
}