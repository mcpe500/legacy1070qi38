// Multi-token W4A8 dot products (AVX2) — Point A of the 2026-11-16 optimization round.
// ggml's vec_dot decodes (nibble extraction + scale extraction) every weight block once per token; the MTP
// verification batch therefore repeats that decode work T times over data that is already in L1. Here each
// quantized weight block is decoded once into stack arrays and multiplied against all T activation rows.
// Integer dot products and integer scale application are exact (32-bit accumulation, ggml-style); the float
// epilogue per block (one conversion and two fma per token) matches ggml's AVX2 vec_dot shape, so the float
// rounding order differs only through the T-token batching (accepted: parity bar for this path is
// cosine >= 0.995 plus greedy token checks).
// Block layouts and scale extraction mirror src/cuda/dp4a_gemv.cu, which gemv_parity verifies against libggml.
// Activation rows use the libggml layout: T rows of `arow` bytes (block_q8_K = 292 B: float d, int8 qs[256],
// int16 bsums[16]; block_q8_0 = 34 B: fp16 d, int8 qs[32]).
// Kernels are specialized on the batch size TT in {2,3,4} so per-token accumulators live in registers;
// other T values take the generic (stack-array) version, correct but slower.
#include "mt_dot.hpp"

#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define MT_DOT_X86 1
#endif

#include "f16bits.hpp"

namespace {

#ifdef MT_DOT_X86

constexpr int Q4K_B = 144, Q5K_B = 176, Q6K_B = 210, IQ4XS_B = 136, IQ4NL_B = 18, Q8_0_B = 34, Q8K_B = 292;
constexpr int MT_TMAX = 16;

inline uint32_t rd16(const uint8_t * p) { return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8); }
inline float half_at(const uint8_t * p) { return f16_to_f32(rd16(p)); }

__m256i g_zero;
__m128i g_lut, g_f4_128, g_zero128;

// kvalues_iq4nl (16 signed bytes) copied from the KV0..KV3 words of dp4a_gemv.cu, which gemv_parity checks
// against libggml entry by entry.
const uint8_t IQ4NL[16] = {
    static_cast<uint8_t>(0xBFAD9881u), static_cast<uint8_t>(0xBFAD9881u >> 8),
    static_cast<uint8_t>(0xBFAD9881u >> 16), static_cast<uint8_t>(0xBFAD9881u >> 24),
    static_cast<uint8_t>(0xF6EADDCFu), static_cast<uint8_t>(0xF6EADDCFu >> 8),
    static_cast<uint8_t>(0xF6EADDCFu >> 16), static_cast<uint8_t>(0xF6EADDCFu >> 24),
    static_cast<uint8_t>(0x26190D01u), static_cast<uint8_t>(0x26190D01u >> 8),
    static_cast<uint8_t>(0x26190D01u >> 16), static_cast<uint8_t>(0x26190D01u >> 24),
    static_cast<uint8_t>(0x71594535u), static_cast<uint8_t>(0x71594535u >> 8),
    static_cast<uint8_t>(0x71594535u >> 16), static_cast<uint8_t>(0x71594535u >> 24)};

void init_tables() {
    g_zero = _mm256_setzero_si256();
    g_lut = _mm_loadu_si128(reinterpret_cast<const __m128i *>(IQ4NL));
    g_f4_128 = _mm_set1_epi8(0x0F);
    g_zero128 = _mm_setzero_si128();
}

inline int hsum_epi32_128(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x1B));
    return _mm_cvtsi128_si32(v);
}

inline int hsum_epi32_256(const __m256i & v) {
    return hsum_epi32_128(_mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)));
}

inline float hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

inline float hsum_ps128(__m128 s) {
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

inline void widen_signed(__m256i b, __m256i & lo, __m256i & hi) {
    const __m256i s = _mm256_cmpgt_epi8(g_zero, b);
    lo = _mm256_unpacklo_epi8(b, s);
    hi = _mm256_unpackhi_epi8(b, s);
}

// signed byte pair-dot without widening: maddubs(|x|, sign(y,x)) = pair sums of x*y, exact (|x| <= 104, |y| <= 127,
// pair sums <= 2*104*127 = 26416 < 32767).
inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

inline void widen_signed128(__m128i b, __m128i & lo, __m128i & hi) {
    const __m128i s = _mm_cmpgt_epi8(g_zero128, b);
    lo = _mm_unpacklo_epi8(b, s);
    hi = _mm_unpackhi_epi8(b, s);
}

// K-quant scale/min extraction, branch-free, same as ggml's utmp trick: 12 bytes of scales unpack to
// 8 per-unit scales sc[0..7] followed by 8 mins mn[0..7].
inline void extract_scales_mins(const uint8_t * blk_scales, uint8_t * sc, uint8_t * mn) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    std::memcpy(utmp, blk_scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    std::memcpy(sc, utmp, 8);
    std::memcpy(mn, utmp + 2, 8);
}

// Q4_K / Q5_K: a unit's 32 values come from one 32-byte slice — low nibbles hold the first half of the unit's
// values and high nibbles the second half (K-quant convention, same as dp4a_gemv.cu and the same pairing the
// AVX2 vec_dot uses), plus a fifth bit from qh for Q5_K. Weight decode happens once per block; per token only
// the madd chain runs, and the epilogue follows ggml's shape: acc += d*isum, accm += dmin*(bsums·mins).
template <bool Q5K, int TT>
void row_q4k_t(const uint8_t * wrow, int64_t nb, const uint8_t * act, int64_t arow, int64_t T, float * ycol, int64_t ystride) {
    const int64_t nt = TT ? TT : T;
    const int64_t ntlim = nt < MT_TMAX ? nt : MT_TMAX;
    __m256 accf[TT ? TT : MT_TMAX];
    __m128 accm[TT ? TT : MT_TMAX];
    for (int64_t t = 0; t < ntlim; ++t) { accf[t] = _mm256_setzero_ps(); accm[t] = _mm_setzero_ps(); }
    const __m256i vf4 = _mm256_set1_epi8(0x0F), v0101 = _mm256_set1_epi32(0x01010101);
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = wrow + static_cast<size_t>(b) * (Q5K ? Q5K_B : Q4K_B);
        const float d = half_at(blk), dmin = half_at(blk + 2);
        uint8_t sc[8], mn[8];
        extract_scales_mins(blk + 4, sc, mn);
        const __m256i qh = Q5K ? _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk + 16)) : g_zero;
        // decode all 8 units once: weight values and per-unit scale broadcast
        __m256i wu[8], sv[8];
        for (int u = 0; u < 8; ++u) {
            const int sh = (u & 1) * 4;
            __m256i w = _mm256_and_si256(
                _mm256_srli_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                                      blk + (Q5K ? 48 : 16) + 32 * (u >> 1))),
                                  sh),
                vf4);
            if (Q5K) w = _mm256_or_si256(w, _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh, u), v0101), 4));
            wu[u] = w;
            sv[u] = _mm256_set1_epi16(static_cast<short>(sc[u]));
        }
        const __m128i minv = _mm_set_epi16(static_cast<short>(mn[7]), static_cast<short>(mn[6]), static_cast<short>(mn[5]),
                                           static_cast<short>(mn[4]), static_cast<short>(mn[3]), static_cast<short>(mn[2]),
                                           static_cast<short>(mn[1]), static_cast<short>(mn[0]));
        for (int64_t t = 0; t < ntlim; ++t) {
            const uint8_t * actblk = act + static_cast<size_t>(t) * arow + static_cast<size_t>(b) * Q8K_B;
            const float dx = *reinterpret_cast<const float *>(actblk);
            const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(actblk + 260));
            const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(minv, q8s);
            accm[t] = _mm_fmadd_ps(_mm_set1_ps(-dmin * dx), _mm_cvtepi32_ps(prod), accm[t]);
            __m256i sumi = g_zero;
            for (int u = 0; u < 8; ++u) {
                const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(actblk + 4 + 32 * u));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(sv[u], _mm256_maddubs_epi16(wu[u], x)));
            }
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * dx), _mm256_cvtepi32_ps(sumi), accf[t]);
        }
    }
    for (int64_t t = 0; t < ntlim; ++t) ycol[t * ystride] = hsum_ps(accf[t]) + hsum_ps128(accm[t]);
}

template <bool Q5K>
void row_q4k(const uint8_t * wrow, int64_t nb, const uint8_t * act, int64_t arow, int64_t T, float * ycol, int64_t ystride) {
    switch (T) {
        case 2: row_q4k_t<Q5K, 2>(wrow, nb, act, arow, T, ycol, ystride); break;
        case 3: row_q4k_t<Q5K, 3>(wrow, nb, act, arow, T, ycol, ystride); break;
        case 4: row_q4k_t<Q5K, 4>(wrow, nb, act, arow, T, ycol, ystride); break;
        default: row_q4k_t<Q5K, 0>(wrow, nb, act, arow, T, ycol, ystride); break;
    }
}

// Q6_K: 4 low bits + 2 high bits per value minus 32, per-half int8 scales, Q8_K activations.
// Same structure as ggml's AVX2 vec_dot: raw unsigned 6-bit values go through maddubs, the -32 correction is
// applied once per block as (bsums·scales)<<5, scales fold with madd. Weight decode and the scale broadcasts
// are computed once per block; per token only the madd chain and one float fma run.
template <int TT>
void row_q6k_t(const uint8_t * wrow, int64_t nb, const uint8_t * act, int64_t arow, int64_t T, float * ycol, int64_t ystride) {
    const int64_t nt = TT ? TT : T;
    const int64_t ntlim = nt < MT_TMAX ? nt : MT_TMAX;
    __m256 accf[TT ? TT : MT_TMAX];
    for (int64_t t = 0; t < ntlim; ++t) accf[t] = _mm256_setzero_ps();
    const __m256i m3 = _mm256_set1_epi8(3), m15 = _mm256_set1_epi8(15);
    static const uint8_t k_shuf[128] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = wrow + static_cast<size_t>(b) * Q6K_B;
        const float d = half_at(blk + 208);
        // decode once per block: raw unsigned 6-bit vectors and per-pair scale broadcasts
        __m256i q4raw[8], scl[8];
        for (int j = 0; j < 2; ++j) {
            const uint8_t * ql = blk + 32 * (2 * j);
            const uint8_t * qh = blk + 128 + 32 * j;
            const __m256i q4bits1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql));
            const __m256i q4bits2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 32));
            const __m256i q4bitsH = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh));
            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);
            q4raw[4 * j + 0] = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            q4raw[4 * j + 1] = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            q4raw[4 * j + 2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            q4raw[4 * j + 3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
        }
        const __m128i scales128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 192));
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales128);
        for (int k = 0; k < 8; ++k) {
            const __m128i sk = _mm_shuffle_epi8(scales128, _mm_loadu_si128(reinterpret_cast<const __m128i *>(k_shuf) + k));
            scl[k] = _mm256_cvtepi8_epi16(sk);
        }
        for (int64_t t = 0; t < ntlim; ++t) {
            const uint8_t * actblk = act + static_cast<size_t>(t) * arow + static_cast<size_t>(b) * Q8K_B;
            const float dx = *reinterpret_cast<const float *>(actblk);
            const __m256i q8sums_t = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(actblk + 260));
            const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums_t, scales16), 5);
            __m256i sumi = g_zero;
            for (int k = 0; k < 8; ++k) {
                const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(actblk + 4 + 32 * k));
                const __m256i p16 = _mm256_maddubs_epi16(q4raw[k], x);  // raw (0..63, unsigned) * signed x, exact
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scl[k], p16));
            }
            sumi = _mm256_sub_epi32(sumi, q8sclsub);
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * dx), _mm256_cvtepi32_ps(sumi), accf[t]);
        }
    }
    for (int64_t t = 0; t < ntlim; ++t) ycol[t * ystride] = hsum_ps(accf[t]);
}

void row_q6k(const uint8_t * wrow, int64_t nb, const uint8_t * act, int64_t arow, int64_t T, float * ycol, int64_t ystride) {
    switch (T) {
        case 2: row_q6k_t<2>(wrow, nb, act, arow, T, ycol, ystride); break;
        case 3: row_q6k_t<3>(wrow, nb, act, arow, T, ycol, ystride); break;
        case 4: row_q6k_t<4>(wrow, nb, act, arow, T, ycol, ystride); break;
        default: row_q6k_t<0>(wrow, nb, act, arow, T, ycol, ystride); break;
    }
}

// IQ4_XS: 16-entry signed LUT over 4-bit codes (first half of a unit from low nibbles, second half from high
// nibbles), per-unit integer scale (ls-32), Q8_K activations. LUT decode happens once per block; per token the
// signed byte pair-dot runs through the sign-epi8 trick (no widening) plus one madd and one float fma.
template <int TT>
void row_iq4xs_t(const uint8_t * wrow, int64_t nb, const uint8_t * act, int64_t arow, int64_t T, float * ycol, int64_t ystride) {
    const int64_t nt = TT ? TT : T;
    const int64_t ntlim = nt < MT_TMAX ? nt : MT_TMAX;
    __m256 accf[TT ? TT : MT_TMAX];
    for (int64_t t = 0; t < ntlim; ++t) accf[t] = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = wrow + static_cast<size_t>(b) * IQ4XS_B;
        const float d = half_at(blk);
        const uint32_t sh = rd16(blk + 2);
        __m256i wu[8], sv[8];
        for (int u = 0; u < 8; ++u) {
            const int ls = ((blk[4 + (u >> 1)] >> (4 * (u & 1))) & 0xF) | (((sh >> (2 * u)) & 3) << 4);
            const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 8 + 16 * u));
            const __m128i lo = _mm_and_si128(raw, g_f4_128);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), g_f4_128);
            wu[u] = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(g_lut, lo)),
                                            _mm_shuffle_epi8(g_lut, hi), 1);
            sv[u] = _mm256_set1_epi16(static_cast<short>(ls - 32));
        }
        for (int64_t t = 0; t < ntlim; ++t) {
            const uint8_t * actblk = act + static_cast<size_t>(t) * arow + static_cast<size_t>(b) * Q8K_B;
            const float dx = *reinterpret_cast<const float *>(actblk);
            __m256i sumi = g_zero;
            for (int u = 0; u < 8; ++u) {
                const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(actblk + 4 + 32 * u));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(sv[u], mul_add_epi8(wu[u], x)));
            }
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * dx), _mm256_cvtepi32_ps(sumi), accf[t]);
        }
    }
    for (int64_t t = 0; t < ntlim; ++t) ycol[t * ystride] = hsum_ps(accf[t]);
}

void row_iq4xs(const uint8_t * wrow, int64_t nb, const uint8_t * act, int64_t arow, int64_t T, float * ycol, int64_t ystride) {
    switch (T) {
        case 2: row_iq4xs_t<2>(wrow, nb, act, arow, T, ycol, ystride); break;
        case 3: row_iq4xs_t<3>(wrow, nb, act, arow, T, ycol, ystride); break;
        case 4: row_iq4xs_t<4>(wrow, nb, act, arow, T, ycol, ystride); break;
        default: row_iq4xs_t<0>(wrow, nb, act, arow, T, ycol, ystride); break;
    }
}

// IQ4_NL and Q8_0: one 32-value weight block with a symmetric fp16 scale, Q8_0 activations (fp16 d per block).
// LUT: first 16 values from low nibbles of the 16 code bytes, second 16 from high nibbles. Q8_0: 32 raw int8.
template <bool LUT, int TT>
void row_sym32_t(const uint8_t * wrow, int64_t nb, const uint8_t * act, int64_t arow, int64_t T, float * ycol, int64_t ystride) {
    const size_t WB = LUT ? IQ4NL_B : Q8_0_B;
    const int64_t nt = TT ? TT : T;
    const int64_t ntlim = nt < MT_TMAX ? nt : MT_TMAX;
    float acc[TT ? TT : MT_TMAX];
    for (int64_t t = 0; t < ntlim; ++t) acc[t] = 0.0f;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = wrow + static_cast<size_t>(b) * WB;
        const float dw = half_at(blk);
        __m256i w;
        if (LUT) {
            const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
            const __m128i lo = _mm_and_si128(raw, g_f4_128);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), g_f4_128);
            w = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(g_lut, lo)),
                                        _mm_shuffle_epi8(g_lut, hi), 1);
        } else {
            w = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2))),
                                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 18)), 1);
        }
        __m256i wlo16, whi16;
        widen_signed(w, wlo16, whi16);
        for (int64_t t = 0; t < ntlim; ++t) {
            const uint8_t * actb = act + static_cast<size_t>(t) * arow + static_cast<size_t>(b) * Q8_0_B;
            const float s = dw * half_at(actb);
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(actb + 2));
            __m256i xlo16, xhi16;
            widen_signed(x, xlo16, xhi16);
            const __m256i p32 = _mm256_add_epi32(_mm256_madd_epi16(wlo16, xlo16), _mm256_madd_epi16(whi16, xhi16));
            acc[t] += s * static_cast<float>(hsum_epi32_256(p32));
        }
    }
    for (int64_t t = 0; t < ntlim; ++t) ycol[t * ystride] = acc[t];
}

template <bool LUT>
void row_sym32(const uint8_t * wrow, int64_t nb, const uint8_t * act, int64_t arow, int64_t T, float * ycol, int64_t ystride) {
    switch (T) {
        case 2: row_sym32_t<LUT, 2>(wrow, nb, act, arow, T, ycol, ystride); break;
        case 3: row_sym32_t<LUT, 3>(wrow, nb, act, arow, T, ycol, ystride); break;
        case 4: row_sym32_t<LUT, 4>(wrow, nb, act, arow, T, ycol, ystride); break;
        default: row_sym32_t<LUT, 0>(wrow, nb, act, arow, T, ycol, ystride); break;
    }
}

#else

void init_tables() {}

#endif

}  // namespace

bool mt_dot_supported(int wtype) {
#if defined(MT_DOT_X86) && (defined(__AVX2__) || defined(__znver1__))
    switch (wtype) {
        case 12:   // Q4_K
        case 13:   // Q5_K
        case 14:   // Q6_K
        case 23:   // IQ4_XS
        case 20:   // IQ4_NL
        case 8:    // Q8_0
            return true;
        default:
            return false;
    }
#else
    (void) wtype;
    return false;
#endif
}

bool mt_dot_use(int wtype, int64_t T) {
    static const bool env_on = [] {
        const char * e = std::getenv("QWEN_MT_DOT");
        return !(e && e[0] == '0');
    }();
    return env_on && T >= 2 && T <= MT_TMAX && mt_dot_supported(wtype);
}

void mt_dot(Pool & pool, const QMat & W, const uint8_t * act, int64_t arow, int64_t T, float * y, int64_t grain) {
#if defined(MT_DOT_X86)
    init_tables();
    const int64_t nb = (W.type == 20 || W.type == 8) ? W.in / 32 : W.in / 256;
    pool.parallel_for(W.out, grain, [&](int64_t b, int64_t e) {
        for (int64_t o = b; o < e; ++o) {
            const uint8_t * wrow = W.p + static_cast<size_t>(o) * W.row_bytes;
            float * ycol = y + o;
            switch (W.type) {
                case 12: row_q4k<false>(wrow, nb, act, arow, T, ycol, W.out); break;
                case 13: row_q4k<true>(wrow, nb, act, arow, T, ycol, W.out); break;
                case 14: row_q6k(wrow, nb, act, arow, T, ycol, W.out); break;
                case 23: row_iq4xs(wrow, nb, act, arow, T, ycol, W.out); break;
                case 20: row_sym32<true>(wrow, nb, act, arow, T, ycol, W.out); break;
                case 8: row_sym32<false>(wrow, nb, act, arow, T, ycol, W.out); break;
                default: std::abort();
            }
        }
    });
#else
    (void) pool; (void) W; (void) act; (void) arow; (void) T; (void) y; (void) grain;
    std::abort();
#endif
}
