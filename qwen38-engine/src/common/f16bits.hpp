// FP16 <-> FP32 conversion with integer bit operations only. Works in host and device code and never uses a
// half-precision instruction, so the SASS gate (zero HFMA2/HMUL2) holds even in kernels that read fp16 scales.
// Rounding of f32_to_f16 is round-to-nearest-even, identical to the F16C instruction used by ggml on the CPU.
#pragma once
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define QHD __host__ __device__ __forceinline__
#else
#define QHD inline
#endif

QHD uint32_t qf2u(float f) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(f);
#else
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
#endif
}

QHD float qu2f(uint32_t u) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, 4);
    return f;
#endif
}

QHD float f16_to_f32(uint32_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t out;
    if (e == 0) {
        if (m == 0) {
            out = sign;
        } else {  // subnormal: value = m * 2^-24; normalize so bit 10 is set
            uint32_t s = 0;
            while (!(m & 0x400u)) { m <<= 1; ++s; }
            out = sign | ((113u - s) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (e == 31) {
        out = sign | 0x7F800000u | (m << 13);
    } else {
        out = sign | ((e + 112u) << 23) | (m << 13);
    }
    return qu2f(out);
}

QHD uint16_t f32_to_f16(float f) {
    const uint32_t x = qf2u(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t e = static_cast<int32_t>((x >> 23) & 0xFFu);
    uint32_t m = x & 0x7FFFFFu;
    if (e == 255) return static_cast<uint16_t>(sign | 0x7C00u | (m ? 0x200u : 0u));
    const int32_t he = e - 127 + 15;
    if (he >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    if (he <= 0) {
        if (he < -10) return static_cast<uint16_t>(sign);
        m |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - he);
        uint32_t hm = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (hm & 1u))) ++hm;
        return static_cast<uint16_t>(sign | hm);
    }
    const uint32_t hm = m >> 13;
    const uint32_t rem = m & 0x1FFFu;
    uint32_t r = (static_cast<uint32_t>(he) << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) ++r;  // a carry into the exponent is correct, up to infinity
    return static_cast<uint16_t>(sign | r);
}
