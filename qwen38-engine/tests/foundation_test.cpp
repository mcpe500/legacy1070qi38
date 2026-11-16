// Foundation checks that every GPU kernel relies on:
//  1. f16_to_f32 matches the F16C instruction for all 65536 bit patterns (NaN excluded).
//  2. f32_to_f16 (round to nearest even, integer bit operations) matches F16C on random floats, including the
//     subnormal and overflow ranges.
//  3. iso3_quantize_block_ref / iso3_dequantize_block_ref match, byte for byte, the GGML_TYPE_ISO3_0 type of the
//     patched libggml built from the baseline tree (the implementation PR #38 ships).
//  4. iso3 round trip error stays inside the PR #38 threshold (relative MSE 0.05 on Gaussian data).
#include <immintrin.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "f16bits.hpp"
#include "ggml.h"
#include "iso3_ref.hpp"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; std::printf("  FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

int main() {
    // 1. fp16 -> fp32, exhaustive
    {
        long bad = 0, n = 0;
        for (uint32_t h = 0; h < 65536; ++h) {
            if (((h >> 10) & 0x1F) == 31 && (h & 0x3FF)) continue;  // NaN payloads are not compared
            const float a = f16_to_f32(h), b = _cvtsh_ss(static_cast<unsigned short>(h));
            ++n;
            if (qf2u(a) != qf2u(b)) { if (bad++ < 5) std::printf("  mismatch h=0x%04x mine=%a ref=%a\n", h, a, b); }
        }
        std::printf("f16->f32: %ld patterns, %ld mismatches\n", n, bad);
        CHECK(bad == 0, "f16_to_f32 differs from F16C");
    }
    // 2. fp32 -> fp16, random across exponents
    {
        std::mt19937_64 rng(1234);
        long bad = 0, n = 0;
        for (long i = 0; i < 4000000; ++i) {
            uint32_t bits = static_cast<uint32_t>(rng());
            if (i % 3 == 0) bits = (bits & 0x807FFFFFu) | (static_cast<uint32_t>(95 + (rng() % 50)) << 23);  // 2^-32 .. 2^18
            const float f = qu2f(bits);
            if (std::isnan(f)) continue;
            ++n;
            const uint16_t a = f32_to_f16(f);
            const uint16_t b = _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            if (a != b) { if (bad++ < 5) std::printf("  mismatch f=%a mine=0x%04x ref=0x%04x\n", f, a, b); }
        }
        std::printf("f32->f16: %ld floats, %ld mismatches\n", n, bad);
        CHECK(bad == 0, "f32_to_f16 differs from F16C");
    }
    // 3 and 4. iso3 against libggml
    {
        const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_ISO3_0);
        CHECK(tr && tr->from_float_ref && tr->to_float, "libggml has no iso3 type");
        CHECK(ggml_row_size(GGML_TYPE_ISO3_0, 128) == ISO3_BYTES, "iso3 block is not %d bytes", ISO3_BYTES);
        if (tr && tr->from_float_ref && tr->to_float) {
            std::mt19937 rng(7);
            long qbad = 0, dbad = 0, blocks = 0;
            double err2 = 0, ref2 = 0;
            const float scales[] = {0.01f, 1.0f, 50.0f, 1e-4f};
            for (int s = 0; s < 4; ++s) {
                std::normal_distribution<float> nd(0.0f, scales[s]);
                for (int b = 0; b < 3000; ++b) {
                    float x[128];
                    for (float & v : x) v = nd(rng);
                    uint8_t mine[ISO3_BYTES], theirs[ISO3_BYTES];
                    iso3_quantize_block_ref(x, mine);
                    tr->from_float_ref(x, theirs, 128);
                    ++blocks;
                    if (std::memcmp(mine, theirs, ISO3_BYTES) != 0) ++qbad;
                    float dm[128], dt[128];
                    iso3_dequantize_block_ref(mine, dm);
                    tr->to_float(theirs, dt, 128);
                    if (std::memcmp(dm, dt, sizeof(dm)) != 0) ++dbad;
                    for (int j = 0; j < 128; ++j) { err2 += (dm[j] - x[j]) * (dm[j] - x[j]); ref2 += x[j] * x[j]; }
                }
            }
            const double nmse = err2 / ref2;
            std::printf("iso3: %ld blocks, %ld quantize byte mismatches, %ld dequantize mismatches, round-trip nmse %.4f\n",
                        blocks, qbad, dbad, nmse);
            CHECK(qbad == 0, "iso3 quantize differs from libggml");
            CHECK(dbad == 0, "iso3 dequantize differs from libggml");
            CHECK(nmse < 0.05, "iso3 round-trip nmse %.4f above 0.05", nmse);
        }
    }
    std::printf("result: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
