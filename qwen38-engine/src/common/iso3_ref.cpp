#include "iso3_ref.hpp"

#include <cmath>
#include <cstring>

#include "f16bits.hpp"

#define QT_BEGIN(type, name, size) static const type name[size] = {
#define QT_END() };
#include "tables.inc"
#undef QT_BEGIN
#undef QT_END

namespace {

// Left product q * v of the unit quaternion of group g (conjugated when inv) with v = (v0, v1, v2, v3).
void rotate(int g, bool inv, const float * v, float * r) {
    const float w = iso3_quat_w[g];
    const float s = inv ? -1.0f : 1.0f;
    const float x = s * iso3_quat_x[g], y = s * iso3_quat_y[g], z = s * iso3_quat_z[g];
    r[0] = w * v[0] - x * v[1] - y * v[2] - z * v[3];
    r[1] = w * v[1] + x * v[0] + y * v[3] - z * v[2];
    r[2] = w * v[2] - x * v[3] + y * v[0] + z * v[1];
    r[3] = w * v[3] + x * v[2] - y * v[1] + z * v[0];
}

int nearest(float v) {  // index of the nearest codebook entry, ties go to the lower index
    int idx = 0;
    for (int i = 0; i < 7; ++i) idx += v > 0.5f * (iso3_codebook[i] + iso3_codebook[i + 1]);
    return idx;
}

}  // namespace

void iso3_quantize_block_ref(const float * src, uint8_t * out) {
    float sum = 0.0f;
    for (int j = 0; j < ISO3_QK; ++j) sum += src[j] * src[j];
    const float norm = std::sqrt(sum);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

    uint8_t * qs = out + 2;
    uint8_t * signs = out + 34;
    std::memset(qs, 0, 32);
    std::memset(signs, 0, 16);

    float recon = 0.0f;
    for (int g = 0; g < ISO3_QK / 4; ++g) {
        const float v[4] = {src[4 * g + 0] * inv_norm, src[4 * g + 1] * inv_norm, src[4 * g + 2] * inv_norm,
                            src[4 * g + 3] * inv_norm};
        float r[4];
        rotate(g, false, v, r);
        for (int c = 0; c < 4; ++c) {
            const int j = 4 * g + c;
            const int idx = nearest(r[c]);
            qs[j / 4] = static_cast<uint8_t>(qs[j / 4] | ((idx & 3) << (2 * (j % 4))));
            signs[j / 8] = static_cast<uint8_t>(signs[j / 8] | ((idx >> 2) << (j % 8)));
            recon += iso3_codebook[idx] * iso3_codebook[idx];
        }
    }
    const float recon_norm = std::sqrt(recon);
    const uint16_t nh = f32_to_f16(recon_norm > 1e-10f ? norm / recon_norm : norm);
    out[0] = static_cast<uint8_t>(nh & 0xFF);
    out[1] = static_cast<uint8_t>(nh >> 8);
}

void iso3_dequantize_block_ref(const uint8_t * in, float * y) {
    const float norm = f16_to_f32(static_cast<uint32_t>(in[0]) | (static_cast<uint32_t>(in[1]) << 8));
    const uint8_t * qs = in + 2;
    const uint8_t * signs = in + 34;
    for (int g = 0; g < ISO3_QK / 4; ++g) {
        float q[4];
        for (int c = 0; c < 4; ++c) {
            const int j = 4 * g + c;
            const int idx = ((qs[j / 4] >> (2 * (j % 4))) & 3) | (((signs[j / 8] >> (j % 8)) & 1) << 2);
            q[c] = iso3_codebook[idx];
        }
        float r[4];
        rotate(g, true, q, r);
        for (int c = 0; c < 4; ++c) y[4 * g + c] = r[c] * norm;
    }
}
