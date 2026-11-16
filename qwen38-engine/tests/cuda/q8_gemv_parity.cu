// Stage 3 kernel parity: q8_0_gemv_dp4a on real layer-0 Q8_0 weights (ssm_alpha, ssm_beta) with the
// reference input attn_norm-0 (Stage 1 dump). Compared with the reference alpha-0 (raw) and
// beta_sigmoid-0 (after sigmoid). A host emulation that quantizes activations the same way checks
// the kernel arithmetic exactly.
// Usage: q8_gemv_parity <GGUF> <ref_dir>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf_index.hpp"
#include "q8_gemv.cuh"

static bool read_ref(const std::string & path, std::vector<float> & v) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    f.seekg(0);
    v.resize(static_cast<size_t>(n) / sizeof(float));
    f.read(reinterpret_cast<char *>(v.data()), n);
    return static_cast<bool>(f);
}

struct Cmp { double cos, max_abs; };
static Cmp compare(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0, mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
        mx = std::fmax(mx, std::fabs((double)a[i] - b[i]));
    }
    return {dot / std::sqrt(na * nb), mx};
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

// Host emulation: quantize x per 32-block exactly like the kernel, then integer dot products.
static void host_q8_gemv(const uint8_t * W, int64_t in, int64_t out, const float * x, int64_t T, float * y) {
    const int64_t nb = in / 32;
    std::vector<int8_t> q(static_cast<size_t>(in));
    std::vector<float> d(static_cast<size_t>(nb));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t b = 0; b < nb; ++b) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[t * in + b * 32 + i]));
            const float dd = amax / 127.0f;
            d[b] = dd;
            const float inv = dd > 0.0f ? 1.0f / dd : 0.0f;
            for (int i = 0; i < 32; ++i) {
                int v = (int)std::nearbyint(x[t * in + b * 32 + i] * inv);
                v = v < -127 ? -127 : (v > 127 ? 127 : v);
                q[b * 32 + i] = (int8_t)v;
            }
        }
        for (int64_t o = 0; o < out; ++o) {
            double acc = 0.0;
            for (int64_t b = 0; b < nb; ++b) {
                const uint8_t * blk = W + (o * nb + b) * 34;
                const float dw = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(blk));
                int isum = 0;
                for (int i = 0; i < 32; ++i) isum += (int)(int8_t)blk[2 + i] * (int)q[b * 32 + i];
                acc += (double)dw * d[b] * isum;
            }
            y[t * out + o] = (float)acc;
        }
    }
}

int main(int argc, char ** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s <GGUF> <ref_dir>\n", argv[0]); return 2; }
    const std::string ref = argv[2];
    GgufIndex ix;
    ix.open(argv[1]);
    const int64_t in = 5120;
    const GgufTensor & ta = ix.get("blk.0.ssm_alpha.weight");
    const GgufTensor & tb = ix.get("blk.0.ssm_beta.weight");
    if (ta.type != GGML_TYPE_Q8_0 || tb.type != GGML_TYPE_Q8_0) { std::fprintf(stderr, "expected Q8_0\n"); return 1; }
    const int64_t out = ta.ne[1];  // 48

    std::vector<float> x, alpha_ref, beta_ref;
    if (!read_ref(ref + "/attn_norm-0.bin", x) || !read_ref(ref + "/alpha-0.bin", alpha_ref) ||
        !read_ref(ref + "/beta_sigmoid-0.bin", beta_ref)) {
        std::fprintf(stderr, "missing reference files\n"); return 1;
    }
    const int64_t T = (int64_t)x.size() / in;

    uint8_t * dWa; uint8_t * dWb; float * dx; float * dya; float * dyb;
    CK(cudaMalloc(&dWa, ta.nbytes)); CK(cudaMalloc(&dWb, tb.nbytes));
    CK(cudaMemcpy(dWa, ix.data(ta), ta.nbytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWb, ix.data(tb), tb.nbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx, sizeof(float) * x.size()));
    CK(cudaMalloc(&dya, sizeof(float) * T * out)); CK(cudaMalloc(&dyb, sizeof(float) * T * out));
    CK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    Q8Scratch sc;
    CK(q8_scratch_alloc(sc, T, in));

    CK(q8_0_gemv_dp4a(dWa, in, out, dx, T, dya, sc));
    CK(q8_0_gemv_dp4a(dWb, in, out, dx, T, dyb, sc));
    CK(cudaDeviceSynchronize());

    std::vector<float> ya(T * out), yb(T * out);
    CK(cudaMemcpy(ya.data(), dya, sizeof(float) * ya.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(yb.data(), dyb, sizeof(float) * yb.size(), cudaMemcpyDeviceToHost));

    // Host emulation of the same quantized arithmetic.
    std::vector<float> ha(T * out), hb(T * out);
    const std::vector<uint8_t> Wa_host(ix.data(ta), ix.data(ta) + ta.nbytes);
    const std::vector<uint8_t> Wb_host(ix.data(tb), ix.data(tb) + tb.nbytes);
    host_q8_gemv(Wa_host.data(), in, out, x.data(), T, ha.data());
    host_q8_gemv(Wb_host.data(), in, out, x.data(), T, hb.data());

    std::vector<float> beta_gpu(yb.size());
    for (size_t i = 0; i < yb.size(); ++i) beta_gpu[i] = 1.0f / (1.0f + std::exp(-yb[i]));

    const Cmp ca = compare(ya, alpha_ref), cb = compare(beta_gpu, beta_ref);
    const Cmp ea = compare(ya, ha), eb = compare(yb, hb);
    std::printf("T=%lld out=%lld in=%lld\n", (long long)T, (long long)out, (long long)in);
    std::printf("alpha vs ref alpha-0      cos=%.6f max_abs=%.3e\n", ca.cos, ca.max_abs);
    std::printf("beta_sig vs ref beta_sig  cos=%.6f max_abs=%.3e\n", cb.cos, cb.max_abs);
    std::printf("GPU vs host emulation     alpha max_abs=%.3e  beta max_abs=%.3e\n", ea.max_abs, eb.max_abs);

    const bool pass = ca.cos >= 0.995 && cb.cos >= 0.995 && ea.max_abs < 1e-3 && eb.max_abs < 1e-3;
    std::printf("result: %s\n", pass ? "PASS" : "FAIL");

    q8_scratch_free(sc);
    cudaFree(dWa); cudaFree(dWb); cudaFree(dx); cudaFree(dya); cudaFree(dyb);
    return pass ? 0 : 1;
}
