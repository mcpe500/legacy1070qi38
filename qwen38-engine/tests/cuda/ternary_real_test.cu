// Real-weight check of the ternary path against llama.cpp on Ternary Bonsai 2 27B (Q2_0): for three matrices of layer 0 the GPU
// path ternary_rotate -> actq8_quantize -> gemv_q2_0_dp4a is applied to llama.cpp's own input tensor and compared with
// llama.cpp's output tensor (dumped by tests/tools/dump_ref_tensors with DUMP_SUFFIXES=-0), and with a double-precision
// reference (libggml dequantization + exact rotation) that isolates the kernel error from llama.cpp's numerics.
//   ternary_real_test MODEL.gguf REF_DIR      (REF_DIR also holds signs_<width>.f32 extracted from prism.hadamard.sign_values)
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "dp4a_gemv.cuh"
#include "ggml.h"
#include "gguf_index.hpp"
#include "ternary_gemv.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

namespace {

std::vector<float> readf(const std::string & p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::printf("missing %s\n", p.c_str()); std::exit(2); }
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<float> v(static_cast<size_t>(n) / 4);
    f.read(reinterpret_cast<char *>(v.data()), n);
    return v;
}

double cosine(const std::vector<float> & a, const std::vector<double> & b) {
    double d = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { d += a[i] * b[i]; na += static_cast<double>(a[i]) * a[i]; nb += b[i] * b[i]; }
    return d / std::sqrt(na * nb);
}

void fwht(std::vector<double> & v, size_t off) {
    for (size_t h = 1; h < 1024; h <<= 1)
        for (size_t i = 0; i < 1024; i += 2 * h)
            for (size_t j = i; j < i + h; ++j) { const double a = v[off + j], b = v[off + j + h]; v[off + j] = a + b; v[off + j + h] = a - b; }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) { std::printf("usage: %s MODEL.gguf REF_DIR\n", argv[0]); return 2; }
    GgufIndex ix;
    ix.open(argv[1]);
    const std::string dir = argv[2];
    struct Case { const char * weight; const char * in; const char * out; };
    const Case cases[] = {
        {"blk.0.ffn_up.weight", "attn_post_norm-0", "ffn_up-0"},
        {"blk.0.ffn_gate.weight", "attn_post_norm-0", "ffn_gate-0"},
        {"blk.0.ffn_down.weight", "ffn_swiglu-0", "ffn_out-0"},
    };
    int fails = 0;
    for (const Case & c : cases) {
        const GgufTensor & t = ix.get(c.weight);
        const int in = static_cast<int>(t.ne[0]), out = static_cast<int>(t.ne[1]);
        if (static_cast<int>(t.type) != kGtQ2_0) { std::printf("%s is type %d, not Q2_0\n", c.weight, static_cast<int>(t.type)); return 2; }
        const std::vector<float> x = readf(dir + "/" + c.in + ".bin"), yref = readf(dir + "/" + c.out + ".bin");
        const std::vector<float> s = readf(dir + "/signs_" + std::to_string(in) + ".f32");
        const int T = static_cast<int>(x.size() / in);
        if (static_cast<int>(s.size()) != in || static_cast<int>(yref.size()) != T * out) { std::printf("shape mismatch for %s\n", c.weight); return 2; }

        float *dx, *ds, *dy;
        uint8_t * dW;
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMalloc(&ds, s.size() * 4));
        CK(cudaMalloc(&dy, yref.size() * 4));
        CK(cudaMalloc(&dW, t.nbytes));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ds, s.data(), s.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW, ix.data(t), t.nbytes, cudaMemcpyHostToDevice));
        ActQ8 a;
        CK(actq8_alloc(a, T, in));
        CK(ternary_rotate(dx, ds, dx, in, T, 0));
        CK(actq8_quantize(dx, in, T, a, 0));
        CK(gemv_q2_0_dp4a(dW, in, out, a, T, dy, 0));
        std::vector<float> y(yref.size());
        CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));

        // double reference: libggml dequantization of the stored (rotated) weight, exact rotation of the FP32 input
        const ggml_type_traits * tr = ggml_get_type_traits(static_cast<ggml_type>(kGtQ2_0));
        std::vector<float> row(in);
        std::vector<double> xr(x.size()), ydbl(yref.size()), yllm(yref.begin(), yref.end());
        for (size_t i = 0; i < x.size(); ++i) xr[i] = static_cast<double>(x[i]) * s[i % in];
        for (int tt = 0; tt < T; ++tt) for (int b = 0; b < in / 1024; ++b) fwht(xr, static_cast<size_t>(tt) * in + b * 1024);
        for (auto & v : xr) v /= 32.0;
        const size_t rb = static_cast<size_t>(in / 64) * kQ2_0BlockBytes;
        for (int o = 0; o < out; ++o) {
            tr->to_float(ix.data(t) + o * rb, row.data(), in);
            for (int tt = 0; tt < T; ++tt) {
                double acc = 0;
                for (int i = 0; i < in; ++i) acc += static_cast<double>(row[i]) * xr[static_cast<size_t>(tt) * in + i];
                ydbl[static_cast<size_t>(tt) * out + o] = acc;
            }
        }
        std::vector<float> yllm_f(yref);
        const double c_llm = cosine(y, yllm), c_dbl = cosine(y, ydbl), c_ref = cosine(yllm_f, ydbl);
        const bool ok = c_llm > 0.999 && c_dbl > 0.999;
        std::printf("%s %-24s in %5d out %5d T %d | ours vs llama.cpp cos %.6f | ours vs double ref %.6f | llama.cpp vs double ref %.6f\n",
                    ok ? "PASS" : "FAIL", c.weight, in, out, T, c_llm, c_dbl, c_ref);
        if (!ok) ++fails;
        actq8_free(a);
        cudaFree(dx); cudaFree(ds); cudaFree(dy); cudaFree(dW);
    }
    std::printf("ternary_real_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
