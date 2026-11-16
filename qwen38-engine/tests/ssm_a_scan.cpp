// Point D evidence (2026-11-16): the proposed optimization replaced the ssm_a gate term with a
// sigmoid(-x) identity that is only valid when a head's ssm_a equals exactly -1.0. This tool scans every
// blk.<L>.ssm_a tensor in the GGUF (raw F32, [n_heads]) and reports min/max per layer plus how many values
// are bitwise -1.0f or within 1e-6 of -1.0. Its output is the recorded justification for skipping Point D.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "gguf_index.hpp"

int main(int argc, char ** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <GGUF>\n", argv[0]); return 2; }
    GgufIndex ix;
    ix.open(argv[1]);
    int layers = 0;
    double gmin = 1e30, gmax = -1e30;
    long long exact = 0, near = 0, total = 0;
    for (int L = 0; L < 128; ++L) {
        const std::string name = "blk." + std::to_string(L) + ".ssm_a";
        if (!ix.tensors.count(name)) continue;
        const GgufTensor & t = ix.get(name);
        if (t.type != 0) {  // GGML_TYPE_F32
            std::fprintf(stderr, "%s: unexpected tensor type %u\n", name.c_str(), t.type);
            return 1;
        }
        const float * p = reinterpret_cast<const float *>(ix.data(t));
        double mn = 1e30, mx = -1e30;
        long long e = 0, n = 0;
        for (int64_t i = 0; i < t.ne[0]; ++i) {
            const float v = p[i];
            mn = mn < v ? mn : v;
            mx = mx > v ? mx : v;
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            if (bits == 0xBF800000u) ++e;                       // bitwise -1.0f
            if (std::fabs(static_cast<double>(v) + 1.0) < 1e-6) ++n;
        }
        ++layers;
        total += t.ne[0];
        exact += e;
        near += n;
        if (mn < gmin) gmin = mn;
        if (mx > gmax) gmax = mx;
        std::printf("blk.%-3d ssm_a ne=%-3lld min=%.6f max=%.6f bitwise(-1.0f)=%lld near(1e-6)=%lld\n",
                    L, static_cast<long long>(t.ne[0]), mn, mx, e, n);
    }
    std::printf("layers=%d values=%lld global min=%.6f max=%.6f bitwise(-1.0f)=%lld |x+1|<1e-6=%lld\n",
                layers, total, gmin, gmax, exact, near);
    std::printf("CONCLUSION: sigmoid(-x) identity for ssm_a is %s (Point D %s)\n",
                (exact == 0 && near == 0) ? "INVALID — no head equals -1.0, so the exact form still needs powf"
                                          : "VALID",
                (exact == 0 && near == 0) ? "stays skipped" : "could apply");
    return 0;
}
