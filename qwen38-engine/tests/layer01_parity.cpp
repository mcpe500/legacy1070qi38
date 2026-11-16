// Stage 2 parity: layers 0 and 1 of Qwen3.8-27B against the Stage 1 reference tensors.
// Layer 0 input is model.input_embed (ref). Layer 1 input is the reference l_out-0, so each layer
// is tested in isolation. Every captured stage is compared by cosine similarity and max abs error.
// Usage: layer01_parity <GGUF> <ref_dir>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "dn_layer.hpp"
#include "gguf_index.hpp"

static bool read_ref(const std::string & path, std::vector<float> & v) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    f.seekg(0);
    v.resize(static_cast<size_t>(n) / sizeof(float));
    f.read(reinterpret_cast<char *>(v.data()), n);
    return static_cast<bool>(f);
}

struct Cmp { double cos, max_abs, rel_rms; size_t n; };

static Cmp compare(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0, mx = 0, err2 = 0;
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; ++i) {
        const double x = a[i], y = b[i];
        dot += x * y; na += x * x; nb += y * y;
        const double d = std::fabs(x - y);
        if (d > mx) mx = d;
        err2 += (x - y) * (x - y);
    }
    Cmp c;
    c.cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
    c.max_abs = mx;
    c.rel_rms = nb > 0 ? std::sqrt(err2 / nb) : 0.0;
    c.n = n;
    return c;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <GGUF> <ref_dir>\n", argv[0]);
        return 2;
    }
    const std::string ref = argv[2];
    GgufIndex ix;
    ix.open(argv[1]);
    const auto it = ix.f32_kv.find("qwen35.attention.layer_norm_rms_epsilon");
    if (it == ix.f32_kv.end() || it->second != g_rms_eps()) {
        std::fprintf(stderr, "eps mismatch: GGUF=%g code=%g\n",
                     it == ix.f32_kv.end() ? -1.0 : it->second, g_rms_eps());
        return 1;
    }

    const double COS_MIN = 0.995;
    int failures = 0;
    std::vector<float> input;
    if (!read_ref(ref + "/model.input_embed.bin", input)) { std::fprintf(stderr, "missing input\n"); return 1; }

    for (int il = 0; il <= 1; ++il) {
        if (il == 1) {
            if (!read_ref(ref + "/l_out-0.bin", input)) { std::fprintf(stderr, "missing l_out-0\n"); return 1; }
        }
        const int64_t T = static_cast<int64_t>(input.size()) / kHidden;
        DNWeights w = load_dn_weights(ix, il);
        DNState st;
        Trace tr;
        std::vector<float> out;
        dn_layer_forward(w, il, input, T, st, out, &tr);

        std::printf("\n== layer %d (T=%lld) ==\n", il, static_cast<long long>(T));
        std::printf("%-26s %10s %12s %12s\n", "stage", "cosine", "max_abs", "rel_rms");
        for (const auto & kv : tr.t) {
            std::vector<float> rv;
            const std::string path = ref + "/" + kv.first + ".bin";
            if (!read_ref(path, rv)) {
                std::printf("%-26s   (no reference file)\n", kv.first.c_str());
                continue;
            }
            const Cmp c = compare(kv.second, rv);
            const bool ok = c.n == rv.size() && c.cos >= COS_MIN;
            if (!ok) ++failures;
            std::printf("%-26s %10.6f %12.3e %12.3e %s%s\n", kv.first.c_str(), c.cos, c.max_abs, c.rel_rms,
                        c.n == rv.size() ? "" : "[SIZE] ", ok ? "" : "<-- FAIL");
        }
        // The layer output itself is the key result for the next layer.
        std::vector<float> lout_ref;
        read_ref(ref + "/l_out-" + std::to_string(il) + ".bin", lout_ref);
        const Cmp fin = compare(out, lout_ref);
        std::printf("FINAL l_out-%d: cosine=%.6f max_abs=%.3e rel_rms=%.3e\n", il, fin.cos, fin.max_abs, fin.rel_rms);
        if (fin.cos < COS_MIN) ++failures;
    }
    std::printf("\nresult: %s (threshold cosine >= %.3f, failures=%d)\n", failures ? "FAIL" : "PASS", COS_MIN, failures);
    return failures ? 1 : 0;
}
