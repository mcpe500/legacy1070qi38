// GPU layers 0..N-1 against the reference dump of the baseline llama.cpp run (tests/ref_full, 5-token prompt).
// The residual stream after each layer is compared with l_out-<il>.bin by cosine and max abs error.
// Usage: gpu_layers_parity <GGUF> <ref_dir> [n_layers=24] [kv=iso3|f32]
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "gguf_index.hpp"
#include "gpu_layers.hpp"

static bool read_floats(const std::string & path, std::vector<float> & v) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    f.seekg(0);
    v.resize(static_cast<size_t>(n) / sizeof(float));
    f.read(reinterpret_cast<char *>(v.data()), n);
    return static_cast<bool>(f);
}

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <GGUF> <ref_dir> [n_layers] [iso3|f32]\n", argv[0]); return 2; }
    const std::string ref = argv[2];
    GpuConfig cfg;
    cfg.n_layers = argc > 3 ? std::atoi(argv[3]) : 24;
    cfg.kv_iso3 = !(argc > 4 && std::string(argv[4]) == "f32");
    cfg.dp4a_mask = argc > 5 ? parse_dp4a_mask(argv[5]) : 0;
    cfg.t_max = 64;

    size_t free0 = 0, total = 0, free1 = 0;
    cudaMemGetInfo(&free0, &total);
    GgufIndex ix;
    ix.open(argv[1]);
    const float eps = ix.f32_kv.at("qwen35.attention.layer_norm_rms_epsilon");
    GpuLayers gpu(ix, cfg, eps);
    cudaMemGetInfo(&free1, &total);
    std::printf("dp4a mask 0x%08x\n", cfg.dp4a_mask);
    std::printf("GPU layers 0..%d, KV %s: arena %.1f MiB (weights %.1f MiB, KV %.1f MiB), device memory used %.1f MiB of %.0f MiB\n",
                cfg.n_layers - 1, cfg.kv_iso3 ? "iso3" : "f32", static_cast<double>(gpu.vram_bytes()) / 1048576.0,
                static_cast<double>(gpu.weight_bytes()) / 1048576.0, static_cast<double>(gpu.kv_bytes()) / 1048576.0,
                static_cast<double>(free0 - free1) / 1048576.0, static_cast<double>(total) / 1048576.0);

    std::vector<float> x;
    if (!read_floats(ref + "/model.input_embed.bin", x)) { std::fprintf(stderr, "missing model.input_embed.bin\n"); return 1; }
    const int T = static_cast<int>(x.size() / 5120);

    int fails = 0;
    double worst = 1.0;
    int worst_il = -1;
    std::printf("\nlayer   cosine     max_abs\n");
    std::function<void(int, const float *, int)> tap = [&](int il, const float * xs, int Tn) {
        std::vector<float> r;
        if (!read_floats(ref + "/l_out-" + std::to_string(il) + ".bin", r) || r.size() != static_cast<size_t>(Tn) * 5120) {
            std::printf("%5d   (no reference)\n", il);
            ++fails;
            return;
        }
        double dot = 0, na = 0, nb = 0, mx = 0;
        for (size_t i = 0; i < r.size(); ++i) {
            dot += static_cast<double>(xs[i]) * r[i];
            na += static_cast<double>(xs[i]) * xs[i];
            nb += static_cast<double>(r[i]) * r[i];
            mx = std::fmax(mx, std::fabs(static_cast<double>(xs[i]) - r[i]));
        }
        const double c = dot / std::sqrt(na * nb);
        if (c < worst) { worst = c; worst_il = il; }
        const bool ok = c >= 0.995;
        if (!ok) ++fails;
        if (il < 8 || il % 4 == 3 || !ok) std::printf("%5d   %.6f  %.3e%s\n", il, c, mx, ok ? "" : "   <-- below 0.995");
    };
    gpu.forward(x.data(), T, 0, &tap);
    std::printf("\nworst layer cosine %.6f at layer %d\n", worst, worst_il);
    std::printf("result: %s (%d layers below 0.995)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
