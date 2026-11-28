// Teacher-forced DECODE comparison of attention kernels at a real context (E37, option C of the int8 attention proposal).
// For every mode (fp32 = the current decode v2 kernel, i8 = the int8 K prototype, split = the older split-K kernel, an equally
// correct FP32 kernel that only sums in a different order = the "numerical noise floor") the same text is run:
//   reset, prefill the first P tokens (chunks of t_max, prefill kernels, identical in every mode), then M steps of forward(1 token)
//   with the TRUE next token of the text, so every step goes through the decode attention kernel at position P + i.
// The full log-softmax of every step is kept; mode 0 is the reference. Reported per mode against mode 0:
//   mean / median / p95 / max KL(P_ref || P_mode), top-1 agreement, and the NLL (perplexity) of the true next token for both.
// CUDA graphs are disabled (a captured graph would keep replaying the kernel of the first mode).
// Needs a ops object with op_attention_set_i8 (see the build recipe in the log entry); fp32/split work with the stock ops.cu.
// Usage: kld_decode -m MODEL.gguf -f TEXT --prefill P --steps M [--ctx C] [--modes fp32,i8,split]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "engine.hpp"

void op_attention_set_dec2(bool on);
void op_attention_set_i8(bool on);

static std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> o; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, c)) o.push_back(t);
    return o;
}

int main(int argc, char ** argv) {
    std::string model, textf, modes_s = "fp32,i8";
    int P = 4000, M = 128, ctx = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&] { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "-m") model = next(); else if (a == "-f") textf = next(); else if (a == "--prefill") P = std::atoi(next().c_str());
        else if (a == "--steps") M = std::atoi(next().c_str()); else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--modes") modes_s = next();
    }
    if (model.empty() || textf.empty()) { std::fprintf(stderr, "usage: kld_decode -m MODEL -f TEXT --prefill P --steps M [--ctx C] [--modes fp32,i8,split]\n"); return 2; }
    EngineConfig cfg;
    cfg.gguf = model;
    cfg.ctx_max = ctx > 0 ? ctx : ((P + M + 4095) / 4096 + 1) * 4096;
    cfg.gpu_layers = -1;
    cfg.mtp_steps = 0;
    cfg.cuda_graph = false;
    Engine eng(cfg);
    std::ifstream f(textf, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<int32_t> ids = eng.tok.encode(text, false);
    if (static_cast<int>(ids.size()) < P + M + 1) { std::fprintf(stderr, "text has %zu tokens, need %d\n", ids.size(), P + M + 1); return 2; }
    const int V = eng.n_vocab;
    const int T = std::max(1, eng.config().t_max);
    std::fprintf(stderr, "text tokens %zu, prefill %d, steps %d, ctx_max %d, vocab %d\n", ids.size(), P, M, eng.config().ctx_max, V);
    const std::vector<std::string> modes = split(modes_s, ',');
    std::vector<float> logits(V);
    std::vector<std::vector<float>> lp(modes.size());
    for (size_t mi = 0; mi < modes.size(); ++mi) {
        const std::string & md = modes[mi];
        op_attention_set_i8(md == "i8");
        op_attention_set_dec2(md != "split");
        const auto t0 = std::chrono::steady_clock::now();
        eng.reset();
        for (int off = 0; off < P; off += T) eng.forward(ids.data() + off, std::min(T, P - off), off, logits.data());
        const auto t1 = std::chrono::steady_clock::now();
        lp[mi].resize(static_cast<size_t>(M) * V);
        for (int s = 0; s < M; ++s) {
            eng.forward(ids.data() + P + s, 1, P + s, logits.data());
            double mx = -1e300;
            for (int v = 0; v < V; ++v) mx = std::max(mx, static_cast<double>(logits[v]));
            double sum = 0;
            for (int v = 0; v < V; ++v) sum += std::exp(static_cast<double>(logits[v]) - mx);
            const double lse = mx + std::log(sum);
            float * dst = lp[mi].data() + static_cast<size_t>(s) * V;
            for (int v = 0; v < V; ++v) dst[v] = static_cast<float>(logits[v] - lse);
        }
        const auto t2 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "mode %-6s prefill %.1f s, %d decode steps %.1f s\n", md.c_str(), std::chrono::duration<double>(t1 - t0).count(), M,
                     std::chrono::duration<double>(t2 - t1).count());
    }
    std::printf("context %d tokens + %d teacher-forced decode steps (positions %d .. %d), reference mode = %s\n", P, M, P, P + M - 1, modes[0].c_str());
    auto nll_of = [&](size_t mi, int s) { return -static_cast<double>(lp[mi][static_cast<size_t>(s) * V + ids[P + s + 1]]); };
    double ref_nll = 0;
    for (int s = 0; s < M; ++s) ref_nll += nll_of(0, s);
    std::printf("%-8s mean NLL %.5f  PPL %.4f\n", modes[0].c_str(), ref_nll / M, std::exp(ref_nll / M));
    for (size_t mi = 1; mi < modes.size(); ++mi) {
        std::vector<double> kl(M), dn(M);
        int top1 = 0;
        double nll = 0;
        for (int s = 0; s < M; ++s) {
            const float * a = lp[0].data() + static_cast<size_t>(s) * V, * b = lp[mi].data() + static_cast<size_t>(s) * V;
            double k = 0; int ia = 0, ib = 0;
            for (int v = 0; v < V; ++v) {
                k += std::exp(static_cast<double>(a[v])) * (static_cast<double>(a[v]) - b[v]);
                if (a[v] > a[ia]) ia = v;
                if (b[v] > b[ib]) ib = v;
            }
            kl[s] = k; top1 += ia == ib;
            dn[s] = nll_of(mi, s) - nll_of(0, s); nll += nll_of(mi, s);
        }
        if (std::getenv("KLD_VERBOSE")) {
            std::vector<int> ord(M); for (int s = 0; s < M; ++s) ord[s] = s;
            std::sort(ord.begin(), ord.end(), [&](int x, int y) { return kl[x] > kl[y]; });
            for (int r = 0; r < std::min(M, 6); ++r) std::printf("    %s worst step %3d (pos %d): KL %.4e  true-token NLL ref %.3f mode %.3f\n", modes[mi].c_str(), ord[r], P + ord[r], kl[ord[r]], nll_of(0, ord[r]), nll_of(mi, ord[r]));
        }
        std::vector<double> ks = kl; std::sort(ks.begin(), ks.end());
        double mean = 0; for (double k : kl) mean += k; mean /= M;
        double dm = 0; for (double d : dn) dm += d; dm /= M;
        double dv = 0; for (double d : dn) dv += (d - dm) * (d - dm); dv /= std::max(1, M - 1);
        std::printf("%-8s vs %s: KLD mean %.3e  median %.3e  p95 %.3e  max %.3e | top-1 agree %d/%d (%.1f %%) | mean NLL %.5f PPL %.4f | dNLL %+.5f +- %.5f\n",
                    modes[mi].c_str(), modes[0].c_str(), mean, ks[M / 2], ks[std::min(M - 1, static_cast<int>(0.95 * M))], ks.back(), top1, M,
                    100.0 * top1 / M, nll / M, std::exp(nll / M), dm, std::sqrt(dv / M));
    }
    return 0;
}
