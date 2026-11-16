// qwen38-engine: hybrid engine for Qwen3.8-27B. Defaults: 32 layers on the GPU, iso3 KV, __dp4a for every quantization type,
// prefill streamed through the GPU, MTP draft head with 2 steps. CLI generation, or an HTTP/SSE server with --server.
//   qwen38-engine -m MODEL.gguf -p "prompt" -n 32 --temp 0 --seed 42 [--parity-dir DIR] [--resident-gib G]
//     [--gpu-layers N] [--kv iso3|f32] [--dp4a all|none|LIST] [--mtp-steps 0..3] [--stream-min-t N (0 = off)]
//   qwen38-engine -m MODEL.gguf --tokenize-file corpus.txt        (samples separated by the 0x1E byte)
// Prints a JSON summary on the last line of stdout.
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "arena.hpp"
#include "engine.hpp"
#include "generator.hpp"
#include "gpu_layers.hpp"
#include "server.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

bool read_floats(const std::string & path, std::vector<float> & v) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    f.seekg(0);
    v.resize(static_cast<size_t>(n) / sizeof(float));
    f.read(reinterpret_cast<char *>(v.data()), n);
    return static_cast<bool>(f);
}

double cosine(const float * a, const float * b, size_t n, double * max_abs) {
    double dot = 0, na = 0, nb = 0, mx = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
        mx = std::fmax(mx, std::fabs(static_cast<double>(a[i]) - b[i]));
    }
    if (max_abs) *max_abs = mx;
    return (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
}

long long swap_used_kb() {
    std::ifstream f("/proc/meminfo");
    std::string k, unit;
    long long v, total = 0, freev = 0;
    while (f >> k >> v >> unit) {
        if (k == "SwapTotal:") total = v;
        if (k == "SwapFree:") freev = v;
    }
    return total - freev;
}

long long status_kb(const char * key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    const size_t kl = std::strlen(key);
    while (std::getline(f, line)) {
        if (line.compare(0, kl, key) == 0) return std::atoll(line.c_str() + kl);
    }
    return -1;
}
long long peak_rss_kb() { return status_kb("VmHWM:"); }

std::string json_escape(const std::string & s) {
    std::string r;
    for (unsigned char c : s) {
        switch (c) {
            case '"': r += "\\\""; break;
            case '\\': r += "\\\\"; break;
            case '\n': r += "\\n"; break;
            case '\r': r += "\\r"; break;
            case '\t': r += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); r += b; }
                else r += static_cast<char>(c);
        }
    }
    return r;
}

int argmax(const float * v, int64_t n) {
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
    return static_cast<int>(best);
}

}  // namespace

int main(int argc, char ** argv) {
    EngineConfig cfg;
    GenParams gp;
    gp.sp.temperature = 0.0f;
    gp.sp.seed = 42;
    std::string prompt = "The capital of France is", parity_dir, tok_file;
    int n_predict = 16;
    bool parse_special = false, rollback_test = false, serve = false;
    ServerOptions sopt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m") cfg.gguf = next();
        else if (a == "-p") prompt = next();
        else if (a == "-n") n_predict = std::atoi(next().c_str());
        else if (a == "--temp") gp.sp.temperature = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--top-k") gp.sp.top_k = std::atoi(next().c_str());
        else if (a == "--top-p") gp.sp.top_p = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--min-p") gp.sp.min_p = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--repeat-penalty") gp.sp.repeat_penalty = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--presence-penalty") gp.sp.presence_penalty = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--repeat-last-n") gp.sp.repeat_last_n = std::atoi(next().c_str());
        else if (a == "--seed") gp.sp.seed = std::atoll(next().c_str());
        else if (a == "--ctx") cfg.ctx_max = std::atoi(next().c_str());
        else if (a == "--tmax") cfg.t_max = std::atoi(next().c_str());
        else if (a == "--threads") cfg.threads = std::atoi(next().c_str());
        else if (a == "--resident-gib") cfg.resident_gib = std::atof(next().c_str());
        else if (a == "--gpu-layers") cfg.gpu_layers = std::atoi(next().c_str());
        else if (a == "--dp4a") cfg.dp4a_mask = parse_dp4a_mask(next().c_str());
        else if (a == "--rollback-test") rollback_test = true;
        else if (a == "--server") serve = true;
        else if (a == "--port") sopt.port = std::atoi(next().c_str());
        else if (a == "--host") sopt.host = next();
        else if (a == "--mtp-steps") cfg.mtp_steps = std::atoi(next().c_str());
        else if (a == "--mtp-adaptive") gp.mtp_adaptive = std::atoi(next().c_str()) != 0;
        else if (a == "--mtp-p-keep") gp.mtp_p_keep = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--mtp-p-cont") gp.mtp_p_cont = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--mtp-window") gp.mtp_window = std::atoi(next().c_str());
        else if (a == "--mtp-gpu-prefill") cfg.mtp_gpu_prefill = std::atoi(next().c_str()) != 0;
        else if (a == "--pld") gp.pld_enabled = std::atoi(next().c_str()) != 0;
        else if (a == "--pld-max") gp.pld_max = std::atoi(next().c_str());
        else if (a == "--pld-min-len") gp.pld_min_len = std::atoi(next().c_str());
        else if (a == "--pin-stream") cfg.pin_stream = std::atoi(next().c_str()) != 0;
        else if (a == "--cuda-graph") cfg.cuda_graph = std::atoi(next().c_str()) != 0;
        else if (a == "--stream-min-t") cfg.stream_min_t = std::atoi(next().c_str());
        else if (a == "--kv") { const std::string k = next(); if (k != "iso3" && k != "f32") { std::fprintf(stderr, "--kv must be iso3 or f32\n"); return 2; } cfg.kv_iso3 = (k == "iso3"); }
        else if (a == "--parity-dir") parity_dir = next();
        else if (a == "--tokenize-file") tok_file = next();
        else if (a == "--parse-special") parse_special = true;
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }
    if (cfg.gguf.empty()) { std::fprintf(stderr, "usage: %s -m MODEL.gguf [-p prompt] [-n N] [--server --port 8080]\n", argv[0]); return 2; }
    if (cfg.mtp_steps < 0 || cfg.mtp_steps > 3) { std::fprintf(stderr, "--mtp-steps must be 0..3\n"); return 2; }
    gp.mtp_steps = cfg.mtp_steps;
    gp.max_tokens = n_predict;

    try {
        if (!tok_file.empty()) {  // tokenizer-only mode: no weights are touched
            GgufIndex ix;
            ix.open(cfg.gguf);
            Tokenizer tk;
            tk.init(ix);
            std::ifstream f(tok_file, std::ios::binary);
            const std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            size_t b = 0;
            while (b <= all.size()) {
                size_t e = all.find('\x1e', b);
                if (e == std::string::npos) e = all.size();
                const std::vector<int32_t> ids = tk.encode(all.substr(b, e - b), parse_special);
                std::printf("[");
                for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? "," : "", ids[i]);
                std::printf("]\n");
                b = e + 1;
            }
            return 0;
        }

        const long long swap0 = swap_used_kb();
        const auto t_load0 = Clock::now();
        Engine eng(cfg);
        const double load_s = secs(t_load0, Clock::now());
        std::fprintf(stderr, "loaded in %.2f s; CPU arena %.1f MiB; CPU resident layers %d (%.2f GiB), streamed %.2f GiB\n", load_s,
                     static_cast<double>(eng.arena_used()) / (1024.0 * 1024.0), eng.n_resident_layers,
                     eng.resident_gib_actual, eng.streamed_gib);
        if (cfg.gpu_layers > 0)
            std::fprintf(stderr, "GPU: layers 0..%d, KV %s, arena %.1f MiB (weights %.1f MiB, KV %.1f MiB), device memory in use %.1f MiB\n",
                         cfg.gpu_layers - 1, cfg.kv_iso3 ? "iso3" : "f32", static_cast<double>(eng.gpu_arena_bytes()) / 1048576.0,
                         static_cast<double>(eng.gpu_weight_bytes()) / 1048576.0, static_cast<double>(eng.gpu_kv_bytes()) / 1048576.0,
                         eng.vram_used_mib());

        if (serve) {
            Generator gen(eng);
            return run_server(eng, gen, cfg, sopt);
        }

        std::vector<int32_t> ids = eng.tok.encode(prompt, parse_special);
        const int n_prompt = static_cast<int>(ids.size());
        std::vector<float> logits(static_cast<size_t>(eng.n_vocab));
        std::fprintf(stderr, "prompt tokens: %d\n", n_prompt);

        if (rollback_test) {
            // DeltaNet rollback test. Reference: greedy single-token steps g[0..5] with their logits. Then, for keep = 1, 2, 3:
            // fresh state, prefill, snapshot, one batch of g[0..3], rollback(keep), and the following single steps must give
            // logits that are bit-identical to the reference, and so must the logits rows of the batch itself.
            const size_t V = static_cast<size_t>(eng.n_vocab);
            const int NG = 6, TB = 4;
            std::vector<int32_t> g(NG);
            std::vector<float> ref(static_cast<size_t>(NG) * V), rows(static_cast<size_t>(TB) * V), lg(V);
            eng.forward(ids.data(), n_prompt, 0, logits.data());
            g[0] = argmax(logits.data(), eng.n_vocab);
            for (int i = 0; i < NG; ++i) {
                eng.forward(&g[i], 1, n_prompt + i, ref.data() + static_cast<size_t>(i) * V);
                if (i + 1 < NG) g[i + 1] = argmax(ref.data() + static_cast<size_t>(i) * V, eng.n_vocab);
            }
            int bad = 0;
            for (int keep = 1; keep <= 3; ++keep) {
                eng.reset();
                eng.forward(ids.data(), n_prompt, 0, lg.data());
                eng.dn_snapshot(true);
                eng.forward_ex(g.data(), TB, n_prompt, rows.data(), TB, nullptr);
                eng.dn_snapshot(false);
                size_t rows_diff = 0;
                for (int j = 0; j < TB; ++j)
                    rows_diff += std::memcmp(rows.data() + static_cast<size_t>(j) * V, ref.data() + static_cast<size_t>(j) * V, V * sizeof(float)) != 0;
                eng.dn_rollback(keep);
                int steps_diff = 0;
                for (int i = keep; i < NG; ++i) {
                    eng.forward(&g[i], 1, n_prompt + i, lg.data());
                    if (std::memcmp(lg.data(), ref.data() + static_cast<size_t>(i) * V, V * sizeof(float)) != 0) ++steps_diff;
                }
                std::printf("rollback keep=%d: batch rows differing from single-step logits: %zu of %d; steps after rollback differing: %d of %d\n",
                            keep, rows_diff, TB, steps_diff, NG - keep);
                bad += static_cast<int>(rows_diff) + steps_diff;
            }
            std::printf("rollback test: %s\n", bad ? "FAIL (not bit-identical)" : "PASS (bit-identical)");
            return bad ? 1 : 0;
        }

        // Optional per-layer parity against the reference dump (prefill pass only).
        struct LayerCmp { int il; double cos, mx; bool ok; };
        std::vector<LayerCmp> cmps;
        if (!parity_dir.empty()) {
            eng.tap = [&](int il, const float * x, int T) {
                std::vector<float> ref;
                const std::string path = parity_dir + "/l_out-" + std::to_string(il) + ".bin";
                if (!read_floats(path, ref)) { cmps.push_back({il, 0, 0, false}); return; }
                // The last layer keeps only the last token in the reference graph (one row of 5120 values).
                const bool last_row_only = ref.size() == 5120 && T > 1;
                if (!last_row_only && ref.size() != static_cast<size_t>(T) * 5120) { cmps.push_back({il, 0, 0, false}); return; }
                double mx;
                const double c = cosine(last_row_only ? x + static_cast<size_t>(T - 1) * 5120 : x, ref.data(), ref.size(), &mx);
                cmps.push_back({il, c, mx, true});
            };
        }

        Generator gen(eng);
        std::vector<float> pf_logits;
        if (!parity_dir.empty()) gp.capture_logits = &pf_logits;
        GenStats st;
        std::vector<int32_t> out_ids;
        gen.run(ids, gp, [&](int32_t t) { out_ids.push_back(t); return true; }, st);
        eng.tap = nullptr;

        int parity_fail = 0;
        double logits_cos = -1;
        bool top1_match = false;
        if (!parity_dir.empty()) {
            const std::vector<float> & lg = pf_logits;
            std::printf("\nper-layer parity (l_out after each layer, T=%d)\n  layer   cosine     max_abs\n", n_prompt);
            double worst = 1.0;
            int worst_il = -1;
            for (const auto & c : cmps) {
                if (!c.ok) { std::printf("  %3d   (no reference)\n", c.il); ++parity_fail; continue; }
                if (c.cos < worst) { worst = c.cos; worst_il = c.il; }
                if (c.il < 8 || c.il % 8 == 7 || c.cos < 0.995)
                    std::printf("  %3d   %.6f  %.3e%s\n", c.il, c.cos, c.mx, c.cos < 0.995 ? "   <-- below 0.995" : "");
                if (c.cos < 0.995) ++parity_fail;
            }
            std::printf("worst layer cosine %.6f at layer %d\n", worst, worst_il);
            std::vector<float> ref;
            if (read_floats(parity_dir + "/result_output.bin", ref) && ref.size() == lg.size()) {
                logits_cos = cosine(lg.data(), ref.data(), ref.size(), nullptr);
                // KL divergence D(ref || ours) over the softmax of the two logit vectors (Point A parity bar).
                {
                    double rmax = -INFINITY, omax = -INFINITY;
                    for (size_t i = 0; i < ref.size(); ++i) {
                        rmax = std::fmax(rmax, ref[i]);
                        omax = std::fmax(omax, lg[i]);
                    }
                    double lse_r = 0, lse_o = 0;
                    for (size_t i = 0; i < ref.size(); ++i) {
                        lse_r += std::exp(ref[i] - rmax);
                        lse_o += std::exp(lg[i] - omax);
                    }
                    lse_r = rmax + std::log(lse_r);
                    lse_o = omax + std::log(lse_o);
                    double kld = 0;
                    for (size_t i = 0; i < ref.size(); ++i) {
                        const double pr = std::exp(ref[i] - lse_r);
                        kld += pr * ((ref[i] - lse_r) - (lg[i] - lse_o));
                    }
                    std::printf("logits KLD(ref||ours) %.6g\n", kld);
                }
                top1_match = argmax(lg.data(), eng.n_vocab) == argmax(ref.data(), eng.n_vocab);
                std::vector<int> a(ref.size()), b(ref.size());
                for (size_t i = 0; i < a.size(); ++i) a[i] = b[i] = static_cast<int>(i);
                std::partial_sort(a.begin(), a.begin() + 5, a.end(), [&](int x, int y) { return lg[x] > lg[y]; });
                std::partial_sort(b.begin(), b.begin() + 5, b.end(), [&](int x, int y) { return ref[x] > ref[y]; });
                std::printf("logits cosine %.6f, top-1 %s (ours %d vs ref %d)\n", logits_cos, top1_match ? "MATCH" : "DIFFER", a[0], b[0]);
                std::printf("top-5 ours:");
                for (int i = 0; i < 5; ++i) std::printf(" %d(%.2f)", a[i], lg[a[i]]);
                std::printf("\ntop-5 ref: ");
                for (int i = 0; i < 5; ++i) std::printf(" %d(%.2f)", b[i], ref[b[i]]);
                std::printf("\n");
            } else {
                std::printf("no result_output.bin reference\n");
                ++parity_fail;
            }
        }

        const std::string text = eng.tok.decode(out_ids, false);
        std::printf("\nprompt: %s\ngenerated: %s\n", prompt.c_str(), text.c_str());
        const int D = st.mtp_steps;
        const double decode_tok_s = st.decode_s > 0 ? static_cast<double>(st.generated) / st.decode_s : 0.0;
        std::printf("{\"prompt_tokens\":%d,\"generated_tokens\":%zu,\"ids\":[", n_prompt, out_ids.size());
        for (size_t i = 0; i < out_ids.size(); ++i) std::printf("%s%d", i ? "," : "", out_ids[i]);
        std::printf("],\"text\":\"%s\",\"load_s\":%.2f,\"prefill_s\":%.2f,\"prefill_tok_s\":%.3f,\"decode_s\":%.2f,"
                    "\"decode_tok_s\":%.4f,\"heap_allocs_in_decode_forward\":%llu,\"peak_rss_kb\":%lld,\"rss_anon_kb\":%lld,\"rss_file_kb\":%lld,"
                    "\"swap_used_kb_before\":%lld,\"swap_used_kb_after\":%lld,\"parity_failures\":%d,\"logits_cos\":%.6f,"
                    "\"top1_match\":%s,\"gpu_layers\":%d,\"gpu_arena_mib\":%.1f,\"gpu_weights_mib\":%.1f,\"gpu_kv_mib\":%.1f,\"vram_used_mib\":%.1f,\"kv\":\"%s\",\"dp4a_mask\":%u,\"decode_gpu_s\":%.3f,\"decode_cpu_layers_s\":%.3f,\"decode_head_s\":%.3f,"
                    "\"mtp_steps\":%d,\"cycles\":%lld,\"draft_n\":%lld,\"draft_n_accepted\":%lld,\"accept_hist\":[%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld],\"mtp_adaptive\":%d,"
                    "\"mtp_prefill_s\":%.3f,\"prefill_main_s\":%.3f,\"mtp_draft_s\":%.3f,\"mtp_verify_s\":%.3f,\"mtp_rollback_s\":%.3f,\"stream_min_t\":%d,"
                    "\"pld_enabled\":%d,\"pld_hits\":%lld,\"pld_candidates\":%lld,\"mtp_fills\":%lld,"
                    "\"ttft_s\":%.3f,\"finish\":\"%s\",\"temperature\":%.3f}\n",
                    json_escape(text).c_str(), load_s, st.prefill_s, n_prompt / std::max(st.prefill_s, 1e-9), st.decode_s,
                    decode_tok_s,
                    static_cast<unsigned long long>(st.heap_allocs), peak_rss_kb(), status_kb("RssAnon:"), status_kb("RssFile:"), swap0, swap_used_kb(), parity_fail,
                    logits_cos, top1_match ? "true" : "false", cfg.gpu_layers,
                    static_cast<double>(eng.gpu_arena_bytes()) / 1048576.0, static_cast<double>(eng.gpu_weight_bytes()) / 1048576.0,
                    static_cast<double>(eng.gpu_kv_bytes()) / 1048576.0, cfg.gpu_layers > 0 ? eng.vram_used_mib() : 0.0,
                    cfg.kv_iso3 ? "iso3" : "f32", cfg.dp4a_mask, st.dec_gpu, st.dec_cpu, st.dec_head,
                    D, st.cycles, st.draft_n, st.draft_acc,
                    st.accept_hist[0], st.accept_hist[1], st.accept_hist[2], st.accept_hist[3], st.accept_hist[4],
                    st.accept_hist[5], st.accept_hist[6], st.accept_hist[7], st.accept_hist[8],
                    gp.mtp_adaptive ? 1 : 0,
                    st.mtp_prefill_s, st.prefill_main_s, st.t_draft, st.t_verify, st.t_roll, cfg.stream_min_t,
                    gp.pld_enabled ? 1 : 0, st.pld_hits, st.pld_candidates, st.mtp_fills,
                    st.ttft_s, st.finish.c_str(), static_cast<double>(gp.sp.temperature));
        return parity_fail ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
