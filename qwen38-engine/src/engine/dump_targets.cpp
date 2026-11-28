// --dump-targets: native PTQ1_0 teacher-free target precompute (see dump_targets.hpp).
// Same package format as tools/bonsai_mtp_train/precompute_teacher_targets.py: per position the token ids,
// the INT4 group-64 (or INT8) output-normed hidden, the sparse top-K logit ids/vals (FP16) and the FP16
// logsumexp over the full vocabulary. Engine shards are plain .npy directories (the Python loader accepts
// the old .npz shards and these), one meta.json per shard, one atomic manifest.json for the package.
//
// Per sample: reset() -> one full prefill (forward_ex, the output-normed hidden of every position lands in
// host h_buf) -> head in kHeadChunk-row batches: on a GPU head the top-K/lse reduce ON DEVICE (head_topk
// kernel; only K*6+2 bytes per row cross D2H) and the hidden quantizes inline on the CPU (41K ops per
// 8-row chunk — negligible); a CPU head falls back to head_rows + a host heap reduction. Per shard the
// timing breakdown (prefill/head/quant/io ms) is printed so a slow component is visible at a glance.
#include "dump_targets.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

#include "../common/f16bits.hpp"
#include "../cpu/dn_layer.hpp"   // kHidden, kFFN (the constructor expect()s every other dim from the GGUF)
#include "engine.hpp"

namespace {

using Clk = std::chrono::steady_clock;

constexpr int64_t kGroup = 64;   // hidden quantization group: 5120 / 64 = 80 scales per token (GROUP in the Python tools)
constexpr float kScaleFloor = 1e-4f;   // INT4_SCALE_FLOOR of the Python tools
constexpr int kHeadChunk = 8;    // rows per head_rows/head_rows_topk call (GPU head cap)

std::string base_name(const std::string & p) {
    const size_t k = p.find_last_of('/');
    return k == std::string::npos ? p : p.substr(k + 1);
}

bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return true;
}

void write_file_atomic(const std::string & path, const std::string & data) {
    const std::string tmp = path + ".tmp";
    { std::ofstream f(tmp, std::ios::binary | std::ios::trunc); f.write(data.data(), std::streamsize(data.size())); }
    std::remove(path.c_str());
    std::rename(tmp.c_str(), path.c_str());
}

// --- plan.json: {"seq_len": N, "items": [{"off": N, "len": N, ...}, ...]} — a tiny scanner for our own writer.
struct PlanItem { int64_t off = 0, len = 0; };

int64_t json_number_after(const std::string & s, const std::string & key, size_t from) {
    const size_t p = s.find(key, from);
    if (p == std::string::npos) return -1;
    const size_t d = s.find_first_of("-0123456789", p + key.size());
    if (d == std::string::npos) return -1;
    return std::strtoll(s.c_str() + d, nullptr, 10);
}

void parse_plan(const std::string & path, int & seq_len, std::vector<PlanItem> & items) {
    std::string s;
    if (!read_file(path, s)) throw std::runtime_error("cannot read plan json " + path);
    seq_len = static_cast<int>(json_number_after(s, "\"seq_len\"", 0));
    if (seq_len <= 0) throw std::runtime_error("plan.json has no seq_len");
    const size_t arr = s.find("\"items\"");
    if (arr != std::string::npos) {
        size_t p = s.find('[', arr);
        while (p != std::string::npos) {
            const int64_t off = json_number_after(s, "\"off\"", p);
            const int64_t len = json_number_after(s, "\"len\"", p);
            if (off < 0 || len < 0) break;
            items.push_back({off, len});
            const size_t close = s.find('}', p);
            if (close == std::string::npos) break;
            p = close + 1;
            if (p < s.size() && s[p] == ']') break;
        }
    }
    if (items.empty()) throw std::runtime_error("plan.json has no items");
}

// --- .npy v1.0 writer (plain little-endian host arrays; dtypes: <i4, |u1, |i1, <f2, <f4). Returns bytes written.
int64_t npy_bytes(const std::vector<int64_t> & shape, const char * descr) {
    const int64_t elems = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    const size_t b = std::strcmp(descr, "<i4") == 0 ? 4 : (std::strcmp(descr, "|u1") == 0 || std::strcmp(descr, "|i1") == 0) ? 1 :
                     (std::strcmp(descr, "<f2") == 0 ? 2 : 4);
    return elems * static_cast<int64_t>(b);
}

void write_npy(const std::string & path, const void * data, const std::vector<int64_t> & shape, const char * descr) {
    std::string sh;
    for (size_t i = 0; i < shape.size(); ++i) sh += (i ? ", " : "") + std::to_string(shape[i]);
    if (shape.size() == 1) sh += ",";
    std::string dict = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': (" + sh + "), }";
    const size_t pad = (64 - ((10 + dict.size()) % 64)) % 64;
    dict.append(pad, ' ');
    dict += '\n';
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write("\x93NUMPY\x01\x00", 8);
    const uint16_t hl = static_cast<uint16_t>(dict.size());
    f.write(reinterpret_cast<const char *>(&hl), 2);
    f.write(dict.data(), std::streamsize(dict.size()));
    f.write(reinterpret_cast<const char *>(data), std::streamsize(npy_bytes(shape, descr)));
    if (!f) throw std::runtime_error("short write " + path);
}

// --- per-row math -----------------------------------------------------------

// Symmetric group-64 quantizer, byte-identical layout to quantize_hidden() in precompute_teacher_targets.py:
// scale = max(|x|_inf / qmax, floor) stored as FP16, codes = clip(round-to-even(x / scale), -qmax-1, qmax).
void quant_row(const float * x, bool int4, uint8_t * hq, uint16_t * hs) {
    constexpr int64_t G = kHidden / kGroup;
    const float qmax = int4 ? 7.0f : 127.0f;
    const int lo_q = int4 ? -8 : -128;
    for (int64_t g = 0; g < G; ++g) {
        const float * p = x + g * kGroup;
        float amax = 0;
        for (int64_t i = 0; i < kGroup; ++i) amax = std::fmax(amax, std::fabs(p[i]));
        const float scale = std::fmax(amax / qmax, kScaleFloor);
        hs[g] = f32_to_f16(scale);
        const float inv = 1.0f / scale;
        if (int4) {
            uint8_t * out = hq + g * (kGroup / 2);
            for (int64_t i = 0; i < kGroup; i += 2) {
                const int a = std::min(std::max(static_cast<int>(std::nearbyint(p[i] * inv)), lo_q), 7);
                const int b = std::min(std::max(static_cast<int>(std::nearbyint(p[i + 1] * inv)), lo_q), 7);
                out[i / 2] = static_cast<uint8_t>((a & 0x0F) | ((b & 0x0F) << 4));   // even element = low nibble
            }
        } else {
            int8_t * out = reinterpret_cast<int8_t *>(hq) + g * kGroup;
            for (int64_t i = 0; i < kGroup; ++i)
                out[i] = static_cast<int8_t>(std::min(std::max(static_cast<int>(std::nearbyint(p[i] * inv)), lo_q), 127));
        }
    }
}

// CPU fallback only (the GPU head path reduces on-device): full-vocabulary logsumexp (FP32 math, FP16 out)
// + top-K ids/vals descending, for one row of head logits.
void topk_row(const float * z, int64_t V, int K, int32_t * ids, uint16_t * vals, uint16_t * lse_out) {
    float m = -INFINITY;
    for (int64_t i = 0; i < V; ++i) m = std::fmax(m, z[i]);
    double s = 0;
    for (int64_t i = 0; i < V; ++i) s += std::exp(static_cast<double>(z[i]) - m);
    const float lse = m + static_cast<float>(std::log(s));
    std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, std::greater<>> pq;
    for (int64_t i = 0; i < V; ++i) {
        if (static_cast<int>(pq.size()) < K) pq.push({z[i], i});
        else if (z[i] > pq.top().first) { pq.pop(); pq.push({z[i], i}); }
    }
    for (int k = K - 1; k >= 0; --k) {   // pop = ascending; store descending like torch.topk
        ids[k] = static_cast<int32_t>(pq.top().second);
        vals[k] = f32_to_f16(pq.top().first);
        pq.pop();
    }
    lse_out[0] = f32_to_f16(lse);
}

// Per-shard timing breakdown (ms) — printed at every shard end so a slow component is immediately visible.
struct ShardProf {
    double prefill_ms = 0, head_ms = 0, quant_ms = 0, io_ms = 0;
    double total_ms() const { return prefill_ms + head_ms + quant_ms + io_ms; }
};

double ms_since(Clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
}

}  // namespace

int run_dump(EngineConfig cfg, const DumpConfig & dc) {
    if (dc.tokens_bin.empty() || dc.targets_out.empty())
        throw std::runtime_error("--dump-targets needs --tokens-bin and --targets-out");
    if (dc.hidden_dtype != "int4g64" && dc.hidden_dtype != "int8")
        throw std::runtime_error("--hidden-dtype must be int4g64 or int8");
    const bool int4 = dc.hidden_dtype == "int4g64";

    // Plan first: it fixes seq_len, which fixes ctx_max / t_max BEFORE the engine allocates anything.
    const std::string plan = dc.plan_json.empty()
        ? dc.tokens_bin.substr(0, dc.tokens_bin.find_last_of("/\\")) + "/plan.json" : dc.plan_json;
    int plan_len = 0;
    std::vector<PlanItem> plan_items;
    parse_plan(plan, plan_len, plan_items);
    const int seq_len = dc.seq_len > 0 ? dc.seq_len : plan_len;
    const int n_samples = dc.samples > 0 ? std::min<int>(dc.samples, static_cast<int>(plan_items.size()))
                                         : static_cast<int>(plan_items.size());
    for (const PlanItem & it : plan_items)
        if (it.len < seq_len) throw std::runtime_error("plan sample shorter than seq_len (re-run tokenize_for_engine.py)");
    std::fprintf(stderr, "dump: plan %s -> %d samples x %d tokens, top_k %d, hidden %s\n",
                 plan.c_str(), n_samples, seq_len, dc.top_k, dc.hidden_dtype.c_str());

    // Native residency: every layer and the LM head on the GPU (the PTQ1_0 27B is 5.54 GiB — it sits whole in a
    // T4 15 GiB; a card that does not fit it needs a higher --vram-budget-mib, not layer streaming).
    cfg.ctx_max = seq_len;
    cfg.t_max = seq_len;
    cfg.gpu_layers = 64;                    // dump mode overrides --gpu-layers: partial residency loses the GPU head
    cfg.gpu_head = 1;                       // the dump head is GPU-first; a CPU fallback exists but is slow
    cfg.mtp_steps = 0;
    cfg.f16_gemm = true;                    // sm_75+ FP16 Tensor Core GEMM (T4/Ampere/Ada); refused elsewhere

    const auto t_load0 = Clk::now();
    Engine eng(cfg);
    const double load_s = std::chrono::duration<double>(Clk::now() - t_load0).count();
    std::fprintf(stderr, "loaded in %.2f s; gpu_layers %d, gpu head %d, vram used %.1f MiB\n", load_s,
                 eng.config().gpu_layers, eng.head_ready() ? 1 : 0,
                 eng.config().gpu_layers > 0 ? eng.vram_used_mib() : 0.0);
    if (!eng.head_ready())
        std::fprintf(stderr, "WARNING: LM head not on the GPU - the dump runs the slow CPU head + host top-K\n");

    std::string tok_bytes;
    if (!read_file(dc.tokens_bin, tok_bytes)) throw std::runtime_error("cannot read tokens bin " + dc.tokens_bin);
    {
        const int64_t n_tok_file = static_cast<int64_t>(tok_bytes.size()) / 4;
        int64_t last_end = 0;
        for (int i = 0; i < n_samples; ++i) last_end = std::max(last_end, plan_items[i].off + plan_items[i].len);
        if (last_end > n_tok_file) throw std::runtime_error("tokens.bin shorter than the plan offsets");
    }
    const int32_t * toks = reinterpret_cast<const int32_t *>(tok_bytes.data());

    // package layout: shard dirs target-shard-XXXXX/ with ids/hq/hs/tk_ids/tk_vals/tk_lse (+hf32) .npy + meta.json
    const int SS = dc.shard_samples;
    const int64_t L = seq_len, V = eng.n_vocab, K = dc.top_k;
    if (K < 1 || K > 256) throw std::runtime_error("--dump-top-k must be 1..256 (head_topk staging)");
    if (SS < 1 || SS > n_samples) throw std::runtime_error("--shard-samples out of range");
    const int64_t tok_per_shard = static_cast<int64_t>(SS) * L;
    std::vector<int32_t> sh_ids(tok_per_shard);
    std::vector<uint8_t> sh_hq(tok_per_shard * (kHidden / (int4 ? 2 : 1)));
    std::vector<uint16_t> sh_hs(tok_per_shard * (kHidden / kGroup));
    std::vector<int32_t> sh_tk(tok_per_shard * K);
    std::vector<uint16_t> sh_tv(tok_per_shard * K), sh_tl(tok_per_shard);
    std::vector<float> sh_f32(dc.dump_f32_hidden ? tok_per_shard * kHidden : 0);
    const int64_t kv_unit = cfg.kv_iso3 ? 400 : 1024 * 4;   // bytes/position/tensor: 8 iso3 half-blocks of 50 B (or FP32 KVW)
    std::vector<uint8_t> sh_k63(dc.kv63 ? tok_per_shard * kv_unit : 0);
    std::vector<uint8_t> sh_v63(dc.kv63 ? tok_per_shard * kv_unit : 0);
    std::vector<float> h_buf(static_cast<int64_t>(L) * kHidden);
    std::vector<float> lg(static_cast<int64_t>(kHeadChunk) * V);   // CPU-head fallback only
    std::vector<int32_t> tk_buf(static_cast<int64_t>(kHeadChunk) * K);
    std::vector<uint16_t> tv_buf(static_cast<int64_t>(kHeadChunk) * K), tl_buf(kHeadChunk);

    if (std::system(("mkdir -p " + dc.targets_out).c_str()) != 0)
        throw std::runtime_error("cannot create " + dc.targets_out);
    {
        // dims.json for the trainer's Dims(**json) — the constructor just enforced every value against the GGUF
        const std::string dims_p = dc.targets_out + "/dims.json";
        std::string dhave;
        if (!read_file(dims_p, dhave)) {
            char djson[512];
            std::snprintf(djson, sizeof djson,
                          "{\"hidden\": %lld, \"n_head\": 24, \"n_kv\": 4, \"head_dim\": 256, \"rope_dim\": 64, "
                          "\"ffn\": %lld, \"theta\": 10000000.0, \"eps\": 1e-06}",
                          static_cast<long long>(kHidden), static_cast<long long>(kFFN));
            write_file_atomic(dims_p, djson);
        }
    }

    // auto-resume: count consecutive complete shard dirs (meta.json present)
    int done_shards = 0;
    for (;; ++done_shards) {
        char p[512];
        std::snprintf(p, sizeof p, "%s/target-shard-%05d/meta.json", dc.targets_out.c_str(), done_shards);
        std::string s;
        if (!read_file(p, s)) break;
    }
    const int total_shards = (n_samples + SS - 1) / SS;
    if (done_shards > 0)
        std::fprintf(stderr, "resume: %d of %d shards already on disk\n", done_shards, total_shards);

    double bytes = 0;
    const auto t0 = Clk::now();
    int64_t done_tok = 0;
    ShardProf all_prof;
    for (int shard = done_shards; shard < total_shards; ++shard) {
        const int64_t rows_all = static_cast<int64_t>(total_shards - shard) * tok_per_shard;
        const double est = npy_bytes({tok_per_shard}, "<i4") + npy_bytes({tok_per_shard, kHidden / (int4 ? 2 : 1)}, "|u1") +
                           npy_bytes({tok_per_shard, kHidden / kGroup}, "<f2") + npy_bytes({tok_per_shard, K}, "<i4") +
                           npy_bytes({tok_per_shard, K}, "<f2") + npy_bytes({tok_per_shard}, "<f2") +
                           npy_bytes({tok_per_shard, kHidden}, "<f4") * (dc.dump_f32_hidden ? 1 : 0) +
                           npy_bytes({tok_per_shard, static_cast<int64_t>(kv_unit)}, "|u1") * (dc.kv63 ? 2 : 0);
        if (dc.max_gb > 0 && bytes + est > dc.max_gb * 1e9) {
            std::fprintf(stderr, "--max-gb %.0f reached after %d shards (%.2f GB); manifest is consistent\n",
                         dc.max_gb, shard, bytes / 1e9);
            break;
        }
        const int s0 = shard * SS;
        const int rows_n = std::min(SS, n_samples - s0);
        const int64_t rows_tok = static_cast<int64_t>(rows_n) * L;
        for (int r = 0; r < rows_n; ++r)   // ids need no forward pass
            std::memcpy(sh_ids.data() + static_cast<int64_t>(r) * L, toks + plan_items[s0 + r].off,
                        sizeof(int32_t) * static_cast<size_t>(L));
        const auto shard_t0 = Clk::now();
        ShardProf prof;
        for (int r = 0; r < rows_n; ++r) {
            const PlanItem & it = plan_items[s0 + r];
            const auto tp0 = Clk::now();
            eng.reset();   // clean KV/DeltaNet state per sample (positions restart at 0)
            eng.forward_ex(toks + it.off, static_cast<int>(L), 0, nullptr, 0, h_buf.data());
            prof.prefill_ms += ms_since(tp0);
            if (dc.kv63) {   // issue #70 P1-rev: the teacher's layer-63 K/V exactly as the engine caches them
                eng.copy_layer_kv(63, L, sh_k63.data() + static_cast<int64_t>(r) * L * kv_unit,
                                  sh_v63.data() + static_cast<int64_t>(r) * L * kv_unit);
            }

            const int64_t dst_row0 = static_cast<int64_t>(r) * L;
            for (int c0 = 0; c0 < L; c0 += kHeadChunk) {
                const int n = static_cast<int>(std::min<int64_t>(kHeadChunk, L - c0));
                float * hc = h_buf.data() + static_cast<int64_t>(c0) * kHidden;
                const int64_t dst0 = dst_row0 + c0;
                const auto th0 = Clk::now();
                const bool gpu_topk = eng.head_rows_topk(hc, n, static_cast<int>(K), tk_buf.data(),
                                                         tv_buf.data(), tl_buf.data());
                if (!gpu_topk) {
                    eng.head_rows(hc, n, lg.data());
                    for (int i = 0; i < n; ++i)
                        topk_row(lg.data() + static_cast<int64_t>(i) * V, V, static_cast<int>(K),
                                 tk_buf.data() + static_cast<int64_t>(i) * K,
                                 tv_buf.data() + static_cast<int64_t>(i) * K, tl_buf.data() + i);
                }
                prof.head_ms += ms_since(th0);
                const auto tq0 = Clk::now();
                std::memcpy(sh_tk.data() + dst0 * K, tk_buf.data(), sizeof(int32_t) * static_cast<size_t>(n) * static_cast<size_t>(K));
                std::memcpy(sh_tv.data() + dst0 * K, tv_buf.data(), sizeof(uint16_t) * static_cast<size_t>(n) * static_cast<size_t>(K));
                std::memcpy(sh_tl.data() + dst0, tl_buf.data(), sizeof(uint16_t) * static_cast<size_t>(n));
                const int64_t hq_row = kHidden / (int4 ? 2 : 1), hs_row = kHidden / kGroup;
                for (int i = 0; i < n; ++i)
                    quant_row(hc + static_cast<int64_t>(i) * kHidden, int4,
                              sh_hq.data() + (dst0 + i) * hq_row, sh_hs.data() + (dst0 + i) * hs_row);
                if (dc.dump_f32_hidden)
                    std::memcpy(sh_f32.data() + dst0 * kHidden, hc, sizeof(float) * static_cast<size_t>(n) * static_cast<size_t>(kHidden));
                prof.quant_ms += ms_since(tq0);
            }
            done_tok += L;
            // Fixed ETA: remaining tokens / measured rate (the old line divided tokens by elapsed seconds).
            if ((r + 1) % 16 == 0 || r + 1 == rows_n) {
                const double dt = std::chrono::duration<double>(Clk::now() - shard_t0).count();
                const double rate = static_cast<double>(r + 1) * static_cast<double>(L) / std::max(dt, 1e-9);
                const double remain_shard = static_cast<double>(rows_n - r - 1) * static_cast<double>(L);
                const double all_dt = std::chrono::duration<double>(Clk::now() - t0).count();
                const double rate_all = static_cast<double>(done_tok) / std::max(all_dt, 1e-9);
                const double remain_all = static_cast<double>(rows_all - done_tok);
                std::fprintf(stderr,
                             "shard %d/%d sample %d/%d: %.1f tok/s (shard window), eta %.1f min (shard), "
                             "%.1f tok/s (all), eta %.1f min (all)\n",
                             shard + 1, total_shards, r + 1, rows_n, rate, rate > 0 ? remain_shard / rate / 60.0 : 0.0,
                             rate_all, rate_all > 0 ? std::max(remain_all, 0.0) / rate_all / 60.0 : 0.0);
            }
        }
        // write the shard directory atomically: target-shard-XXXXX.tmp -> target-shard-XXXXX
        const auto tio0 = Clk::now();
        char dir[512];
        std::snprintf(dir, sizeof dir, "%s/target-shard-%05d", dc.targets_out.c_str(), shard);
        const std::string tmpd = std::string(dir) + ".tmp";
        if (std::system(("rm -rf " + tmpd + " && mkdir -p " + tmpd).c_str()) != 0)
            throw std::runtime_error("cannot prepare " + tmpd);
        double sh_bytes = 0;
        auto wr = [&](const char * name, const void * d, std::vector<int64_t> shape, const char * descr) {
            const std::string p = tmpd + "/" + name;
            write_npy(p, d, shape, descr);
            sh_bytes += static_cast<double>(npy_bytes(shape, descr));
        };
        wr("ids.npy", sh_ids.data(), {rows_n, L}, "<i4");
        wr("hq.npy", sh_hq.data(), {rows_n, L, kHidden / (int4 ? 2 : 1)}, int4 ? "|u1" : "|i1");
        wr("hs.npy", sh_hs.data(), {rows_n, L, kHidden / kGroup}, "<f2");
        wr("tk_ids.npy", sh_tk.data(), {rows_n, L, K}, "<i4");
        wr("tk_vals.npy", sh_tv.data(), {rows_n, L, K}, "<f2");
        wr("tk_lse.npy", sh_tl.data(), {rows_n, L}, "<f2");
        if (dc.dump_f32_hidden) wr("hf32.npy", sh_f32.data(), {rows_n, L, kHidden}, "<f4");
        if (dc.kv63) {
            wr("kv63_k.npy", sh_k63.data(), {rows_n, L, kv_unit}, "|u1");
            wr("kv63_v.npy", sh_v63.data(), {rows_n, L, kv_unit}, "|u1");
        }
        {
            char meta[832];
            std::snprintf(meta, sizeof meta,
                          "{\"version\": 1, \"format\": \"engine-npy-dir-1\", \"hidden_dtype\": \"%s\", "
                          "\"top_k\": %d, \"hidden\": %lld, \"seq_len\": %lld, \"samples\": %d, \"group\": %lld, "
                          "\"kv63\": %s, \"kv_unit\": %lld, \"index\": %d, \"model\": \"%s\"}",
                          dc.hidden_dtype.c_str(), static_cast<int>(K), static_cast<long long>(kHidden),
                          static_cast<long long>(L), rows_n, static_cast<long long>(kGroup),
                          dc.kv63 ? "true" : "false", static_cast<long long>(kv_unit), shard,
                          base_name(cfg.gguf).c_str());
            write_file_atomic(std::string(tmpd) + "/meta.json", meta);
        }
        if (std::rename(tmpd.c_str(), dir) != 0) throw std::runtime_error("shard rename failed for " + std::string(dir));
        bytes += sh_bytes;
        prof.io_ms = ms_since(tio0);
        {
            char man[1024];
            std::snprintf(man, sizeof man,
                          "{\"version\": 1, \"format\": \"engine-npy-dir-1\", \"model\": \"%s\", "
                          "\"hidden_dtype\": \"%s\", \"top_k\": %d, \"hidden\": %lld, \"seq_len\": %lld, "
                          "\"plan_samples\": %d, \"shards\": %d, \"total_samples\": %d, \"total_tokens\": %lld, "
                          "\"total_bytes\": %.0f}",
                          base_name(cfg.gguf).c_str(), dc.hidden_dtype.c_str(), static_cast<int>(K),
                          static_cast<long long>(kHidden), static_cast<long long>(L), n_samples, shard + 1,
                          std::min<int>((shard + 1) * SS, n_samples),
                          static_cast<long long>(std::min<int64_t>((shard + 1) * SS, n_samples)) * L, bytes);
            write_file_atomic(dc.targets_out + "/manifest.json", man);
        }
        const double dt = std::chrono::duration<double>(Clk::now() - shard_t0).count();
        const double all_dt = std::chrono::duration<double>(Clk::now() - t0).count();
        const double all_rate = done_tok / std::max(all_dt, 1e-9);
        all_prof.prefill_ms += prof.prefill_ms;
        all_prof.head_ms += prof.head_ms;
        all_prof.quant_ms += prof.quant_ms;
        all_prof.io_ms += prof.io_ms;
        // Head GEMM throughput estimate: rows_tok * 2 * V * kHidden FLOPs through the head path.
        const double head_gflops = static_cast<double>(rows_tok) * 2.0 * static_cast<double>(V) *
                                   static_cast<double>(kHidden) / (prof.head_ms * 1e6);
        std::fprintf(stderr,
                     "[dump] shard %d/%d done: %d samples, %.0f tok/s (shard), %.0f tok/s (all), eta %.1f min, %.2f GB\n"
                     "       breakdown: prefill %.0f ms, head %.0f ms (~%.0f GFLOPS), quant %.0f ms, io %.0f ms\n",
                     shard + 1, total_shards, rows_n, rows_tok / std::max(dt, 1e-9), all_rate,
                     all_rate > 0 ? std::max(static_cast<double>(rows_all) - static_cast<double>(done_tok), 0.0) / all_rate / 60.0 : 0.0,
                     bytes / 1e9,
                     prof.prefill_ms, prof.head_ms, head_gflops, prof.quant_ms, prof.io_ms);
    }

    const double total_s = std::chrono::duration<double>(Clk::now() - t0).count();
    std::printf("{\"dump_targets\": \"%s\", \"samples\": %d, \"shards\": %d, \"tokens\": %lld, \"tok_s\": %.1f, "
                "\"elapsed_s\": %.1f, \"load_s\": %.2f, \"package_gb\": %.2f, \"vram_used_mib\": %.1f, "
                "\"gpu_layers\": %d, \"top_k\": %d, \"hidden_dtype\": \"%s\", "
                "\"prefill_ms\": %.0f, \"head_ms\": %.0f, \"quant_ms\": %.0f, \"io_ms\": %.0f}\n",
                dc.targets_out.c_str(), n_samples, total_shards, static_cast<long long>(done_tok),
                done_tok / std::max(total_s, 1e-9), total_s, load_s, bytes / 1e9,
                cfg.gpu_layers > 0 ? eng.vram_used_mib() : 0.0, eng.config().gpu_layers, static_cast<int>(K),
                dc.hidden_dtype.c_str(), all_prof.prefill_ms, all_prof.head_ms, all_prof.quant_ms, all_prof.io_ms);
    return 0;
}
