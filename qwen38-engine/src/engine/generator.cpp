#include "generator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "arena.hpp"

namespace {
using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
constexpr int H = 5120;

int argmax(const float * v, int64_t n) {
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
    return static_cast<int>(best);
}

// Softmax probability of token `best` (the argmax) over n logits.
float top1_prob(const float * v, int64_t n, int best) {
    const float mx = v[best];
    double s = 0;
    for (int64_t i = 0; i < n; ++i) s += std::exp(static_cast<double>(v[i] - mx));
    return static_cast<float>(1.0 / s);
}

// Entropy of the softmax distribution (nats). High entropy = low confidence.
float entropy(const float * v, int64_t n) {
    float mx = v[0];
    for (int64_t i = 1; i < n; ++i) if (v[i] > mx) mx = v[i];
    double sum = 0;
    for (int64_t i = 0; i < n; ++i) sum += std::exp(static_cast<double>(v[i] - mx));
    double ent = 0;
    for (int64_t i = 0; i < n; ++i) {
        double p = std::exp(static_cast<double>(v[i] - mx)) / sum;
        if (p > 0) ent -= p * std::log(p);
    }
    return static_cast<float>(ent);
}

// PLD (Prompt Lookup Decoding) — fixed open-address hash table for 3-gram and 4-gram suffixes.
// Maps suffix hash -> vector of next-token candidates (deterministic, from prompt + generated history).
struct PLDMatcher {
    static constexpr uint32_t EMPTY = 0xFFFFFFFF;
    struct Entry { uint32_t key; int32_t pos; int32_t pos2; };   // pos2 = previous occurrence (-1)

    std::vector<Entry> table3;   // 3-gram table
    std::vector<Entry> table4;   // 4-gram table
    uint32_t mask3, mask4;
    const std::vector<int32_t> * hist = nullptr;   // history walked forward on a match

    PLDMatcher(size_t expected_tokens = 0) { init(expected_tokens); }

    void init(size_t expected_tokens) {
        size_t cap3 = 1;
        while (cap3 < expected_tokens * 2) cap3 <<= 1;
        size_t cap4 = 1;
        while (cap4 < expected_tokens * 2) cap4 <<= 1;
        table3.assign(cap3, Entry{EMPTY, -1, -1});
        table4.assign(cap4, Entry{EMPTY, -1, -1});
        mask3 = static_cast<uint32_t>(cap3 - 1);
        mask4 = static_cast<uint32_t>(cap4 - 1);
    }

    static uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
        uint64_t h = (static_cast<uint64_t>(a) * 1000003) ^ (static_cast<uint64_t>(b) * 1000033) ^ (static_cast<uint64_t>(c) * 1000037);
        return static_cast<uint32_t>((h ^ (h >> 32)) * 11400714819323198485ull);
    }
    static uint32_t hash4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        uint64_t h = (static_cast<uint64_t>(a) * 1000003) ^ (static_cast<uint64_t>(b) * 1000033) ^ (static_cast<uint64_t>(c) * 1000037) ^ (static_cast<uint64_t>(d) * 1000039);
        return static_cast<uint32_t>((h ^ (h >> 32)) * 11400714819323198485ull);
    }

    void attach(const std::vector<int32_t> * h) { hist = h; }

    // Register the suffix that ends at tokens[pos] (its continuation lives at tokens[pos+1 ...]).
    void add(const int32_t * tokens, int pos) {
        if (pos >= 2) {
            uint32_t h = hash3(tokens[pos-2], tokens[pos-1], tokens[pos]) & mask3;
            for (;; h = (h + 1) & mask3) {
                Entry & e = table3[h];
                if (e.key == EMPTY) { e.key = h; e.pos = pos; return; }
                if (e.key == h) { e.pos2 = e.pos; e.pos = pos; return; }
            }
        }
        if (pos >= 3) {
            uint32_t h = hash4(tokens[pos-3], tokens[pos-2], tokens[pos-1], tokens[pos]) & mask4;
            for (;; h = (h + 1) & mask4) {
                Entry & e = table4[h];
                if (e.key == EMPTY) { e.key = h; e.pos = pos; return; }
                if (e.key == h) { e.pos2 = e.pos; e.pos = pos; return; }
            }
        }
    }

    // Locate the suffix (its last token is suffix[suffix_len]) and copy the historical continuation.
    // The match is only trusted when the same continuation already followed an earlier occurrence of
    // the suffix (pos2): that marks a genuine repeat of the pattern instead of a one-off hit, so PLD
    // replaces MTP only where it is likely to pay off. Returns the token count written to `out`.
    int query(const int32_t * suffix, int suffix_len, int32_t * out, int max_out) const {
        if (!hist || max_out < 1) return 0;
        const int hs = static_cast<int>(hist->size());
        int32_t matched = -1;
        if (suffix_len >= 3) {   // 4-gram key (more specific) first
            uint32_t h = hash4(suffix[suffix_len-3], suffix[suffix_len-2], suffix[suffix_len-1], suffix[suffix_len]) & mask4;
            for (int probe = 0; probe < 16; ++probe, h = (h + 1) & mask4) {
                const Entry & e = table4[h];
                if (e.key == EMPTY) break;
                if (e.key == h) { if (valid_repeat(e.pos, e.pos2, hs)) matched = e.pos; break; }
            }
        }
        if (matched < 0 && suffix_len >= 2) {   // 3-gram fallback
            uint32_t h = hash3(suffix[suffix_len-2], suffix[suffix_len-1], suffix[suffix_len]) & mask3;
            for (int probe = 0; probe < 16; ++probe, h = (h + 1) & mask3) {
                const Entry & e = table3[h];
                if (e.key == EMPTY) break;
                if (e.key == h) { if (valid_repeat(e.pos, e.pos2, hs)) matched = e.pos; break; }
            }
        }
        if (matched < 0) return 0;
        int written = 0;
        for (int p = matched + 1; p < hs && written < max_out; ++p)
            out[written++] = (*hist)[p];
        return written;
    }

    // A match is trusted when the previous occurrence continued identically for at least
    // kMinRepeat tokens: that marks a repeated multi-token block (where PLD pays off) instead of a
    // one-token coincidence (where the MTP head, which also has the causal context, is better).
    static constexpr int kMinRepeat = 4;
    bool valid_repeat(int32_t pos, int32_t pos2, int hs) const {
        if (pos < 0 || pos2 < 0) return false;
        for (int i = 1; i <= kMinRepeat; ++i) {
            if (pos + i >= hs || pos2 + i >= hs) return false;
            if ((*hist)[pos + i] != (*hist)[pos2 + i]) return false;
        }
        return true;
    }

    // Rebuild from full history (prompt + generated so far).
    void rebuild(const std::vector<int32_t> & history) {
        init(history.size());
        hist = &history;
        for (int i = 0; i < static_cast<int>(history.size()); ++i) add(history.data(), i);
    }
};
}  // namespace

Generator::Generator(Engine & e) : eng_(e), sampler_(e.n_vocab) {
    const size_t V = static_cast<size_t>(eng_.n_vocab);
    const size_t Tm = static_cast<size_t>(eng_.config().t_max);
    last_h_.assign(H, 0.0f);
    hbuf_.assign(Tm * H, 0.0f);
    hin_.assign(Tm * H, 0.0f);
    hv_.assign(5 * static_cast<size_t>(H), 0.0f);      // 5 rows for T<=5 verify
    hv_next_.assign(5 * static_cast<size_t>(H), 0.0f);
    hm_.assign(H, 0.0f);
    hm2_.assign(H, 0.0f);
    lrows_.assign(5 * V, 0.0f);                         // 5 rows of logits
    lmtp_.assign(V, 0.0f);
    cached_.reserve(static_cast<size_t>(eng_.config().ctx_max) + 8);
}

bool Generator::is_stop(int32_t t) const { return t == eng_.tok.eos_id || (eng_.tok.is_control(t) && t != 0); }

void Generator::run(const std::vector<int32_t> & prompt, const GenParams & gp, const std::function<bool(int32_t)> & on_token, GenStats & st,
                    const std::function<void(int, int)> & on_prefill) {
    const EngineConfig & cfg = eng_.config();
    const int V = static_cast<int>(eng_.n_vocab);
    const bool adaptive = eng_.has_mtp() && gp.mtp_adaptive && gp.mtp_steps != 0;
    const int D = !eng_.has_mtp() ? 0 : adaptive ? 4 : std::max(0, std::min(gp.mtp_steps, 3));   // D = largest number of drafts
    const int n_prompt = static_cast<int>(prompt.size());
    if (n_prompt < 1) throw std::runtime_error("empty prompt");
    if (n_prompt + 1 >= cfg.ctx_max) throw std::runtime_error("prompt of " + std::to_string(n_prompt) + " tokens does not fit in the context of " + std::to_string(cfg.ctx_max));
    struct Guard {   // an exception leaves the model state in an unknown position: forget the cache
        Generator * g;
        bool ok = false;
        ~Guard() { if (!ok) g->reset_cache(); }
    } guard{this};
    st = GenStats{};
    st.prompt_tokens = n_prompt;
    st.mtp_steps = D;
    const int max_tokens = std::max(1, std::min(gp.max_tokens, cfg.ctx_max - n_prompt - 1));
    auto t_start = Clock::now();

    // ---- prefix reuse: the recurrent state cannot be rewound, so the cached tokens must be a proper prefix of the prompt
    size_t L = 0;
    if (gp.reuse && !cached_.empty() && cached_.size() < prompt.size() && (D == 0 || have_h_) &&
        std::equal(cached_.begin(), cached_.end(), prompt.begin())) {
        L = cached_.size();
        st.reuse = "prefix";
    }
    if (L == 0 && gp.reuse) {
        // The longest valid checkpoint that is a proper prefix of the prompt. Its KV positions are intact (only later positions
        // were written since it was taken), so restoring its recurrent state is enough to continue from it.
        Checkpoint * best = nullptr;
        for (Checkpoint & c : ck_) {
            if (c.tokens.empty() || c.tokens.size() >= prompt.size() || (D > 0 && !c.have_h)) continue;
            if (!std::equal(c.tokens.begin(), c.tokens.end(), prompt.begin())) continue;
            if (!best || c.tokens.size() > best->tokens.size()) best = &c;
        }
        if (best) {
            size_t off = 0;
            eng_.load_state(0, [&](void * p, size_t n) { std::memcpy(p, best->state.data() + off, n); off += n; });
            L = best->tokens.size();
            have_h_ = best->have_h;
            if (best->have_h) std::memcpy(last_h_.data(), best->h.data(), sizeof(float) * H);
            mtp_start_ = best->mtp_start;
            eng_.set_mtp_kv_start(mtp_start_);
            st.reuse = "checkpoint";
        }
    }
    // A run that does not extend a checkpoint rewrites some of its KV positions: forget it.
    for (Checkpoint & c : ck_)
        if (!c.tokens.empty() && L < c.tokens.size()) c.tokens.clear();
    const bool had_h = have_h_;
    cached_.clear();
    have_h_ = false;
    if (L == 0) {
        eng_.reset();
        // The MTP window skips the MTP KV below mtp_start_ (CPU attention only); the GPU MTP block attends over every position.
        mtp_start_ = (D > 0 && gp.mtp_window > 0 && !eng_.gpu_mtp_on()) ? std::max(0, n_prompt - gp.mtp_window) : 0;
        eng_.set_mtp_kv_start(mtp_start_);
    }
    st.reused_tokens = static_cast<int>(L);

    // ---- chunked prefill (chunks of t_max tokens; each chunk streams the CPU layers through the GPU when it is large)
    std::vector<float> & logits = lrows_;   // first row used for the prefill logits
    int pos = static_cast<int>(L);
    const int Tm = cfg.t_max;
    if (D > 0) {
        if (L > 0 && had_h) std::memcpy(hv_.data(), last_h_.data(), sizeof(float) * H);   // h of position L-1
        else std::memset(hv_.data(), 0, sizeof(float) * H);
    }
    // h row of the previous position for the MTP pairing of the first token of a chunk
    std::vector<float> & hprev = last_h_;
    if (D > 0 && !(L > 0 && had_h)) std::fill(hprev.begin(), hprev.end(), 0.0f);
    while (pos < n_prompt) {
        const int T = std::min(Tm, n_prompt - pos);
        const bool last = pos + T == n_prompt;
        const auto c0 = Clock::now();
        eng_.forward_ex(prompt.data() + pos, T, pos, last ? logits.data() : nullptr, last ? 1 : 0, D > 0 ? hbuf_.data() : nullptr);
        const auto c1 = Clock::now();
        st.prefill_main_s += secs(c0, c1);
        if (D > 0) {
            // MTP rows for positions s0 .. pos+T-1 (s0 > pos when the prefill window starts inside this chunk). Position k pairs with the
            // hidden row of k-1: hprev for the first position of the chunk (zeros at position 0, as llama.cpp), hbuf otherwise.
            const int s0 = std::max(pos, mtp_start_);
            if (s0 < pos + T) {
                const int n = pos + T - s0;
                for (int k = s0; k < pos + T; ++k) {
                    const float * src = k == pos ? hprev.data() : hbuf_.data() + static_cast<size_t>(k - 1 - pos) * H;
                    std::memcpy(hin_.data() + static_cast<size_t>(k - s0) * H, src, sizeof(float) * H);
                }
                eng_.mtp_prefill(prompt.data() + s0, hin_.data(), n, s0);
            }
            std::memcpy(hprev.data(), hbuf_.data() + static_cast<size_t>(T - 1) * H, sizeof(float) * H);
            st.mtp_prefill_s += secs(c1, Clock::now());
        }
        pos += T;
        if (on_prefill) on_prefill(pos - static_cast<int>(L), n_prompt - static_cast<int>(L));
    }
    if (gp.checkpoint && L < prompt.size()) take_checkpoint(ck_[1], prompt, D > 0);   // state of exactly the prompt
    st.prefill_s = secs(t_start, Clock::now());
    cached_.assign(prompt.begin(), prompt.end());   // fed tokens; extended below as tokens are fed
    if (gp.prefill_only) {   // the state holds the prompt; last_h_ (= hprev) is the h row of position n_prompt-1
        st.finish = "prefill";
        have_h_ = D > 0;
        guard.ok = true;
        return;
    }

    if (gp.capture_logits) gp.capture_logits->assign(logits.begin(), logits.begin() + V);
    sampler_.reset(gp.sp, prompt.data(), prompt.size());
    int cur = sampler_.sample(logits.data());
    sampler_.accept(cur);

    auto guarded = [&](auto && fn) {
        const uint64_t c0 = alloc_guard_count();
        alloc_guard_enable(true);
        fn();
        alloc_guard_enable(false);
        st.heap_allocs += alloc_guard_count() - c0;
    };
    bool first = true;
    auto emit = [&](int tok) -> bool {   // false: stop generating (abort or length)
        if (first) { st.ttft_s = secs(t_start, Clock::now()); first = false; }
        st.ids.push_back(tok);
        ++st.generated;
        if (!on_token(tok)) { st.finish = "abort"; return false; }
        if (st.generated >= max_tokens) { st.finish = "length"; return false; }
        return true;
    };
    const auto t_dec = Clock::now();

    if (D == 0) {
        while (true) {
            if (is_stop(cur)) { st.finish = "stop"; break; }
            if (!emit(cur)) break;
            if (pos + 1 >= cfg.ctx_max) { st.finish = "length"; break; }
            guarded([&] { eng_.forward(&cur, 1, pos, logits.data()); });
            st.dec_gpu += eng_.last_gpu_s; st.dec_cpu += eng_.last_cpu_s; st.dec_head += eng_.last_head_s;
            cached_.push_back(cur);
            ++pos;
            cur = sampler_.sample(logits.data());
            sampler_.accept(cur);
        }
} else {
        // Cascaded Speculative Decoding: PLD (Prompt Lookup Decoding) + Adaptive MTP.
        // Pure priority routing inside the T<=5 batch limit (kRollT = 5), as in main:
        //  1. the PLD 3/4-gram suffix matcher over prompt + generated history is consulted first;
        //  2. on a match (1..4 tokens) the verify batch is [cur] + PLD candidates and MTP is skipped,
        //     so the PLD context stays pure and is not blocked by the MTP drafts;
        //  3. with no PLD candidate the standard MTP path [cur] + MTP drafts runs exactly like main.
        // T<=5 keeps the 5-row buffers, the MTP head strides and the rollback path identical to main
        // (32 GPU layers fit the VRAM budget again).
        std::vector<int32_t> bt(6), vt(5), dr(4);   // max 5 verify tokens (4 drafts + 1 actual), as main
        std::vector<float> * hv = &hv_, * hvn = &hv_next_;
        // hv row 0 = h of position pos-1 (the last prefill row)
        std::memcpy(hv->data(), hprev.data(), sizeof(float) * H);
        std::vector<float> * hm = &hm_, * hm2 = &hm2_;
        int k_prev = 0;
        int pos_new = pos;

        // PLD matcher: built once (no allocation inside the token loop) and extended incrementally.
        PLDMatcher pld(cached_.size() + 16);
        pld.rebuild(cached_);
        pld.attach(&cached_);

        while (true) {
            if (is_stop(cur)) { st.finish = "stop"; break; }
            if (!emit(cur)) break;
            if (pos_new + 5 > cfg.ctx_max) { st.finish = "length"; break; }   // max 4 drafts + 1
            const auto q0 = Clock::now();

            // ---- PLD candidates FIRST (priority routing): deterministic 3-gram + 4-gram suffix match.
            // The suffix must end at `cur` (the token at position pos_new), because the candidate
            // predicts the token at pos_new+1; cached_ only holds the tokens before cur at this point.
            int32_t pld_cand[4];
            int n_pld = 0;
            if (gp.pld_enabled) {
                const size_t nc = cached_.size();
                int32_t suffix[4] = {0, 0, 0, cur};
                for (int i = 1; i <= 3; ++i) if (nc >= static_cast<size_t>(i)) suffix[3 - i] = cached_[nc - i];
                n_pld = pld.query(suffix, 3, pld_cand, std::min(gp.pld_max, 4));
            }
            st.pld_candidates += n_pld;
            if (n_pld > 0) st.pld_hits++;

            int De = 0;   // MTP drafts in this cycle (0 in the PLD path: MTP is skipped)
            if (n_pld > 0) {
                // PLD path: keep the candidates as dr[] so the next MTP catch-up replays them.
                for (int i = 0; i < n_pld; ++i) dr[i] = pld_cand[i];
            } else if (D > 0) {
                // Standard MTP path, identical to the main branch.
                for (int j = 0; j < k_prev; ++j) bt[j] = dr[j];
                bt[k_prev] = cur;
                guarded([&] { eng_.mtp_forward(bt.data(), hv->data(), k_prev + 1, pos_new - k_prev, hm->data(), lmtp_.data()); });
                dr[0] = argmax(lmtp_.data(), V);
                De = D;
                if (adaptive) {
                    float p = top1_prob(lmtp_.data(), V, dr[0]);
                    De = p >= gp.mtp_p_keep ? 1 : 0;
                    while (De < D && p >= gp.mtp_p_cont) {
                        guarded([&] { eng_.mtp_forward(&dr[De - 1], hm->data(), 1, pos_new + De, hm2->data(), lmtp_.data()); });
                        dr[De] = argmax(lmtp_.data(), V);
                        p = top1_prob(lmtp_.data(), V, dr[De]);
                        std::swap(hm, hm2);
                        if (p < gp.mtp_p_keep) break;   // too uncertain to be worth a verification row
                        ++De;
                    }
                } else {
                    for (int i = 1; i < D; ++i) {
                        guarded([&] { eng_.mtp_forward(&dr[i - 1], hm->data(), 1, pos_new + i, hm2->data(), lmtp_.data()); });
                        dr[i] = argmax(lmtp_.data(), V);
                        std::swap(hm, hm2);
                    }
                }
            }
            st.mtp_fills += De;
            const auto q1 = Clock::now();

            // ---- Verify batch (<= 5 tokens, one DDR4 pass through the CUDA graph)
            // PLD path: [cur] + pld_cand[0..n_pld-1]   MTP path: [cur] + dr[0..De-1] (main layout)
            vt[0] = cur;
            int vt_idx = 1;
            if (n_pld > 0) { for (int i = 0; i < n_pld; ++i) vt[vt_idx++] = pld_cand[i]; }
            else            { for (int i = 0; i < De; ++i) vt[vt_idx++] = dr[i]; }
            const int verify_T = vt_idx;

            guarded([&] {
                eng_.dn_snapshot(verify_T > 1);
                eng_.forward_ex(vt.data(), verify_T, pos_new, lrows_.data(), verify_T, hvn->data());
                eng_.dn_snapshot(false);
            });
            st.dec_gpu += eng_.last_gpu_s; st.dec_cpu += eng_.last_cpu_s; st.dec_head += eng_.last_head_s;
            const auto q2 = Clock::now();

            // ---- Acceptance sampling. lrows_[j] predicts the token at position pos_new+j+1, so the
            // candidate to check against it is vt[j+1]; the first mismatch ends the acceptance and
            // supplies the new actual token. Over verify_T-1 drafts this is the main branch loop.
            int k = 0, cur_new = 0;
            for (int j = 0; j < verify_T; ++j) {
                const int t = sampler_.sample(lrows_.data() + static_cast<size_t>(j) * V);
                sampler_.accept(t);
                if (j + 1 < verify_T && t == vt[j + 1]) {
                    k = j + 1;
                    cur_new = t;
                    // An accepted end token ends the reply: drafts after it are never part of the conversation, so they must not
                    // enter the model state (otherwise the cached prefix no longer matches the next turn's prompt).
                    if (is_stop(t)) break;
                    continue;
                }
                cur_new = t;
                break;
            }
            guarded([&] { if (k + 1 < verify_T) eng_.dn_rollback(k + 1); });
            const auto q3 = Clock::now();

            st.t_draft += secs(q0, q1); st.t_verify += secs(q1, q2); st.t_roll += secs(q2, q3);
            ++st.cycles; st.draft_n += n_pld + De; st.draft_acc += k; ++st.accept_hist[k];

            // ---- State update and emission
            cached_.push_back(cur);
            for (int j = 0; j < k; ++j) cached_.push_back(vt[j + 1]);

            // PLD matcher: register the n-gram keys ending at each newly accepted token (no allocation).
            {
                const int last = static_cast<int>(cached_.size()) - 1;
                const int base = last - k;   // index of `cur` in cached_
                for (int pos = base - 1; pos <= last; ++pos)
                    if (pos >= 0) pld.add(cached_.data(), pos);
            }

            // The verification hidden states (hvn) are exactly what the next cycle's catch-up needs.
            // After swap, hv points to hvn data: hv[0]=cur, hv[1]=dr[0], hv[2]=dr[1], ...
            // Next cycle's catch-up will use hv[0..k] for tokens at positions pos_new..pos_new+k.
            pos_new += k + 1;
            k_prev = k;
            cur = cur_new;
            std::swap(hv, hvn);

            bool go = true;
            for (int j = 0; j < k && go; ++j) {
                if (is_stop(vt[j + 1])) { st.finish = "stop"; go = false; break; }
                go = emit(vt[j + 1]);
            }
            if (!go) break;
        }
        std::memcpy(last_h_.data(), hv->data() + static_cast<size_t>(k_prev) * H, sizeof(float) * H);
        have_h_ = true;
    }
    st.decode_s = secs(t_dec, Clock::now());
    if (D == 0) have_h_ = false;
    guard.ok = true;
}

void Generator::take_checkpoint(Checkpoint & c, const std::vector<int32_t> & toks, bool have_h) {
    // Recurrent part only (Engine::save_state(0)); the positional KV stays in the caches. The buffer is sized once.
    const size_t nb = eng_.state_bytes(0);
    if (c.state.size() != nb) c.state.resize(nb);
    size_t off = 0;
    eng_.save_state(0, [&](const void * p, size_t n) { std::memcpy(c.state.data() + off, p, n); off += n; });
    c.tokens.assign(toks.begin(), toks.end());
    c.have_h = have_h;
    if (have_h) c.h.assign(last_h_.begin(), last_h_.end());
    c.mtp_start = mtp_start_;
}

// ---- state file --------------------------------------------------------------------------------------------------------
namespace {
constexpr char kStateMagic[8] = {'Q', '3', '8', 'S', 'T', 'A', 'T', 'E'};
constexpr uint32_t kStateVersion = 1;
struct StateHeader {
    char magic[8];
    uint32_t version;
    uint32_t n_pos;           // cached tokens = positions 0 .. n_pos-1 of every cache
    uint64_t fingerprint;     // Engine::state_fingerprint() of the writer
    uint64_t payload_bytes;   // Engine::state_bytes(n_pos)
    int32_t mtp_start;
    uint32_t have_h;          // 1: a 5120-float h row follows the tokens
};
struct FileCloser { void operator()(std::FILE * f) const { if (f) std::fclose(f); } };
}  // namespace

size_t Generator::save_state(const std::string & path) const {
    if (cached_.empty()) throw std::runtime_error("state-save: nothing to save (empty cache)");
    const std::string tmp = path + ".tmp";
    std::unique_ptr<std::FILE, FileCloser> f(std::fopen(tmp.c_str(), "wb"));
    if (!f) throw std::runtime_error("state-save: cannot open " + tmp);
    std::vector<char> vbuf(8u << 20);
    std::setvbuf(f.get(), vbuf.data(), _IOFBF, vbuf.size());
    StateHeader h{};
    std::memcpy(h.magic, kStateMagic, 8);
    h.version = kStateVersion;
    h.n_pos = static_cast<uint32_t>(cached_.size());
    h.fingerprint = eng_.state_fingerprint();
    h.payload_bytes = eng_.state_bytes(static_cast<int>(cached_.size()));
    h.mtp_start = mtp_start_;
    h.have_h = have_h_ ? 1 : 0;
    auto put = [&](const void * p, size_t n) {
        if (n && std::fwrite(p, 1, n, f.get()) != n) throw std::runtime_error("state-save: write failed (disk full?)");
    };
    put(&h, sizeof(h));
    put(cached_.data(), cached_.size() * sizeof(int32_t));
    if (have_h_) put(last_h_.data(), last_h_.size() * sizeof(float));
    size_t moved = 0;
    eng_.save_state(static_cast<int>(cached_.size()), [&](const void * p, size_t n) { put(p, n); moved += n; });
    if (moved != h.payload_bytes) throw std::runtime_error("state-save: payload size mismatch");
    if (std::fflush(f.get()) != 0) throw std::runtime_error("state-save: flush failed");
    f.reset();
    if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("state-save: rename to " + path + " failed");
    return moved;
}

size_t Generator::load_state(const std::string & path) {
    std::unique_ptr<std::FILE, FileCloser> f(std::fopen(path.c_str(), "rb"));
    if (!f) throw std::runtime_error("state-load: cannot open " + path);
    std::vector<char> vbuf(8u << 20);
    std::setvbuf(f.get(), vbuf.data(), _IOFBF, vbuf.size());
    auto get = [&](void * p, size_t n) {
        if (n && std::fread(p, 1, n, f.get()) != n) throw std::runtime_error("state-load: file is truncated");
    };
    StateHeader h{};
    get(&h, sizeof(h));
    if (std::memcmp(h.magic, kStateMagic, 8) != 0) throw std::runtime_error("state-load: not a state file");
    if (h.version != kStateVersion) throw std::runtime_error("state-load: unsupported version " + std::to_string(h.version));
    if (h.fingerprint != eng_.state_fingerprint())
        throw std::runtime_error("state-load: the file was written by a different model or layout (model file, --gpu-layers, --kv)");
    const EngineConfig & cfg = eng_.config();
    if (h.n_pos < 1 || static_cast<int>(h.n_pos) + 1 >= cfg.ctx_max)
        throw std::runtime_error("state-load: " + std::to_string(h.n_pos) + " saved positions do not fit --ctx " + std::to_string(cfg.ctx_max));
    if (h.payload_bytes != eng_.state_bytes(static_cast<int>(h.n_pos))) throw std::runtime_error("state-load: payload size mismatch");
    if (h.have_h > 1 || (h.have_h && !eng_.has_mtp())) throw std::runtime_error("state-load: bad MTP row flag");
    std::vector<int32_t> toks(h.n_pos);
    get(toks.data(), toks.size() * sizeof(int32_t));
    for (int32_t t : toks)
        if (t < 0 || t >= eng_.n_vocab) throw std::runtime_error("state-load: token id out of range");
    if (h.have_h) get(last_h_.data(), last_h_.size() * sizeof(float));
    cached_.clear();
    have_h_ = false;
    reset_cache();   // the loaded state replaces every KV position the checkpoints relied on
    eng_.reset();
    size_t moved = 0;
    eng_.load_state(static_cast<int>(h.n_pos), [&](void * p, size_t n) { get(p, n); moved += n; });
    if (std::fgetc(f.get()) != EOF) throw std::runtime_error("state-load: trailing bytes after the payload");
    cached_ = std::move(toks);
    cached_.reserve(static_cast<size_t>(cfg.ctx_max) + 8);
    have_h_ = h.have_h != 0;
    mtp_start_ = h.mtp_start;
    eng_.set_mtp_kv_start(mtp_start_);
    if (pin_loaded_) take_checkpoint(ck_[0], cached_, have_h_);   // pinned: questions about a loaded document reuse it
    return moved;
}
