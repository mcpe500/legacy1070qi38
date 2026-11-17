#include "generator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
    struct Entry { uint32_t key; uint16_t count; int32_t tokens[8]; };

    std::vector<Entry> table3;   // 3-gram table
    std::vector<Entry> table4;   // 4-gram table
    uint32_t mask3, mask4;

    PLDMatcher(size_t expected_tokens = 0) { init(expected_tokens); }

    void init(size_t expected_tokens) {
        size_t cap3 = 1;
        while (cap3 < expected_tokens * 2) cap3 <<= 1;
        size_t cap4 = 1;
        while (cap4 < expected_tokens * 2) cap4 <<= 1;
        table3.assign(cap3, Entry{EMPTY, 0, {}});
        table4.assign(cap4, Entry{EMPTY, 0, {}});
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

    // Add suffix at position `pos` (next token is at pos+1).
    void add(const int32_t * tokens, int pos, int max_pos) {
        if (pos + 1 > max_pos) return;
        // 3-gram
        if (pos >= 2) {
            uint32_t h = hash3(tokens[pos-2], tokens[pos-1], tokens[pos]) & mask3;
            for (;; h = (h + 1) & mask3) {
                Entry & e = table3[h];
                if (e.key == EMPTY) { e.key = h; e.count = 1; e.tokens[0] = tokens[pos+1]; break; }
                if (e.key == h) {
                    // deduplicate
                    bool found = false;
                    for (uint16_t i = 0; i < e.count; ++i) if (e.tokens[i] == tokens[pos+1]) { found = true; break; }
                    if (!found && e.count < 8) e.tokens[e.count++] = tokens[pos+1];
                    break;
                }
            }
        }
        // 4-gram
        if (pos >= 3) {
            uint32_t h = hash4(tokens[pos-3], tokens[pos-2], tokens[pos-1], tokens[pos]) & mask4;
            for (;; h = (h + 1) & mask4) {
                Entry & e = table4[h];
                if (e.key == EMPTY) { e.key = h; e.count = 1; e.tokens[0] = tokens[pos+1]; break; }
                if (e.key == h) {
                    bool found = false;
                    for (uint16_t i = 0; i < e.count; ++i) if (e.tokens[i] == tokens[pos+1]) { found = true; break; }
                    if (!found && e.count < 8) e.tokens[e.count++] = tokens[pos+1];
                    break;
                }
            }
        }
    }

    // Query candidates for the current suffix (up to 3-gram + 4-gram). Returns count written to `out`.
    int query(const int32_t * suffix, int suffix_len, int32_t * out, int max_out) const {
        int written = 0;
        // 4-gram first (longer context = more specific)
        if (suffix_len >= 3) {
            uint32_t h = hash4(suffix[suffix_len-3], suffix[suffix_len-2], suffix[suffix_len-1], suffix[suffix_len]) & mask4;
            for (int probe = 0; probe < 16 && written < max_out; ++probe, h = (h + 1) & mask4) {
                const Entry & e = table4[h];
                if (e.key == EMPTY) break;
                if (e.key == h) {
                    for (uint16_t i = 0; i < e.count && written < max_out; ++i) out[written++] = e.tokens[i];
                    break;
                }
            }
        }
        // 3-gram
        if (suffix_len >= 2 && written < max_out) {
            uint32_t h = hash3(suffix[suffix_len-2], suffix[suffix_len-1], suffix[suffix_len]) & mask3;
            for (int probe = 0; probe < 16 && written < max_out; ++probe, h = (h + 1) & mask3) {
                const Entry & e = table3[h];
                if (e.key == EMPTY) break;
                if (e.key == h) {
                    for (uint16_t i = 0; i < e.count && written < max_out; ++i) out[written++] = e.tokens[i];
                    break;
                }
            }
        }
        return written;
    }

    // Rebuild from full history (prompt + generated so far).
    void rebuild(const std::vector<int32_t> & history) {
        init(history.size());
        for (int i = 0; i + 1 < static_cast<int>(history.size()); ++i) add(history.data(), i, static_cast<int>(history.size()) - 1);
    }
};
}  // namespace

Generator::Generator(Engine & e) : eng_(e), sampler_(e.n_vocab) {
    const size_t V = static_cast<size_t>(eng_.n_vocab);
    const size_t Tm = static_cast<size_t>(eng_.config().t_max);
    last_h_.assign(H, 0.0f);
    hbuf_.assign(Tm * H, 0.0f);
    hin_.assign(Tm * H, 0.0f);
    hv_.assign(8 * static_cast<size_t>(H), 0.0f);      // 8 rows for T<=8 verify
    hv_next_.assign(8 * static_cast<size_t>(H), 0.0f);
    hm_.assign(H, 0.0f);
    hm2_.assign(H, 0.0f);
    lrows_.assign(8 * V, 0.0f);                         // 8 rows of logits
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
        ~Guard() { if (!ok) { g->cached_.clear(); g->have_h_ = false; } }
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
    }
    const bool had_h = have_h_;
    cached_.clear();
    have_h_ = false;
    if (L == 0) {
        eng_.reset();
        mtp_start_ = (D > 0 && gp.mtp_window > 0) ? std::max(0, n_prompt - gp.mtp_window) : 0;
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
    st.prefill_s = secs(t_start, Clock::now());
    cached_.assign(prompt.begin(), prompt.end());   // fed tokens; extended below as tokens are fed

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
        // Cascaded Speculative Decoding: PLD (Prompt Lookup Decoding) + Adaptive MTP
        // PLD: 3-gram and 4-gram suffix hash/matcher over full prompt + generated history.
        //      Up to pld_max deterministic candidates first when context patterns match.
        // MTP: fills remaining verify slots (up to kRollT - 1 drafts + 1 actual token).
        // Total verify batch <= 8 (kRollT), single DDR4 pass via CUDA graph.
        // Adaptive early-stop: probability-based (original logic) on MTP draft head.
        // This loop is the ORIGINAL speculative decoding with minimal PLD insertion.
        std::vector<int32_t> bt(9), vt(8), dr(7);   // max 8 verify tokens (7 drafts + 1 actual)
        std::vector<float> * hv = &hv_, * hvn = &hv_next_;
        // hv row 0 = h of position pos-1 (the last prefill row)
        std::memcpy(hv->data(), hprev.data(), sizeof(float) * H);
        std::vector<float> * hm = &hm_, * hm2 = &hm2_;
        int k_prev = 0;
        int pos_new = pos;

        // PLD matcher: build from prompt + generated so far, incrementally updated each cycle
        PLDMatcher pld(cached_.size() + 16);
        pld.rebuild(cached_);

        while (true) {
            if (is_stop(cur)) { st.finish = "stop"; break; }
            if (!emit(cur)) break;
            if (pos_new + 8 > cfg.ctx_max) { st.finish = "length"; break; }   // max 7 drafts + 1
            const auto q0 = Clock::now();

            // ---- Build bt[] = [accepted drafts from prev cycle] + [current token]
            for (int j = 0; j < k_prev; ++j) bt[j] = dr[j];
            bt[k_prev] = cur;
            const int bt_len = k_prev + 1;

            // ---- MTP catch-up on accepted drafts + draft step 0
            guarded([&] { eng_.mtp_forward(bt.data(), hv->data(), bt_len, pos_new - k_prev, hm->data(), lmtp_.data()); });
            dr[0] = argmax(lmtp_.data(), V);

            // ---- PLD candidates (deterministic, from hash table)
            int32_t pld_cand[6];
            int n_pld = 0;
            if (gp.pld_enabled) {
                n_pld = pld.query(cached_.data(), static_cast<int>(cached_.size()) - 1, pld_cand, gp.pld_max);
            }
            st.pld_candidates += n_pld;
            if (n_pld > 0) st.pld_hits++;

            // ---- MTP adaptive drafting (ORIGINAL LOGIC - unchanged)
            int De = D;   // drafts in this cycle
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
            const auto q1 = Clock::now();

            // ---- Verify batch: [cur] + MTP drafts dr[0]..dr[De-1] + PLD candidates
            // Total verify tokens = 1 + De + n_pld <= 8 (kRollT)
            // MTP drafts at contiguous positions 1..De to preserve hm/hm2 chain
            vt[0] = cur;
            int vt_idx = 1;
            for (int i = 0; i < De && vt_idx < 8; ++i) vt[vt_idx++] = dr[i];
            for (int i = 0; i < n_pld && vt_idx < 8; ++i) vt[vt_idx++] = pld_cand[i];
            const int verify_T = vt_idx;

            guarded([&] {
                eng_.dn_snapshot(verify_T > 1);
                eng_.forward_ex(vt.data(), verify_T, pos_new, lrows_.data(), verify_T, hvn->data());
                eng_.dn_snapshot(false);
            });
            st.dec_gpu += eng_.last_gpu_s; st.dec_cpu += eng_.last_cpu_s; st.dec_head += eng_.last_head_s;
            const auto q2 = Clock::now();

            // ---- Acceptance sampling (matching main branch logic)
            // verify_T = 1 (cur) + De (MTP drafts) + n_pld (PLD candidates)
            // lrows_[0] = logits for cur (position pos_new)
            // lrows_[1..De] = logits for MTP drafts dr[0..De-1] (positions pos_new+1..pos_new+De)
            // lrows_[De+1..] = logits for PLD candidates
            int k = 0, cur_new = 0;
            for (int j = 0; j < verify_T; ++j) {
                const int t = sampler_.sample(lrows_.data() + static_cast<size_t>(j) * V);
                sampler_.accept(t);
                if (j == 0) {
                    // Position pos_new: should match cur (already known)
                    cur_new = t;
                } else if (j <= De) {
                    // Position pos_new + j: MTP draft dr[j-1]
                    if (t == dr[j - 1]) {
                        k = j;   // accepted j drafts (dr[0]..dr[j-1])
                        cur_new = t;  // next actual token is from this position
                        continue;
                    }
                } else {
                    // Position pos_new + De + (j - De - 1): PLD candidates
                    int pld_idx = j - De - 1;
                    if (pld_idx < n_pld && t == pld_cand[pld_idx]) {
                        k = j;
                        cur_new = t;
                        continue;
                    }
                }
                break;
            }
            // Note: k is the number of accepted drafts (vt[1]..vt[k] accepted)
            guarded([&] { if (k < n_pld + De) eng_.dn_rollback(k + 1); });
            const auto q3 = Clock::now();

            st.t_draft += secs(q0, q1); st.t_verify += secs(q1, q2); st.t_roll += secs(q2, q3);
            ++st.cycles; st.draft_n += n_pld + De; st.draft_acc += k; ++st.accept_hist[k];

            // ---- State update and emission
            cached_.push_back(cur);
            for (int j = 0; j < k; ++j) cached_.push_back(vt[j + 1]);

            // Update PLD matcher incrementally (add newly accepted tokens to history)
            for (int j = 0; j < k; ++j) {
                int new_pos = static_cast<int>(cached_.size()) - 1;
                if (new_pos >= 0) pld.add(cached_.data(), new_pos - 1, new_pos);
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
