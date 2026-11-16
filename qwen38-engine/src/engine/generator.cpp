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
}  // namespace

Generator::Generator(Engine & e) : eng_(e), sampler_(e.n_vocab) {
    const size_t V = static_cast<size_t>(eng_.n_vocab);
    const size_t Tm = static_cast<size_t>(eng_.config().t_max);
    last_h_.assign(H, 0.0f);
    hbuf_.assign(Tm * H, 0.0f);
    hin_.assign(Tm * H, 0.0f);
    hv_.assign(5 * static_cast<size_t>(H), 0.0f);
    hv_next_.assign(5 * static_cast<size_t>(H), 0.0f);
    hm_.assign(H, 0.0f);
    hm2_.assign(H, 0.0f);
    lrows_.assign(5 * V, 0.0f);
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
        // Speculative decoding. Pairing (as llama.cpp): the MTP block at position p takes token x_p and the main-model hidden row of
        // position p-1. Each cycle: one MTP batch (catch-up on the drafts accepted last time + draft step 0), D-1 single MTP steps,
        // one verification batch [cur, drafts] with logits for every row. Row j is sampled with the normal sampler; a draft is
        // accepted when it equals the sampled token, otherwise the sampled token replaces it. With bit-identical rows (every
        // operator gives a token the same result for any batch size) the output and the RNG stream equal plain decoding.
        std::vector<int32_t> bt(6), vt(5), dr(4);
        std::vector<float> * hv = &hv_, * hvn = &hv_next_;
        // hv row 0 = h of position pos-1 (the last prefill row)
        std::memcpy(hv->data(), hprev.data(), sizeof(float) * H);
        std::vector<float> * hm = &hm_, * hm2 = &hm2_;
        int k_prev = 0;
        int pos_new = pos;
        while (true) {
            if (is_stop(cur)) { st.finish = "stop"; break; }
            if (!emit(cur)) break;
            if (pos_new + D + 1 > cfg.ctx_max) { st.finish = "length"; break; }
            const auto q0 = Clock::now();
            for (int j = 0; j < k_prev; ++j) bt[j] = dr[j];
            bt[k_prev] = cur;
            guarded([&] { eng_.mtp_forward(bt.data(), hv->data(), k_prev + 1, pos_new - k_prev, hm->data(), lmtp_.data()); });
            dr[0] = argmax(lmtp_.data(), V);
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
            vt[0] = cur;
            for (int i = 0; i < De; ++i) vt[i + 1] = dr[i];
            guarded([&] {
                eng_.dn_snapshot(De > 0);
                eng_.forward_ex(vt.data(), De + 1, pos_new, lrows_.data(), De + 1, hvn->data());
                eng_.dn_snapshot(false);
            });
            st.dec_gpu += eng_.last_gpu_s; st.dec_cpu += eng_.last_cpu_s; st.dec_head += eng_.last_head_s;
            const auto q2 = Clock::now();
            int k = 0, cur_new = 0;
            for (int j = 0; j <= De; ++j) {
                const int t = sampler_.sample(lrows_.data() + static_cast<size_t>(j) * V);
                sampler_.accept(t);
                if (j < De && t == dr[j]) { k = j + 1; continue; }
                cur_new = t;
                break;
            }
            guarded([&] { if (k < De) eng_.dn_rollback(k + 1); });
            const auto q3 = Clock::now();
            st.t_draft += secs(q0, q1); st.t_verify += secs(q1, q2); st.t_roll += secs(q2, q3);
            ++st.cycles; st.draft_n += De; st.draft_acc += k; ++st.accept_hist[k];
            // state first, then emission (emission may end the run)
            cached_.push_back(cur);
            for (int j = 0; j < k; ++j) cached_.push_back(dr[j]);
            const int cur_old = cur;
            (void) cur_old;
            pos_new += k + 1;
            k_prev = k;
            cur = cur_new;
            std::swap(hv, hvn);
            bool go = true;
            for (int j = 0; j < k && go; ++j) {
                if (is_stop(dr[j])) { st.finish = "stop"; go = false; break; }
                go = emit(dr[j]);
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
