// Generation session on top of Engine: chunked prefill (prompts up to ctx_max), reuse of the cached prefix for multi-turn chat,
// the sampler, and speculative decoding with the MTP head and DeltaNet rollback. Used by the CLI and by the HTTP server.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "engine.hpp"
#include "sampler.hpp"

struct GenParams {
    SamplerParams sp;
    int max_tokens = 256;
    int mtp_steps = 2;     // 0..3, clamped to 0 when the engine has no MTP block
    // MTP prefill: run the MTP block only on the last `mtp_window` prompt tokens (0 = all of them). Only the draft quality changes;
    // the output never does (every draft is verified by the main model).
    int mtp_window = 0;
    // Adaptive drafting (default since 2026-11-17): draft step 0 is kept when the MTP head's top-1 probability is >= p_keep, and
    // drafting continues while it stays >= p_cont, up to 4 drafts. mtp_steps is ignored (except 0 = MTP off) when adaptive is on.
    // Output is identical with and without it (every draft is verified); it only changes speed. CLI --mtp-adaptive 0 turns it off.
    bool mtp_adaptive = true;
    float mtp_p_keep = 0.5f, mtp_p_cont = 0.85f;
    // Prompt Lookup Decoding (PLD): deterministic n-gram suffix match over prompt + generated history.
    // When enabled, up to pld_max candidates (3-gram + 4-gram) are added to the verify batch before MTP drafts.
    bool pld_enabled = false;
    int pld_max = 6;       // max PLD candidates per cycle (3-gram + 4-gram combined)
    int pld_min_len = 3;   // minimum suffix length to consider (3 or 4)
    bool reuse = true;     // continue from the cached state when it is a prefix of the new prompt
    std::vector<float> * capture_logits = nullptr;   // receives the logits after the prefill (parity tests)
};

struct GenStats {
    int prompt_tokens = 0, reused_tokens = 0, generated = 0, mtp_steps = 0;
    double prefill_s = 0, prefill_main_s = 0, mtp_prefill_s = 0, ttft_s = 0, decode_s = 0;
    double t_draft = 0, t_verify = 0, t_roll = 0, dec_gpu = 0, dec_cpu = 0, dec_head = 0;
    long long cycles = 0, draft_n = 0, draft_acc = 0, accept_hist[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    long long pld_hits = 0, pld_candidates = 0, mtp_fills = 0;
    uint64_t heap_allocs = 0;
    std::string finish = "length";   // "stop" (end token), "length", "abort" (callback returned false)
    std::vector<int32_t> ids;        // generated tokens (the stop token is not included)
};

class Generator {
public:
    explicit Generator(Engine & e);
    // Calls on_token for every generated token as soon as it is final. Returning false aborts the generation.
    // Called with the engine idle; one run at a time. `on_prefill(done, total)` (optional) reports prefill chunks.
    void run(const std::vector<int32_t> & prompt, const GenParams & gp, const std::function<bool(int32_t)> & on_token, GenStats & st,
             const std::function<void(int, int)> & on_prefill = nullptr);
    void reset_cache() { cached_.clear(); have_h_ = false; }
    size_t cached_tokens() const { return cached_.size(); }
    bool is_stop(int32_t t) const;

private:
    Engine & eng_;
    std::vector<int32_t> cached_;      // tokens whose effect is in the model state (positions 0 .. n-1)
    int mtp_start_ = 0;                // first prompt position that went through the MTP block
    bool have_h_ = false;              // last_h_ holds the output-normed hidden row of position n-1 (needed to continue the MTP KV)
    std::vector<float> last_h_, hbuf_, hin_, hv_, hv_next_, hm_, hm2_, lrows_, lmtp_;
    Sampler sampler_;
};
