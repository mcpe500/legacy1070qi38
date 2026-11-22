// Generation session on top of Engine: chunked prefill (prompts up to ctx_max), reuse of the cached prefix for multi-turn chat,
// the sampler, and speculative decoding with the MTP head and DeltaNet rollback. Used by the CLI and by the HTTP server.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "engine.hpp"
#include "sampler.hpp"

// Verification batch of the speculative path: 1 target row plus the accepted drafts. The buffers (attention
// partials, DeltaNet rollback, MTP hidden rows) are sized by the engine for kVerifyMax rows, and the generator
// derives its batch from Engine::roll_t() so a smaller plan degrades the draft count instead of overflowing.
constexpr int kVerifyMax = 17;      // a Drafter v2 batch of 16 drafts + 1 target row
constexpr int kPldMax = 16;         // PLD drafts per cycle (at most kVerifyMax - 1)
constexpr int kVerifyHist = kVerifyMax;

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
    // Prompt Lookup Decoding (PLD): deterministic n-gram suffix match over prompt + generated history. When enabled, up to
    // pld_max candidates fill the verification batch before the MTP drafts (the whole 17-row batch at pld_max = 16).
    bool pld_enabled = false;
    // Default measured: a suffix match that fills the whole 17-row batch wins only on highly repetitive
    // text, and it loses 6x on the Indonesian chat workload (issue #57 M3), so the default stays at 4 drafts
    // while the knob accepts up to kPldMax.
    int pld_max = 4;       // max PLD candidates per cycle (3-gram + 4-gram combined), 1..kPldMax
    int pld_min_len = 3;   // minimum suffix length to consider (3 or 4)
    bool reuse = true;     // continue from the cached state when it is a prefix of the new prompt
    // Prompt checkpoint (server): after the prefill the recurrent state of the prompt (DeltaNet conv/ssm, about 150 MiB of
    // host RAM, Engine::save_state(0)) is copied aside. A later prompt that extends that prompt but not the full cached
    // sequence (the reply was trimmed, cut by a stop string, re-tokenized differently, or ended before <|im_end|>) restores
    // it and reuses the prompt's KV positions instead of prefilling the whole conversation again.
    bool checkpoint = false;
    bool prefill_only = false;   // stop after the prefill (no sampling): the state then holds exactly the prompt (--state-save)
    std::vector<float> * capture_logits = nullptr;   // receives the logits after the prefill (parity tests)
};

struct GenStats {
    int prompt_tokens = 0, reused_tokens = 0, generated = 0, mtp_steps = 0;
    double prefill_s = 0, prefill_main_s = 0, mtp_prefill_s = 0, ttft_s = 0, decode_s = 0;
    double t_draft = 0, t_verify = 0, t_roll = 0, dec_gpu = 0, dec_cpu = 0, dec_head = 0;
    long long cycles = 0, draft_n = 0, draft_acc = 0, accept_hist[kVerifyHist] = {};   // one slot per possible accept count (0..kVerifyHist-1)
    long long pld_hits = 0, pld_candidates = 0, mtp_fills = 0;
    double t_pld = 0;   // CPU time of the PLD suffix query per cycle (the n-gram index, no VRAM)
    uint64_t heap_allocs = 0;
    std::string finish = "length";   // "stop" (end token), "length", "abort" (callback returned false)
    std::string reuse = "none";      // where reused_tokens came from: "none", "prefix" (cached sequence), "checkpoint" (prompt checkpoint)
    std::vector<int32_t> ids;        // generated tokens (the stop token is not included)
};

class Generator {
public:
    explicit Generator(Engine & e);
    // Calls on_token for every generated token as soon as it is final. Returning false aborts the generation.
    // Called with the engine idle; one run at a time. `on_prefill(done, total)` (optional) reports prefill chunks.
    void run(const std::vector<int32_t> & prompt, const GenParams & gp, const std::function<bool(int32_t)> & on_token, GenStats & st,
             const std::function<void(int, int)> & on_prefill = nullptr);
    void reset_cache() { cached_.clear(); have_h_ = false; for (Checkpoint & c : ck_) c.tokens.clear(); }
    // Tokens of the prompt checkpoint (slot 1) and of the pinned state-load checkpoint (slot 0).
    size_t checkpoint_tokens(int slot) const { return ck_[slot].tokens.size(); }
    size_t cached_tokens() const { return cached_.size(); }
    const std::vector<int32_t> & cached() const { return cached_; }
    // Model state file (--state-save / --state-load): header (magic, version, engine fingerprint, n_pos), the cached tokens,
    // the MTP pairing row, then Engine::save_state(n_pos). After load_state the next run() whose prompt extends the saved
    // tokens continues from them without prefilling them again. Both return the number of state bytes moved.
    size_t save_state(const std::string & path) const;
    size_t load_state(const std::string & path);
    // Server: keep the state of load_state as a pinned checkpoint (slot 0), see GenParams::checkpoint.
    void set_pin_loaded(bool on) { pin_loaded_ = on; }
    bool is_stop(int32_t t) const;

private:
    Engine & eng_;
    std::vector<int32_t> cached_;      // tokens whose effect is in the model state (positions 0 .. n-1)
    int mtp_start_ = 0;                // first prompt position that went through the MTP block
    bool have_h_ = false;              // last_h_ holds the output-normed hidden row of position n-1 (needed to continue the MTP KV)
    std::vector<float> last_h_, hbuf_, hin_, hv_, hv_next_, hm_, hm2_, lrows_, lmtp_;
    // Verification batch rows this session may build (Engine::roll_t() clamped to kVerifyMax, 5 without MTP):
    // the draft count is capped to vcap_ - 1 everywhere, so a smaller buffer plan degrades the batch, never overflows.
    int vcap_ = 5;
    // Checkpoints (GenParams::checkpoint): tokens, recurrent state (Engine::save_state(0)), MTP pairing row. A checkpoint is
    // valid while the KV positions 0 .. tokens.size()-1 still hold its tokens (a run that rewrites any of them clears it).
    // Slot 0: the state loaded with load_state (pinned: a --state-load document stays reusable for every question about it);
    // slot 1: the prompt of the last run.
    struct Checkpoint {
        std::vector<int32_t> tokens;
        std::vector<uint8_t> state;
        std::vector<float> h;
        bool have_h = false;
        int mtp_start = 0;
    };
    Checkpoint ck_[2];
    bool pin_loaded_ = false;
    void take_checkpoint(Checkpoint & c, const std::vector<int32_t> & toks, bool have_h);
    Sampler sampler_;
};
