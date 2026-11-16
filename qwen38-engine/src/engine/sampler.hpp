// Token sampler: repetition / presence penalties, top_k, top_p, min_p, temperature, seeded RNG.
// Order (as llama.cpp's default chain): penalties -> top_k -> top_p -> min_p -> temperature -> draw.
// temperature <= 0 is greedy (argmax after the penalties). No allocation after construction.
#pragma once
#include <cstdint>
#include <random>
#include <vector>

struct SamplerParams {
    float temperature = 0.0f;
    int top_k = 0;                 // 0 = no top_k (candidates are still capped at kMaxCand)
    float top_p = 1.0f;            // 1 = off
    float min_p = 0.0f;            // 0 = off
    float repeat_penalty = 1.0f;   // 1 = off; logit > 0 is divided, logit < 0 multiplied
    float presence_penalty = 0.0f; // 0 = off; subtracted once from every token seen in the window
    int repeat_last_n = 64;        // penalty window (tokens of prompt + output), 0 = off, max 1024
    int64_t seed = 42;             // < 0: random seed
};

class Sampler {
public:
    static constexpr int kMaxCand = 4096;
    explicit Sampler(int64_t n_vocab);

    // Clears the history and reseeds. `history` is the prompt (only its tail matters for the penalties).
    void reset(const SamplerParams & p, const int32_t * history, size_t n);
    void accept(int32_t tok);                 // appends a token that became part of the sequence
    int32_t sample(float * logits);           // applies the penalties in place, returns the token. Does not call accept().

    const SamplerParams & params() const { return p_; }

private:
    SamplerParams p_;
    int64_t V_;
    std::vector<int32_t> hist_;
    std::vector<int32_t> idx_;       // permutation of the vocabulary used for the top-k selection
    std::vector<float> cand_p_;      // candidate probabilities
    std::vector<int32_t> win_;       // scratch: sorted window
    std::mt19937_64 rng_;
};
