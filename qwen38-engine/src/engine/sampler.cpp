#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

Sampler::Sampler(int64_t n_vocab) : V_(n_vocab) {
    idx_.resize(static_cast<size_t>(V_));
    std::iota(idx_.begin(), idx_.end(), 0);
    cand_p_.resize(kMaxCand);
    win_.reserve(1024);
    hist_.reserve(1 << 14);
}

void Sampler::reset(const SamplerParams & p, const int32_t * history, size_t n) {
    p_ = p;
    p_.repeat_last_n = std::max(0, std::min(p_.repeat_last_n, 1024));
    hist_.clear();
    const size_t keep = std::min<size_t>(n, 1024);   // only the tail can ever fall into the window
    hist_.insert(hist_.end(), history + (n - keep), history + n);
    rng_.seed(p_.seed >= 0 ? static_cast<uint64_t>(p_.seed) : std::random_device{}());
}

void Sampler::accept(int32_t tok) {
    if (hist_.size() >= 4096) hist_.erase(hist_.begin(), hist_.begin() + 2048);   // rare; keeps the buffer bounded
    hist_.push_back(tok);
}

int32_t Sampler::sample(float * logits) {
    // Penalties over the last repeat_last_n tokens (each distinct token once).
    if (p_.repeat_last_n > 0 && (p_.repeat_penalty != 1.0f || p_.presence_penalty != 0.0f) && !hist_.empty()) {
        const size_t n = std::min<size_t>(hist_.size(), static_cast<size_t>(p_.repeat_last_n));
        win_.assign(hist_.end() - static_cast<std::ptrdiff_t>(n), hist_.end());   // within reserved capacity
        std::sort(win_.begin(), win_.end());
        win_.erase(std::unique(win_.begin(), win_.end()), win_.end());
        for (int32_t t : win_) {
            if (t < 0 || t >= V_) continue;
            float & l = logits[t];
            if (p_.repeat_penalty != 1.0f) l = l > 0.0f ? l / p_.repeat_penalty : l * p_.repeat_penalty;
            l -= p_.presence_penalty;
        }
    }
    if (p_.temperature <= 0.0f) {
        int64_t best = 0;
        for (int64_t i = 1; i < V_; ++i) if (logits[i] > logits[best]) best = i;
        return static_cast<int32_t>(best);
    }

    // Candidates: the K largest logits, sorted descending.
    const int K = static_cast<int>(std::min<int64_t>(p_.top_k > 0 ? p_.top_k : kMaxCand, std::min<int64_t>(kMaxCand, V_)));
    auto gt = [&](int32_t a, int32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
    if (K < V_) std::nth_element(idx_.begin(), idx_.begin() + K, idx_.end(), gt);
    std::sort(idx_.begin(), idx_.begin() + K, gt);

    int n = K;
    const float mx = logits[idx_[0]];
    // top_p and min_p on the temperature-1 distribution of the candidates.
    double sum = 0;
    for (int i = 0; i < n; ++i) { cand_p_[i] = std::exp(logits[idx_[i]] - mx); sum += cand_p_[i]; }
    for (int i = 0; i < n; ++i) cand_p_[i] = static_cast<float>(cand_p_[i] / sum);
    if (p_.top_p < 1.0f) {
        double c = 0;
        int cut = n;
        for (int i = 0; i < n; ++i) {
            c += cand_p_[i];
            if (c >= p_.top_p) { cut = i + 1; break; }
        }
        n = std::max(1, cut);
    }
    if (p_.min_p > 0.0f) {
        const float thr = p_.min_p * cand_p_[0];
        int cut = 1;
        while (cut < n && cand_p_[cut] >= thr) ++cut;
        n = cut;
    }
    // Temperature on the remaining logits, then the draw.
    double s2 = 0;
    for (int i = 0; i < n; ++i) { cand_p_[i] = std::exp((logits[idx_[i]] - mx) / p_.temperature); s2 += cand_p_[i]; }
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng_) * s2;
    for (int i = 0; i < n; ++i) {
        r -= cand_p_[i];
        if (r <= 0.0) return idx_[i];
    }
    return idx_[n - 1];
}
