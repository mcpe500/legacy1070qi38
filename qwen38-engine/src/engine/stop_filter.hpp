// Stop strings for the HTTP server (OpenAI `stop`): generation ends at the first occurrence of any stop string, and neither
// the stop string nor anything after it is sent. Text that could still turn into a stop string (a suffix of the output that
// is a prefix of a stop string) is held back until the next piece decides it, so a stop string split across tokens or SSE
// chunks is never leaked. The cut is always on a UTF-8 boundary because the stop strings are valid UTF-8.
//
// With thinking on, the filter starts after the reasoning block: everything up to and including "</think>" passes through
// unchanged (OpenAI applies `stop` to the answer content). Header-only, no allocation in steady state beyond the held text.
#pragma once
#include <algorithm>
#include <string>
#include <vector>

class StopFilter {
public:
    StopFilter(std::vector<std::string> stops, bool after_think) : stops_(std::move(stops)), active_(!after_think) {}

    bool enabled() const { return !stops_.empty(); }
    bool hit() const { return hit_; }
    const std::string & matched() const { return matched_; }

    // Feeds the next piece of generated text; returns the text that can be sent now. After hit() is true the caller stops
    // generating and ignores further output.
    std::string push(const std::string & piece) {
        if (hit_) return {};
        if (stops_.empty()) return piece;
        if (active_) return scan(piece);
        // Still inside the reasoning block: pass through, watching for the closing tag (which can be split across pieces).
        static const std::string tag = "</think>";
        watch_ += piece;
        const size_t k = watch_.find(tag);
        if (k == std::string::npos) {
            if (watch_.size() > tag.size()) watch_.erase(0, watch_.size() - (tag.size() - 1));
            return piece;
        }
        active_ = true;
        const std::string after = watch_.substr(k + tag.size());   // part of this piece after the tag
        watch_.clear();
        const std::string before = piece.substr(0, piece.size() - after.size());
        return before + scan(after);
    }

    // End of generation without a stop string: returns the held-back text.
    std::string flush() {
        if (hit_) return {};
        std::string r;
        r.swap(buf_);
        return r;
    }

private:
    std::string scan(const std::string & s) {
        buf_ += s;
        size_t best = std::string::npos;
        for (const std::string & st : stops_) {
            const size_t p = buf_.find(st);
            if (p != std::string::npos && (best == std::string::npos || p < best)) { best = p; matched_ = st; }
        }
        if (best != std::string::npos) {
            hit_ = true;
            std::string out = buf_.substr(0, best);
            buf_.clear();
            return out;
        }
        size_t hold = 0;   // longest suffix of buf_ that is a proper prefix of a stop string
        for (const std::string & st : stops_) {
            const size_t m = std::min(buf_.size(), st.size() - 1);
            for (size_t len = m; len > hold; --len) {
                if (buf_.compare(buf_.size() - len, len, st, 0, len) == 0) { hold = len; break; }
            }
        }
        std::string out = buf_.substr(0, buf_.size() - hold);
        buf_.erase(0, out.size());
        return out;
    }

    std::vector<std::string> stops_;
    bool active_;
    bool hit_ = false;
    std::string buf_, watch_, matched_;
};
