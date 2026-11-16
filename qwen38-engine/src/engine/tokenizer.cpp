#include "tokenizer.hpp"

#include <algorithm>
#include <stdexcept>

namespace {

enum : uint8_t { C_L = 1, C_M = 2, C_N = 4, C_S = 8 };

// Class bits of a code point. Exact for ASCII and Latin-1; approximate for the rest (documented in the header).
uint8_t classify(uint32_t cp) {
    if (cp < 0x80) {
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return C_L;
        if (cp >= '0' && cp <= '9') return C_N;
        if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D)) return C_S;
        return 0;
    }
    if (cp < 0x100) {
        if (cp == 0x85 || cp == 0xA0) return C_S;
        if (cp == 0xAA || cp == 0xB5 || cp == 0xBA || (cp >= 0xC0 && cp != 0xD7 && cp != 0xF7)) return C_L;
        if (cp == 0xB2 || cp == 0xB3 || cp == 0xB9 || (cp >= 0xBC && cp <= 0xBE)) return C_N;
        return 0;
    }
    if (cp >= 0x300 && cp <= 0x36F) return C_M;                      // combining diacritics
    if (cp >= 0x483 && cp <= 0x489) return C_M;
    if (cp >= 0x591 && cp <= 0x5C7) return C_M;                      // Hebrew points
    if ((cp >= 0x610 && cp <= 0x61A) || (cp >= 0x64B && cp <= 0x65F) || cp == 0x670) return C_M;  // Arabic marks
    if ((cp >= 0x660 && cp <= 0x669) || (cp >= 0x6F0 && cp <= 0x6F9) || (cp >= 0x966 && cp <= 0x96F)) return C_N;
    if (cp >= 0x1AB0 && cp <= 0x1AFF) return C_M;
    if (cp >= 0x1DC0 && cp <= 0x1DFF) return C_M;
    if (cp >= 0x20D0 && cp <= 0x20FF) return C_M;
    if (cp >= 0xFE00 && cp <= 0xFE0F) return C_M;                    // variation selectors
    if (cp >= 0xFE20 && cp <= 0xFE2F) return C_M;
    if (cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
        cp == 0x205F || cp == 0x3000) return C_S;
    if ((cp >= 0x2150 && cp <= 0x218F) || (cp >= 0x2460 && cp <= 0x249B) || (cp >= 0xFF10 && cp <= 0xFF19)) return C_N;
    if ((cp >= 0x200B && cp <= 0x200F) || (cp >= 0x2010 && cp <= 0x2027) || (cp >= 0x202A && cp <= 0x202E) ||
        (cp >= 0x2030 && cp <= 0x206F)) return 0;                     // general punctuation and format characters
    if (cp >= 0x20A0 && cp <= 0x20CF) return 0;                      // currency
    if ((cp >= 0x2100 && cp <= 0x214F) || (cp >= 0x2190 && cp <= 0x2BFF)) return 0;  // symbols, arrows, math, shapes
    if (cp >= 0x3001 && cp <= 0x3004) return 0;                      // CJK punctuation
    if (cp >= 0x3008 && cp <= 0x3020) return 0;
    if ((cp >= 0xFE30 && cp <= 0xFE6F) || (cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
        (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65)) return 0;
    if ((cp >= 0xE000 && cp <= 0xF8FF) || (cp >= 0xD800 && cp <= 0xDFFF)) return 0;
    if (cp >= 0x1F000 && cp <= 0x1FAFF) return 0;                    // emoji and pictographs
    return C_L;                                                       // everything else treated as a letter
}

// Decodes one UTF-8 code point at s[i]; advances i. Invalid bytes decode as the byte value itself.
uint32_t next_cp(const std::string & s, size_t & i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    auto cont = [&](size_t k) { return i + k < s.size() && (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80; };
    if (c < 0x80) { ++i; return c; }
    if (c >= 0xC2 && c < 0xE0 && cont(1)) {
        const uint32_t cp = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
        i += 2; return cp;
    }
    if (c >= 0xE0 && c < 0xF0 && cont(1) && cont(2)) {
        const uint32_t cp = ((c & 0x0Fu) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
                            (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
        i += 3; return cp;
    }
    if (c >= 0xF0 && c < 0xF5 && cont(1) && cont(2) && cont(3)) {
        const uint32_t cp = ((c & 0x07u) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12) |
                            ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6) |
                            (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
        i += 4; return cp;
    }
    ++i;
    return c;
}

std::string cp_to_utf8(uint32_t cp) {
    std::string r;
    if (cp < 0x80) r += static_cast<char>(cp);
    else if (cp < 0x800) { r += static_cast<char>(0xC0 | (cp >> 6)); r += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        r += static_cast<char>(0xE0 | (cp >> 12)); r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        r += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        r += static_cast<char>(0xF0 | (cp >> 18)); r += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); r += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return r;
}

// GPT-2 byte <-> unicode table: printable bytes map to themselves, the rest to 256, 257, ...
struct ByteMap {
    uint32_t b2u[256];
    int u2b[512];
    ByteMap() {
        for (int i = 0; i < 512; ++i) u2b[i] = -1;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            b2u[b] = direct ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
            u2b[b2u[b]] = b;
        }
    }
};
const ByteMap & bytemap() { static const ByteMap m; return m; }

struct Cp { uint32_t cp; size_t b, e; uint8_t cls; };

// Pre-tokenizer: splits text into pieces following the qwen35 regex (see header).
void pretokenize(const std::string & text, std::vector<std::pair<size_t, size_t>> & out) {
    std::vector<Cp> v;
    for (size_t i = 0; i < text.size();) {
        const size_t b = i;
        const uint32_t cp = next_cp(text, i);
        v.push_back({cp, b, i, classify(cp)});
    }
    const size_t n = v.size();
    auto is_nl = [&](size_t k) { return v[k].cp == '\r' || v[k].cp == '\n'; };
    auto is_letter_or_mark = [&](size_t k) { return (v[k].cls & (C_L | C_M)) != 0; };
    auto lower = [&](size_t k) { const uint32_t c = v[k].cp; return (c >= 'A' && c <= 'Z') ? c + 32 : c; };

    size_t p = 0;
    while (p < n) {
        size_t end = 0;  // exclusive end index of the match (0 = none)

        // 1. contractions
        if (v[p].cp == '\'' && p + 1 < n) {
            const uint32_t a = lower(p + 1);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') end = p + 2;
            else if (p + 2 < n) {
                const uint32_t b2 = lower(p + 2);
                if ((a == 'r' && b2 == 'e') || (a == 'v' && b2 == 'e') || (a == 'l' && b2 == 'l')) end = p + 3;
            }
        }
        // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (!end) {
            size_t q = p;
            const bool prefix_ok = !is_nl(p) && !(v[p].cls & (C_L | C_N));
            if (prefix_ok && p + 1 < n && is_letter_or_mark(p + 1)) q = p + 1;       // consume the optional char
            if (is_letter_or_mark(q)) {
                while (q < n && is_letter_or_mark(q)) ++q;
                end = q;
            }
        }
        // 3. \p{N}
        if (!end && (v[p].cls & C_N)) end = p + 1;
        // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        if (!end) {
            auto punct = [&](size_t k) { return !(v[k].cls & (C_S | C_L | C_M | C_N)); };
            size_t q = p;
            if (v[p].cp == ' ' && p + 1 < n && punct(p + 1)) q = p + 1;
            if (punct(q)) {
                while (q < n && punct(q)) ++q;
                while (q < n && is_nl(q)) ++q;
                end = q;
            }
        }
        // 5-7. whitespace rules
        if (!end && (v[p].cls & C_S)) {
            size_t e = p;
            while (e < n && (v[e].cls & C_S)) ++e;                  // maximal whitespace run [p, e)
            size_t last_nl = n;
            for (size_t k = p; k < e; ++k) if (is_nl(k)) last_nl = k;
            if (last_nl != n) end = last_nl + 1;                    // \s*[\r\n]+
            else if (e == n) end = e;                               // \s+(?!\S) at end of text
            else if (e - p >= 2) end = e - 1;                       // \s+(?!\S): leave the last space for the next word
            else end = e;                                           // \s+
        }
        if (!end) end = p + 1;                                      // cannot happen, keeps the loop finite
        out.emplace_back(v[p].b, end < n ? v[end].b : text.size());
        p = end;
    }
}

}  // namespace

void Tokenizer::init(const GgufIndex & ix) {
    auto m = ix.str_kv.find("tokenizer.ggml.model");
    auto pre = ix.str_kv.find("tokenizer.ggml.pre");
    if (m == ix.str_kv.end() || m->second != "gpt2") throw std::runtime_error("tokenizer: model is not gpt2");
    if (pre == ix.str_kv.end() || pre->second != "qwen35") throw std::runtime_error("tokenizer: pre is not qwen35");
    if (ix.tokens.empty() || ix.token_types.size() != ix.tokens.size()) throw std::runtime_error("tokenizer: bad vocab");
    tokens_ = ix.tokens;
    types_ = ix.token_types;
    tok2id_.reserve(tokens_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); ++i) tok2id_.emplace(tokens_[i], static_cast<int32_t>(i));
    rank_.reserve(ix.merges.size() * 2);
    for (size_t i = 0; i < ix.merges.size(); ++i) rank_.emplace(ix.merges[i], static_cast<int32_t>(i));
    for (size_t i = 0; i < tokens_.size(); ++i) {
        if (types_[i] == 3 || types_[i] == 4) specials_.emplace_back(tokens_[i], static_cast<int32_t>(i));
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
    auto get = [&](const char * k, int64_t d) {
        auto it = ix.u_kv.find(k);
        return it == ix.u_kv.end() ? d : static_cast<int64_t>(it->second);
    };
    eos_id = static_cast<int32_t>(get("tokenizer.ggml.eos_token_id", -1));
    bos_id = static_cast<int32_t>(get("tokenizer.ggml.bos_token_id", -1));
    add_bos = get("tokenizer.ggml.add_bos_token", 0) != 0;
}

void Tokenizer::bpe_word(const std::string & utf8, std::vector<int32_t> & out) const {
    const ByteMap & bm = bytemap();
    std::vector<std::string> sym;
    sym.reserve(utf8.size());
    for (unsigned char c : utf8) sym.push_back(cp_to_utf8(bm.b2u[c]));
    while (sym.size() > 1) {
        int32_t best = INT32_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            auto it = rank_.find(sym[i] + " " + sym[i + 1]);
            if (it != rank_.end() && it->second < best) { best = it->second; at = i; }
        }
        if (best == INT32_MAX) break;
        sym[at] += sym[at + 1];
        sym.erase(sym.begin() + static_cast<std::ptrdiff_t>(at) + 1);
    }
    for (const auto & s : sym) {
        auto it = tok2id_.find(s);
        if (it != tok2id_.end()) { out.push_back(it->second); continue; }
        for (size_t i = 0; i < s.size();) {           // unknown merged symbol: fall back to single characters
            size_t j = i;
            next_cp(s, j);
            auto c = tok2id_.find(s.substr(i, j - i));
            if (c != tok2id_.end()) out.push_back(c->second);
            i = j;
        }
    }
}

void Tokenizer::encode_plain(const std::string & text, std::vector<int32_t> & out) const {
    std::vector<std::pair<size_t, size_t>> parts;
    pretokenize(text, parts);
    for (const auto & pr : parts) bpe_word(text.substr(pr.first, pr.second - pr.first), out);
}

std::vector<int32_t> Tokenizer::encode(const std::string & text, bool parse_special) const {
    std::vector<int32_t> out;
    if (add_bos && bos_id >= 0) out.push_back(bos_id);
    if (!parse_special) {
        encode_plain(text, out);
        return out;
    }
    size_t pos = 0, seg = 0;
    while (pos < text.size()) {
        bool hit = false;
        for (const auto & sp : specials_) {
            if (!sp.first.empty() && text.compare(pos, sp.first.size(), sp.first) == 0) {
                if (pos > seg) encode_plain(text.substr(seg, pos - seg), out);
                out.push_back(sp.second);
                pos += sp.first.size();
                seg = pos;
                hit = true;
                break;
            }
        }
        if (!hit) ++pos;
    }
    if (seg < text.size()) encode_plain(text.substr(seg), out);
    return out;
}

std::string Tokenizer::piece(int32_t id, bool show_special) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return "";
    if (types_[id] == 3 || types_[id] == 4) return show_special ? tokens_[id] : "";
    const ByteMap & bm = bytemap();
    std::string r;
    const std::string & t = tokens_[id];
    for (size_t i = 0; i < t.size();) {
        const uint32_t cp = next_cp(t, i);
        if (cp < 512 && bm.u2b[cp] >= 0) r += static_cast<char>(bm.u2b[cp]);
        else r += cp_to_utf8(cp);
    }
    return r;
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids, bool show_special) const {
    std::string r;
    for (int32_t id : ids) r += piece(id, show_special);
    return r;
}
