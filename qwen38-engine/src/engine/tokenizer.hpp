// Byte-level BPE tokenizer for tokenizer.ggml.model = "gpt2", tokenizer.ggml.pre = "qwen35".
// Built only from GGUF metadata (tokens, merges, token_type). No external library.
//
// Pre-tokenizer regex (from llama.cpp src/llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_QWEN35), matched by hand:
//   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ |\p{N} | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   |\s*[\r\n]+ |\s+(?!\S) |\s+
// Unicode classes (L, M, N, whitespace) are exact for ASCII and Latin-1 and APPROXIMATED above that
// (see classify() in tokenizer.cpp). tests/tokenizer_parity.py measures the difference against llama-tokenize.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf_index.hpp"

class Tokenizer {
public:
    // Builds the vocabulary tables. Throws std::runtime_error when the GGUF is not a qwen35 BPE vocabulary.
    void init(const GgufIndex & ix);

    // Encodes UTF-8 text. When parse_special is true, control and user-defined tokens such as
    // <|im_start|> found in the text become single tokens.
    std::vector<int32_t> encode(const std::string & text, bool parse_special) const;

    // Bytes of one token. Control tokens return their literal text only when show_special is true.
    std::string piece(int32_t id, bool show_special) const;
    std::string decode(const std::vector<int32_t> & ids, bool show_special) const;

    int32_t eos_id = -1;
    int32_t bos_id = -1;
    bool add_bos = false;
    size_t n_vocab() const { return tokens_.size(); }
    bool is_control(int32_t id) const { return id >= 0 && static_cast<size_t>(id) < types_.size() && types_[id] == 3; }

private:
    void bpe_word(const std::string & utf8, std::vector<int32_t> & out) const;
    void encode_plain(const std::string & text, std::vector<int32_t> & out) const;

    std::vector<std::string> tokens_;
    std::vector<int32_t> types_;
    std::unordered_map<std::string, int32_t> tok2id_;
    std::unordered_map<std::string, int32_t> rank_;          // "a b" -> merge rank
    std::vector<std::pair<std::string, int32_t>> specials_;   // text -> id, longest first
};
