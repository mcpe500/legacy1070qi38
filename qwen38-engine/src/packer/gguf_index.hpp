// GGUF reader: header, key/value metadata, tensor index, and a read-only mmap of the file.
// Tensor bytes stay in the page cache (mmap DDR4); nothing is copied here.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct GgufTensor {
    std::string name;
    uint32_t type = 0;            // ggml_type value as stored in the file
    std::vector<int64_t> ne;      // ne[0] is the fastest-varying dimension (ggml order)
    uint64_t offset = 0;          // relative to data_start
    uint64_t nbytes = 0;          // ggml_row_size(type, ne[0]) * prod(ne[1:])
};

struct GgufIndex {
    std::string path;
    const uint8_t * base = nullptr;
    size_t size = 0;
    uint64_t data_start = 0;
    std::map<std::string, GgufTensor> tensors;
    std::map<std::string, float> f32_kv;
    std::map<std::string, uint64_t> u_kv;                 // all integer and bool scalars (signed values are cast)
    std::map<std::string, std::string> str_kv;            // string scalars (includes tokenizer.ggml.model/pre)
    std::map<std::string, std::vector<int64_t>> arr_kv;   // numeric arrays of at most 64 entries
    std::vector<std::string> tokens, merges;              // tokenizer.ggml.tokens / merges
    std::vector<int32_t> token_types;                     // tokenizer.ggml.token_type

    // Opens and indexes the file. Throws std::runtime_error on malformed input.
    void open(const std::string & p);
    ~GgufIndex();

    const GgufTensor & get(const std::string & name) const;   // throws if missing
    const uint8_t * data(const GgufTensor & t) const { return base + data_start + t.offset; }
    int fd() const { return fd_; }

private:
    int fd_ = -1;
};
