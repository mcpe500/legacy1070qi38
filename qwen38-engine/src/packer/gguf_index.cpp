#include "gguf_index.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

#include "ggml.h"

namespace {

struct Cursor {
    const uint8_t * p;
    const uint8_t * end;

    void need(size_t n) const {
        if (static_cast<size_t>(end - p) < n) throw std::runtime_error("gguf: truncated file");
    }
    template <class T> T rd() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = rd<uint64_t>();
        need(n);
        std::string s(reinterpret_cast<const char *>(p), n);
        p += n;
        return s;
    }
    void skip(size_t n) {
        need(n);
        p += n;
    }
};

size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;          // u8, i8, bool
        case 2: case 3: return 2;                  // u16, i16
        case 4: case 5: case 6: return 4;          // u32, i32, f32
        case 10: case 11: case 12: return 8;       // u64, i64, f64
        default: throw std::runtime_error("gguf: unknown scalar type");
    }
}

void read_value(Cursor & c, uint32_t type, const std::string & key, GgufIndex & ix) {
    if (type == 8) {  // string
        ix.str_kv[key] = c.str();
        return;
    }
    if (type == 9) {  // array
        const uint32_t et = c.rd<uint32_t>();
        const uint64_t n = c.rd<uint64_t>();
        if (et == 8) {
            std::vector<std::string> * dst = nullptr;
            if (key == "tokenizer.ggml.tokens") dst = &ix.tokens;
            else if (key == "tokenizer.ggml.merges") dst = &ix.merges;
            if (dst) dst->reserve(n);
            for (uint64_t i = 0; i < n; ++i) {
                std::string v = c.str();
                if (dst) dst->push_back(std::move(v));
            }
        } else if (key == "tokenizer.ggml.token_type" && et == 5) {
            ix.token_types.resize(n);
            for (uint64_t i = 0; i < n; ++i) ix.token_types[i] = c.rd<int32_t>();
        } else if (n <= 64 && (et == 4 || et == 5 || et == 10 || et == 11)) {
            std::vector<int64_t> v;
            for (uint64_t i = 0; i < n; ++i) {
                if (et == 4) v.push_back(c.rd<uint32_t>());
                else if (et == 5) v.push_back(c.rd<int32_t>());
                else if (et == 10) v.push_back(static_cast<int64_t>(c.rd<uint64_t>()));
                else v.push_back(c.rd<int64_t>());
            }
            ix.arr_kv[key] = std::move(v);
        } else {
            c.skip(scalar_size(et) * n);
        }
        return;
    }
    switch (type) {
        case 0: ix.u_kv[key] = c.rd<uint8_t>(); return;
        case 1: ix.u_kv[key] = static_cast<uint64_t>(static_cast<int64_t>(c.rd<int8_t>())); return;
        case 2: ix.u_kv[key] = c.rd<uint16_t>(); return;
        case 3: ix.u_kv[key] = static_cast<uint64_t>(static_cast<int64_t>(c.rd<int16_t>())); return;
        case 4: ix.u_kv[key] = c.rd<uint32_t>(); return;
        case 5: ix.u_kv[key] = static_cast<uint64_t>(static_cast<int64_t>(c.rd<int32_t>())); return;
        case 6: ix.f32_kv[key] = c.rd<float>(); return;
        case 7: ix.u_kv[key] = c.rd<uint8_t>(); return;
        case 10: ix.u_kv[key] = c.rd<uint64_t>(); return;
        case 11: ix.u_kv[key] = static_cast<uint64_t>(c.rd<int64_t>()); return;
        default: c.skip(scalar_size(type)); return;
    }
}

}  // namespace

void GgufIndex::open(const std::string & p) {
    path = p;
    fd_ = ::open(p.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("gguf: cannot open " + p);
    struct stat st;
    if (fstat(fd_, &st) != 0) throw std::runtime_error("gguf: fstat failed");
    size = static_cast<size_t>(st.st_size);
    void * m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (m == MAP_FAILED) throw std::runtime_error("gguf: mmap failed");
    base = static_cast<const uint8_t *>(m);

    Cursor c{base, base + size};
    if (size < 24 || std::memcmp(c.p, "GGUF", 4) != 0) throw std::runtime_error("gguf: bad magic");
    c.p += 4;
    (void) c.rd<uint32_t>();  // version (3 expected; not checked here)
    const uint64_t n_tensors = c.rd<uint64_t>();
    const uint64_t n_kv = c.rd<uint64_t>();

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = c.str();
        const uint32_t type = c.rd<uint32_t>();
        read_value(c, type, key, *this);
    }

    const uint64_t alignment = u_kv.count("general.alignment") ? u_kv["general.alignment"] : 32;

    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensor t;
        t.name = c.str();
        const uint32_t n_dims = c.rd<uint32_t>();
        if (n_dims == 0 || n_dims > 4) throw std::runtime_error("gguf: bad n_dims for " + t.name);
        for (uint32_t d = 0; d < n_dims; ++d) t.ne.push_back(static_cast<int64_t>(c.rd<uint64_t>()));
        for (uint32_t d = n_dims; d < 4; ++d) t.ne.push_back(1);
        t.type = c.rd<uint32_t>();
        t.offset = c.rd<uint64_t>();
        const auto gt = static_cast<enum ggml_type>(t.type);
        int64_t rows = 1;
        for (int d = 1; d < 4; ++d) rows *= t.ne[d];
        t.nbytes = static_cast<uint64_t>(ggml_row_size(gt, t.ne[0])) * static_cast<uint64_t>(rows);
        tensors[t.name] = t;
    }

    const uint64_t pos = static_cast<uint64_t>(c.p - base);
    data_start = (pos + alignment - 1) / alignment * alignment;
    for (const auto & kv : tensors) {
        if (data_start + kv.second.offset + kv.second.nbytes > size) {
            throw std::runtime_error("gguf: tensor " + kv.first + " runs past end of file");
        }
    }
}

GgufIndex::~GgufIndex() {
    if (base) munmap(const_cast<uint8_t *>(base), size);
    if (fd_ >= 0) ::close(fd_);
}

const GgufTensor & GgufIndex::get(const std::string & name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("gguf: missing tensor " + name);
    return it->second;
}
