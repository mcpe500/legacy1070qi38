// Static arena allocator (skill memory-budget-24gb). One anonymous mapping reserved at startup.
// After freeze(), any further alloc() aborts the process: the token loop must not allocate.
#pragma once
#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

class Arena {
public:
    // Sizing pass: no mapping, alloc() only advances the offset and returns null. used() then gives the exact size the
    // same sequence of alloc() calls needs, so the real arena is sized from the configuration instead of a fixed cap.
    struct Measure {};
    explicit Arena(Measure) : cap_(SIZE_MAX), measuring_(true) {}
    explicit Arena(size_t bytes) : cap_(bytes) {
        void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) { std::fprintf(stderr, "arena: mmap of %zu bytes failed\n", bytes); std::abort(); }
        base_ = static_cast<uint8_t *>(p);
    }
    ~Arena() { if (base_) munmap(base_, cap_); }
    Arena(const Arena &) = delete;
    Arena & operator=(const Arena &) = delete;

    void * alloc(size_t bytes, size_t align = 64) {
        if (frozen_) { std::fprintf(stderr, "arena: alloc after freeze (%zu bytes)\n", bytes); std::abort(); }
        size_t off = (used_ + align - 1) / align * align;
        if (off + bytes > cap_) { std::fprintf(stderr, "arena: out of space (%zu + %zu > %zu)\n", off, bytes, cap_); std::abort(); }
        used_ = off + bytes;
        return measuring_ ? nullptr : base_ + off;
    }
    template <class T> T * alloc_n(size_t n) { return static_cast<T *>(alloc(n * sizeof(T))); }

    void freeze() { frozen_ = true; }
    size_t used() const { return used_; }
    size_t capacity() const { return cap_; }

private:
    uint8_t * base_ = nullptr;
    size_t cap_ = 0, used_ = 0;
    bool frozen_ = false;
    bool measuring_ = false;
};

// Counts global operator new calls made while enabled. Used by the engine to prove that the token loop
// does not touch the heap (the count must be 0 for a forward pass after the first warm-up).
void alloc_guard_enable(bool on);
uint64_t alloc_guard_count();
