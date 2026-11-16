#include <atomic>
#include <cstdlib>
#include <new>

#include "arena.hpp"

namespace {
std::atomic<bool> g_on{false};
std::atomic<uint64_t> g_count{0};

void * counted_alloc(std::size_t n) {
    if (g_on.load(std::memory_order_relaxed)) g_count.fetch_add(1, std::memory_order_relaxed);
    void * p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
}  // namespace

void alloc_guard_enable(bool on) { g_on.store(on); }
uint64_t alloc_guard_count() { return g_count.load(); }

void * operator new(std::size_t n) { return counted_alloc(n); }
void * operator new[](std::size_t n) { return counted_alloc(n); }
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }
