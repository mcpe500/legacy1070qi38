// DDR4 read bandwidth probe: N threads sum a large buffer with AVX2 loads (with and without software prefetch).
// Build: g++ -O3 -march=znver1 -pthread tests/tools/membw.cpp -o build/membw   Run: taskset -c 0-3 build/membw [GiB] [threads]
#include <immintrin.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static double run(const float * p, size_t n, int nt, bool prefetch) {
    std::vector<std::thread> th;
    std::vector<float> out(static_cast<size_t>(nt));
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < nt; ++k) {
        th.emplace_back([&, k] {
            const size_t b = n / nt * k, e = (k == nt - 1) ? n : n / nt * (k + 1);
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            for (size_t i = b; i + 16 <= e; i += 16) {
                if (prefetch) _mm_prefetch(reinterpret_cast<const char *>(p + i + 512), _MM_HINT_T0);
                s0 = _mm256_add_ps(s0, _mm256_loadu_ps(p + i));
                s1 = _mm256_add_ps(s1, _mm256_loadu_ps(p + i + 8));
            }
            float t[8];
            _mm256_storeu_ps(t, _mm256_add_ps(s0, s1));
            out[static_cast<size_t>(k)] = t[0] + t[7];
        });
    }
    for (auto & t : th) t.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    volatile float sink = out[0];
    (void) sink;
    return static_cast<double>(n) * sizeof(float) / s / 1e9;
}

int main(int argc, char ** argv) {
    const double gib = argc > 1 ? std::atof(argv[1]) : 2.0;
    const int nt = argc > 2 ? std::atoi(argv[2]) : 4;
    const size_t n = static_cast<size_t>(gib * 1024 * 1024 * 1024 / sizeof(float));
    std::vector<float> a(n, 1.0f);
    for (int r = 0; r < 3; ++r)
        std::printf("threads %d: plain %.1f GB/s, prefetch %.1f GB/s\n", nt, run(a.data(), n, nt, false), run(a.data(), n, nt, true));
    return 0;
}
