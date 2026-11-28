// CPU unit test of the DFly2 shared-KV selection core (issue #70 E-D2b). Compiles the same header the device
// kernel uses (DFLY2_HOST_ONLY strips the __host__ __device__ qualifiers) and checks: deterministic top-32
// against a sort reference, tie handling, window and sink inclusion, causal clamping, ascending deduplicated
// lists, the kStride bound, group isolation in the strided layout, and that the flag-based expansion used by
// the device kernel agrees with the serial reference builder.
// Build: g++ -O2 -std=c++17 -DDFLY2_HOST_ONLY -Isrc/cuda tests/dfly2_select_test.cpp -o build/dfly2_select_test
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "dfly2_select.cuh"

#define CK(c, msg) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); } } while (0)

int main() {
    std::mt19937 rng(20261128);
    std::uniform_real_distribution<float> ud(-20.0f, 20.0f);
    constexpr int G = 4, NB = 2048;   // layout stride 4 groups; block count at ctx 131072
    std::vector<float> bmax(static_cast<size_t>(NB) * G);
    std::vector<int> list(dfly2::kStride), list2(dfly2::kStride);

    // 1. top-32 vs sort reference, with heavy ties (scores quantized to 0.25 steps)
    for (int trial = 0; trial < 200; ++trial) {
        const int nb = 1 + static_cast<int>(rng() % NB);
        const int g = static_cast<int>(rng() % G);
        for (int b = 0; b < nb; ++b) bmax[static_cast<size_t>(b) * G + g] = std::round(ud(rng) * 4.0f) / 4.0f;
        std::vector<std::pair<float, int>> sc;
        for (int b = 0; b < nb; ++b) sc.push_back({bmax[static_cast<size_t>(b) * G + g], b});
        std::sort(sc.begin(), sc.end(), [](const auto & x, const auto & y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
        std::vector<char> ref(nb, 0);
        for (int i = 0; i < nb && i < dfly2::kBlocks; ++i) ref[sc[i].second] = 1;
        for (int b = 0; b < nb; ++b) {
            const bool got = dfly2::topk_selected(bmax.data(), nb, G, g, b);
            CK(got == static_cast<bool>(ref[b]), "top-32 mismatch vs sort reference (with ties)");
        }
    }

    // 2. build_list invariants at a realistic size: ascending, unique, clamped, window + sink present, count bound
    for (int trial = 0; trial < 200; ++trial) {
        const int nb = 33 + static_cast<int>(rng() % (NB - 33));
        const int pos_max = std::min(nb * 64 - 1, 1 + static_cast<int>(rng() % (nb * 64)));
        const int g = static_cast<int>(rng() % G);
        for (int b = 0; b < nb; ++b) bmax[static_cast<size_t>(b) * G + g] = ud(rng);
        const int n = dfly2::build_list(bmax.data(), nb, G, g, pos_max, list.data());
        CK(n > 0 && n <= dfly2::kStride, "list count within bounds");
        for (int i = 1; i < n; ++i) CK(list[i - 1] < list[i], "list strictly ascending (dedup + sort)");
        CK(list[0] == 0 && list[n - 1] <= pos_max, "sink present and causal clamp");
        const int wfirst = std::max(0, pos_max + 1 - dfly2::kWindow);
        CK(list[n - 1] == pos_max, "last window position kept");
        CK(std::find(list.begin(), list.begin() + n, wfirst) != list.begin() + n, "window start covered");
        // expected count: sum over selected blocks of the live positions of that block
        int want = 0;
        for (int b = 0; b * 64 <= pos_max; ++b)
            if (dfly2::always_selected(b, pos_max) || dfly2::topk_selected(bmax.data(), nb, G, g, b)) {
                const int hi = std::min(pos_max, (b + 1) * 64 - 1);
                want += hi - b * 64 + 1;
            }
        CK(n == want, "count equals per-block live positions");
        // flag expansion (device path) must agree with the serial builder
        std::vector<unsigned char> flags(nb, 0);
        for (int b = 0; b < nb; ++b) flags[b] = dfly2::topk_selected(bmax.data(), nb, G, g, b) ? 1 : 0;
        const int n2 = dfly2::expand_list(flags.data(), nb, pos_max, list2.data());
        CK(n2 == n && std::memcmp(list.data(), list2.data(), sizeof(int) * n) == 0, "flag expansion == serial builder");
    }

    // 3. small cases: fewer blocks than kBlocks keeps everything; one partial block; pos_max inside block 0
    {
        const int nb = 10, g = 2, pos_max = nb * 64 - 1;
        for (int b = 0; b < nb; ++b) bmax[static_cast<size_t>(b) * G + g] = ud(rng);
        const int n = dfly2::build_list(bmax.data(), nb, G, g, pos_max, list.data());
        CK(n == nb * 64, "all blocks kept when nblocks < kBlocks");
        CK(list[0] == 0 && list[n - 1] == pos_max, "full range covered");
    }
    {
        const int n = dfly2::build_list(bmax.data(), NB, G, 0, 5, list.data());
        CK(n == 6 && list[5] == 5, "pos_max inside block 0 clamps the list");
    }
    {
        const int n = dfly2::build_list(bmax.data(), NB, G, 1, 63, list.data());
        CK(n == 64 && list[63] == 63, "exactly block 0 at the 64-position boundary");
    }

    // 4. determinism: the same scores must give the same list, run to run
    {
        const int nb = 512, g = 3;
        for (int b = 0; b < nb; ++b) bmax[static_cast<size_t>(b) * G + g] = ud(rng);
        const int n1 = dfly2::build_list(bmax.data(), nb, G, g, nb * 64 - 1, list.data());
        const int n2 = dfly2::build_list(bmax.data(), nb, G, g, nb * 64 - 1, list2.data());
        CK(n1 == n2 && std::memcmp(list.data(), list2.data(), sizeof(int) * n1) == 0, "deterministic rebuild");
    }

    // 5. group isolation: changing one group's scores must not move another group's list
    {
        const int nb = 256, pos_max = nb * 64 - 1;
        for (size_t i = 0; i < bmax.size(); ++i) bmax[i] = ud(rng);
        int n0 = dfly2::build_list(bmax.data(), nb, G, 0, pos_max, list.data());
        std::vector<int> snap(list.begin(), list.begin() + n0);
        for (int b = 0; b < nb; ++b) bmax[static_cast<size_t>(b) * G + 2] = 100.0f;   // boost group 2 only
        const int n1 = dfly2::build_list(bmax.data(), nb, G, 0, pos_max, list2.data());
        CK(n1 == n0 && std::memcmp(snap.data(), list2.data(), sizeof(int) * n0) == 0, "group lists are independent");
    }

    // 6. worst-case bound: 32 disjoint top blocks + full window + sink never exceed kStride
    {
        const int nb = NB, pos_max = nb * 64 - 1;
        for (int b = 0; b < nb; ++b) bmax[static_cast<size_t>(b) * G + 0] = 0.0f;
        for (int b = 100; b < 100 + dfly2::kBlocks; ++b) bmax[static_cast<size_t>(b) * G + 0] = 1000.0f;   // top-32 = blocks 100..131
        const int n = dfly2::build_list(bmax.data(), nb, G, 0, pos_max, list.data());
        CK(n <= dfly2::kStride, "worst case within kStride");
        CK(n == dfly2::kBlocks * 64 + dfly2::kWindow + 64, "disjoint top/window/sink adds up exactly");
    }

    std::printf("[PASS] dfly2_select: top-32 ties, window/sink, clamp, stride bound, group isolation, flag expansion (%d trials x2)\n", 200);
    return 0;
}
