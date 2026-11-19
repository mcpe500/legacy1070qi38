// Gated DeltaNet recurrence: op_dn_rec (state column in registers) vs op_dn_rec_ref (state in global memory) on random
// inputs. The outputs and the updated state must be byte-identical for T = 1..5, 8, 64, and the time per call is printed.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ops.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

int main() {
    constexpr int VH = 48, DH = 128, CONV = 10240, TM = 64;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ug(-1.0f, -0.01f), ub(0.0f, 1.0f);
    std::vector<float> conv(static_cast<size_t>(TM) * CONV), gate(TM * VH), beta(TM * VH), z(static_cast<size_t>(TM) * VH * DH), nrm(DH),
        st(static_cast<size_t>(VH) * DH * DH);
    for (float & v : conv) v = nd(rng);
    for (float & v : gate) v = ug(rng);
    for (float & v : beta) v = ub(rng);
    for (float & v : z) v = nd(rng);
    for (float & v : nrm) v = 1.0f + 0.1f * nd(rng);
    for (float & v : st) v = 0.05f * nd(rng);
    float *dc, *dg, *db, *dz, *dn, *s1, *s2, *o1, *o2;
    CK(cudaMalloc(&dc, conv.size() * 4)); CK(cudaMalloc(&dg, gate.size() * 4)); CK(cudaMalloc(&db, beta.size() * 4));
    CK(cudaMalloc(&dz, z.size() * 4)); CK(cudaMalloc(&dn, nrm.size() * 4));
    CK(cudaMalloc(&s1, st.size() * 4)); CK(cudaMalloc(&s2, st.size() * 4));
    CK(cudaMalloc(&o1, z.size() * 4)); CK(cudaMalloc(&o2, z.size() * 4));
    CK(cudaMemcpy(dc, conv.data(), conv.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dz, z.data(), z.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dn, nrm.data(), nrm.size() * 4, cudaMemcpyHostToDevice));
    int fails = 0;
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int T : {1, 2, 3, 4, 5, 8, 64}) {
        CK(cudaMemcpy(s1, st.data(), st.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(s2, st.data(), st.size() * 4, cudaMemcpyHostToDevice));
        CK(op_dn_rec_ref(dc, dg, db, dz, dn, s1, o1, T, 1e-6f, 0));
        CK(op_dn_rec(dc, dg, db, dz, dn, s2, o2, T, 1e-6f, 0));
        CK(cudaDeviceSynchronize());
        std::vector<float> h1(st.size()), h2(st.size()), p1(static_cast<size_t>(T) * VH * DH), p2(p1.size());
        CK(cudaMemcpy(h1.data(), s1, h1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2.data(), s2, h2.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(p1.data(), o1, p1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(p2.data(), o2, p2.size() * 4, cudaMemcpyDeviceToHost));
        const bool ok = std::memcmp(h1.data(), h2.data(), h1.size() * 4) == 0 && std::memcmp(p1.data(), p2.data(), p1.size() * 4) == 0;
        fails += !ok;
        float ms[2];
        for (int k = 0; k < 2; ++k) {
            CK(cudaEventRecord(a));
            for (int r = 0; r < 50; ++r) CK(k ? op_dn_rec(dc, dg, db, dz, dn, s2, o2, T, 1e-6f, 0) : op_dn_rec_ref(dc, dg, db, dz, dn, s1, o1, T, 1e-6f, 0));
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            CK(cudaEventElapsedTime(&ms[k], a, b));
        }
        std::printf("[%s] T=%2d state + output %s; ref %.1f us, new %.1f us per call\n", ok ? "PASS" : "FAIL", T,
                    ok ? "bit-identical" : "DIFFER", 1000.0f * ms[0] / 50, 1000.0f * ms[1] / 50);
    }
    std::printf("dn_rec_ab_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
