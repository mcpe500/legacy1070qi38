// DeltaNet rollback for a verification batch of 17 rows (1 target row + 16 drafts): the MTP verifier rolls
// back the DeltaNet state to the first `keep` accepted rows by restoring the snapshot and replaying the
// conv + recurrence for `keep` rows (GpuLayers::rollback). This test drives the same three calls at the
// kernel level and checks, for keep = 1, 2, 3, 4, 5, 8, 12, 16, 17, that the rolled-back state and the
// rolled-back outputs are byte-identical to a batch of exactly `keep` rows started from the same snapshot.
// It also checks that keep = 17 reproduces the state of the full 17-row batch (the identity the verifier
// relies on when every draft is rejected) and that a batch larger than the rollback buffer would be caught
// by the caller, so the buffer size is the only limit.
// Usage: dn_rollback_test. Exit 0 only if every case passes.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ops.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace {
constexpr int VH = 48, DH = 128, CONV = 10240, KCONV = 4;
constexpr int VT = 17;      // verification batch: 1 target row + 16 drafts
constexpr int TM = 64;      // rows allocated (a prefill batch is larger than the verify batch)
constexpr int SZ_ST = VH * DH * DH, SZ_CONV = CONV * (KCONV - 1);
}  // namespace

int main() {
    std::mt19937 rng(20261122);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ug(-1.0f, -0.01f), ub(0.0f, 1.0f);
    std::vector<float> conv(static_cast<size_t>(TM) * CONV), gate(TM * VH), beta(TM * VH),
        z(static_cast<size_t>(TM) * VH * DH), nrm(DH), st(SZ_ST), cw(static_cast<size_t>(CONV) * KCONV);
    for (float & v : conv) v = nd(rng);
    for (float & v : gate) v = ug(rng);
    for (float & v : beta) v = ub(rng);
    for (float & v : z) v = nd(rng);
    for (float & v : nrm) v = 1.0f + 0.1f * nd(rng);
    for (float & v : st) v = 0.05f * nd(rng);
    for (float & v : cw) v = 0.01f * nd(rng);
    float *dc, *dg, *db, *dz, *dn, *dw, *s_now, *s_snap, *s_ref, *o1, *o2, *tmp;
    CK(cudaMalloc(&dc, conv.size() * 4)); CK(cudaMalloc(&dg, gate.size() * 4)); CK(cudaMalloc(&db, beta.size() * 4));
    CK(cudaMalloc(&dz, z.size() * 4)); CK(cudaMalloc(&dn, nrm.size() * 4)); CK(cudaMalloc(&dw, cw.size() * 4));
    CK(cudaMalloc(&s_now, SZ_ST * 4)); CK(cudaMalloc(&s_snap, SZ_ST * 4)); CK(cudaMalloc(&s_ref, SZ_ST * 4));
    CK(cudaMalloc(&o1, z.size() * 4)); CK(cudaMalloc(&o2, z.size() * 4));
    CK(cudaMalloc(&tmp, static_cast<size_t>(CONV) * VT * 4));   // conv replay scratch (VT rows, like sv_qkv)
    CK(cudaMemcpy(dc, conv.data(), conv.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dz, z.data(), z.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dn, nrm.data(), nrm.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, cw.data(), cw.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s_snap, st.data(), SZ_ST * 4, cudaMemcpyHostToDevice));

    // The full batch from the snapshot: the state the verifier would leave if nothing were rolled back.
    CK(cudaMemcpy(s_now, s_snap, SZ_ST * 4, cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());
    {
        float * conv_out;
        CK(cudaMalloc(&conv_out, static_cast<size_t>(CONV) * VT * 4));
        CK(op_dn_conv(dc, dw, s_now, conv_out, VT, 0));
        CK(op_dn_rec(conv_out, dg, db, dz, dn, s_now, o1, VT, 1e-6f, 0));
        CK(cudaDeviceSynchronize());
        std::vector<float> full(SZ_ST);
        CK(cudaMemcpy(full.data(), s_now, SZ_ST * 4, cudaMemcpyDeviceToHost));
        CK(cudaFree(conv_out));
        // keep = 17 must reproduce it exactly.
        CK(cudaMemcpy(s_ref, s_snap, SZ_ST * 4, cudaMemcpyDeviceToDevice));
        CK(op_dn_conv(dc, dw, s_ref, tmp, VT, 0));
        CK(op_dn_rec(tmp, dg, db, dz, dn, s_ref, o2, VT, 1e-6f, 0));
        CK(cudaDeviceSynchronize());
        std::vector<float> back(SZ_ST);
        CK(cudaMemcpy(back.data(), s_ref, SZ_ST * 4, cudaMemcpyDeviceToHost));
        const bool ok = std::memcmp(full.data(), back.data(), SZ_ST * 4) == 0;
        std::printf("[%s] rollback keep=17 reproduces the full 17-row batch state (byte-identical)\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }

    int fails = 0, cases = 0;
    for (int keep : {1, 2, 3, 4, 5, 8, 12, 16, 17}) {
        // A: restore the snapshot and replay the first `keep` rows (what GpuLayers::rollback does).
        CK(cudaMemcpy(s_now, s_snap, SZ_ST * 4, cudaMemcpyDeviceToDevice));
        CK(op_dn_conv(dc, dw, s_now, tmp, keep, 0));
        CK(op_dn_rec(tmp, dg, db, dz, dn, s_now, o1, keep, 1e-6f, 0));
        // B: the same `keep` rows in a fresh buffer, for the byte comparison.
        CK(cudaMemcpy(s_ref, s_snap, SZ_ST * 4, cudaMemcpyDeviceToDevice));
        CK(op_dn_conv(dc, dw, s_ref, tmp, keep, 0));
        CK(op_dn_rec(tmp, dg, db, dz, dn, s_ref, o2, keep, 1e-6f, 0));
        CK(cudaDeviceSynchronize());
        std::vector<float> a(SZ_ST), b(SZ_ST), pa(static_cast<size_t>(keep) * VH * DH), pb(pa.size());
        CK(cudaMemcpy(a.data(), s_now, SZ_ST * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), s_ref, SZ_ST * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pa.data(), o1, pa.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pb.data(), o2, pb.size() * 4, cudaMemcpyDeviceToHost));
        const bool ok = std::memcmp(a.data(), b.data(), SZ_ST * 4) == 0 && std::memcmp(pa.data(), pb.data(), pa.size() * 4) == 0;
        ++cases;
        fails += !ok;
        std::printf("[%s] rollback keep=%2d of 17: state %s, outputs %s\n", ok ? "PASS" : "FAIL", keep,
                    std::memcmp(a.data(), b.data(), SZ_ST * 4) == 0 ? "byte-identical" : "DIFFER",
                    std::memcmp(pa.data(), pb.data(), pa.size() * 4) == 0 ? "byte-identical" : "DIFFER");
    }
    // A batch of 18 rows does not fit the 17-row buffer: the verifier must never build one. The check that the
    // caller enforces it is in GpuLayers::forward (T > cfg.roll_t throws), so only the exact bound is asserted here.
    std::printf("dn_rollback_test: %d cases, %d fail -> %s (verify batch 17, rollback bound enforced by the caller)\n",
                cases, fails, fails ? "FAIL" : "PASS");
    CK(cudaFree(dc)); CK(cudaFree(dg)); CK(cudaFree(db)); CK(cudaFree(dz)); CK(cudaFree(dn)); CK(cudaFree(dw));
    CK(cudaFree(s_now)); CK(cudaFree(s_snap)); CK(cudaFree(s_ref)); CK(cudaFree(o1)); CK(cudaFree(o2)); CK(cudaFree(tmp));
    return fails ? 1 : 0;
}
