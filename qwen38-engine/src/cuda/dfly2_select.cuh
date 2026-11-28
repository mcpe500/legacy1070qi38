// DFly2-style shared-KV block selection (issue #70 E-D2b). Portable core: the exact same predicates run in the
// device selector kernel and in the host unit test, so the CPU test exercises the logic the GPU will execute.
// Layout of the sideband scores (written by the decode kernel, op_attention_set_blockmax): one float per
// 64-position block per KV group, (block * group_stride + group). The selection of one KV group is
//   top-32 blocks by score (ties: lower block index wins, deterministic)  UNION  window: the blocks covering the
//   last kWindow positions  UNION  sink: block 0 (attention sinks live at the context start)
// expanded to an ascending, deduplicated position list clamped to pos_max (inclusive). The list of every KV
// group lives at lists[group * kStride]; it holds at most 32*64 + 128 + 64 positions.
#pragma once

#ifdef DFLY2_HOST_ONLY
#define DFLY2_HD
#else
#define DFLY2_HD __host__ __device__
#endif

namespace dfly2 {

constexpr int kBlocks = 32;    // top-k blocks per KV group (gate: softmax-mass coverage >= 0.80 @ 32K, issue #70)
constexpr int kWindow = 128;   // recent positions kept no matter what the scores say
// 32*64 top-k + window span (128 positions can straddle three 64-blocks: up to 191 live positions) + 64 sink block
constexpr int kStride = kBlocks * 64 + kWindow + 128;   // 2368: worst-case distinct positions of one list

// sink block 0 plus the blocks overlapping the last kWindow positions: selected regardless of score
DFLY2_HD inline bool always_selected(int b, int pos_max) {
    if (b == 0) return true;
    const int wfirst = pos_max + 1 - kWindow > 0 ? pos_max + 1 - kWindow : 0;
    return (b + 1) * 64 > wfirst;
}

// deterministic top-k with ties: block b is in the top kBlocks iff (#blocks with a strictly better score)
// + (#blocks with an equal score and a lower index) < kBlocks
DFLY2_HD inline bool topk_selected(const float * bmax, int nblocks, int gstride, int g, int b) {
    const float s = bmax[static_cast<int64_t>(b) * gstride + g];
    int rank = 0;
    for (int c = 0; c < nblocks; ++c) {
        const float t = bmax[static_cast<int64_t>(c) * gstride + g];
        if (t > s) ++rank;
        else if (t == s && c < b) ++rank;
        if (rank >= kBlocks) return false;
    }
    return true;
}

// serial reference builder (host tests; the device uses the kernel that precomputes the top-k flags in parallel)
DFLY2_HD inline int build_list(const float * bmax, int nblocks, int gstride, int g, int pos_max, int * out) {
    int n = 0;
    for (int b = 0; b < nblocks; ++b) {
        if (static_cast<int64_t>(b) * 64 > pos_max) break;   // ascending: every later block is fully beyond pos_max
        const bool sel = always_selected(b, pos_max) || topk_selected(bmax, nblocks, gstride, g, b);
        if (!sel) continue;
        for (int p = b * 64; p <= pos_max && p < (b + 1) * 64; ++p) out[n++] = p;
    }
    return n;
}

// expand from precomputed top-k flags (the device path: kernel fills flags[topk], this adds sink + window)
DFLY2_HD inline int expand_list(const unsigned char * topk_flags, int nblocks, int pos_max, int * out) {
    int n = 0;
    for (int b = 0; b < nblocks; ++b) {
        if (static_cast<int64_t>(b) * 64 > pos_max) break;
        if (!(topk_flags[b] || always_selected(b, pos_max))) continue;
        for (int p = b * 64; p <= pos_max && p < (b + 1) * 64; ++p) out[n++] = p;
    }
    return n;
}

}  // namespace dfly2
