// GPU execution of the first n layers of Qwen3.8-27B on a GTX 1070 Ti (sm_61). One VRAM arena holds the quantized weights
// (kept in their GGUF block format), the Gated DeltaNet state, the iso3 KV cache and every scratch buffer; nothing is
// allocated on the device after the constructor returns. No CUDA types appear here so ordinary C++ code can include it.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "gguf_index.hpp"

struct GpuConfig {
    int n_layers = 24;     // layers 0 .. n_layers-1 run on the GPU
    int ctx_max = 2048;    // KV cache positions
    int t_max = 512;       // largest token batch per call
    bool kv_iso3 = true;   // iso3 (IsoQuant 3-bit) KV cache for the attention layers on the GPU; false keeps FP32
    // Quantization types whose GEMV uses Q8_1 activations and __dp4a. Bit n is ggml type id n (Q4_K = 12, Q5_K = 13,
    // Q6_K = 14, Q3_K = 11, IQ4_NL = 20, IQ3_S = 21, IQ4_XS = 23). A type whose bit is clear dequantizes to FP32 in the
    // kernel (option A). Q8_0 always uses its own __dp4a kernel.
    uint32_t dp4a_mask = 0;
    // Prefill streaming: layers n_layers..63 can run on the GPU for large first batches. Their weights are uploaded one
    // layer at a time into a single staging buffer (sized for the largest layer part). Needs n_layers < 64.
    bool stream_prefill = false;
    // DeltaNet rollback (MTP verification): largest batch whose DeltaNet state can be rolled back (0 = no buffers).
    int roll_t = 0;
    // Replay the whole layer stack as a CUDA graph for batches of 1..4 tokens (decode and MTP verification).
    bool use_graph = false;
    // Streamed prefill: register the CPU-layer weights of the mmap as pinned memory and overlap upload with compute.
    bool pin_stream = true;
    // Ternary Bonsai mode (PTQ1_0 type 143 / Q2_0 type 42 weights in the Hadamard-rotated basis): every projected
    // input is rotated with these sign vectors before its GEMV. Null = K-quant mode (no rotation).
    const float * signs_5120 = nullptr, * signs_6144 = nullptr, * signs_17408 = nullptr;
    bool gdn_v_grouped = false;   // prism.hadamard.gdn_v_grouped: permute the DeltaNet value heads for ssm_out
};

// "all", "none", or a comma list of ggml type ids or names (q4_k, q5_k, q6_k, q3_k, iq4_nl, iq3_s, iq4_xs).
uint32_t parse_dp4a_mask(const char * list);

class GpuLayers {
public:
    GpuLayers(const GgufIndex & ix, const GpuConfig & cfg, float rms_eps);
    ~GpuLayers();
    GpuLayers(const GpuLayers &) = delete;
    GpuLayers & operator=(const GpuLayers &) = delete;

    // x_host: [T][5120] FP32 residual stream (pinned memory is fastest). Runs the GPU layers and writes the result back
    // in place. `tap`, when given, is called with the host copy of the stream after every layer (parity tests only).
    void forward(float * x_host, int T, int pos0, const std::function<void(int, const float *, int)> * tap);

    // Zeroes the DeltaNet state (the KV cache is overwritten as positions advance).
    void reset();

    // ---- DeltaNet snapshot / rollback (needs cfg.roll_t > 0). With snapshots on, forward() of at most roll_t tokens keeps
    // the DeltaNet state from before the call plus the inputs of the recurrence; rollback(keep) restores that state and
    // replays the first `keep` tokens, so the state equals the one of a run that only processed those tokens.
    void set_snapshot(bool on);
    void rollback(int keep);

    // ---- Prefill streaming (needs cfg.stream_prefill). The state and the FP32 KV of the streamed layer live on the host in
    // the CPU layouts (conv_state [10240][3], ssm_state [48][128][128] as M[j][i], K and V FP32 [pos][4][256]); for pos0 > 0
    // they are uploaded first, and the new state / KV rows are copied back.
    bool stream_ready() const;
    void stream_begin(const float * x_host, int T);
    void stream_layer(int il, int T, int pos0, float * conv_state_host, float * ssm_state_host, float * k_host, float * v_host);
    void stream_peek(float * x_host, int T);
    // MTP prefill on the GPU (needs prefill streaming): only the MTP KV is needed for later drafts, so this runs eh_proj, attn_norm,
    // the K and V projections, k_norm and RoPE for T rows of concat_host [T][10240] (= [rmsnorm(emb, enorm) | rmsnorm(h, hnorm)])
    // and writes FP32 K and V rows pos0 .. pos0+T-1 into k_host / v_host ([pos][4][256]).
    void mtp_kv(const float * concat_host, int T, int pos0, float * k_host, float * v_host);   // copies the current stream to the host (parity taps)
    void stream_end(float * x_host, int T);

    size_t vram_bytes() const;     // device bytes owned by this object (arena plus the Q8 scratch)
    size_t weight_bytes() const;   // quantized weights and small vectors
    size_t kv_bytes() const;       // KV cache of the GPU attention layers
    int n_layers() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
