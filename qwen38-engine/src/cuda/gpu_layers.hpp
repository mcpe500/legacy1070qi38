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
    // Largest verification batch the attention partial buffer and the rollback buffers are sized for (1 target row
    // plus the accepted drafts). 17 = a Drafter v2 batch of 16 drafts + 1. The buffer grows with it, so the engine
    // picks the largest value whose whole plan still fits vram_budget_mib and falls back to 8 or 5 otherwise.
    int verify_t = 5;
    // Replay the whole layer stack as a CUDA graph for batches of 1..4 tokens (decode and MTP verification).
    bool use_graph = false;
    // Streamed prefill: register the CPU-layer weights of the mmap as pinned memory and overlap upload with compute.
    bool pin_stream = true;
    // Ternary Bonsai mode (PTQ1_0 type 143 / Q2_0 type 42 weights in the Hadamard-rotated basis): every projected
    // input is rotated with these sign vectors before its GEMV. Null = K-quant mode (no rotation).
    const float * signs_5120 = nullptr, * signs_6144 = nullptr, * signs_17408 = nullptr;
    bool gdn_v_grouped = false;   // prism.hadamard.gdn_v_grouped: permute the DeltaNet value heads for ssm_out
    // Device memory cap of the whole process (CUDA context included). The constructor refuses a plan that would exceed it.
    double vram_budget_mib = 7373.0;
    // LM head on the GPU (output_norm + output.weight, rotated for a ternary head): head() computes the logits of the last
    // rows from the device copy of the stream after the last GPU layer. Only meaningful with n_layers = 64.
    bool head = false;
    // MTP draft head (blk.64) on the GPU: its decoder layer, nextn.eh_proj and shared_head_norm live in the arena, its KV in
    // the format of the other attention layers. Needs head = true (the draft logits use the GPU LM head).
    bool mtp = false;
    // Index holding the blk.64 tensors: null or the main index, or an --mtp-gguf sidecar mmapped separately.
    const GgufIndex * mtp_ix = nullptr;
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

    // Sizing without allocation: device bytes of the arena for this configuration, and the projected device memory of the
    // process (memory in use now, CUDA context included, plus the arena and the scratch buffers) in MiB.
    static size_t plan_bytes(const GgufIndex & ix, const GpuConfig & cfg);
    static double projected_mib(const GgufIndex & ix, const GpuConfig & cfg);

    // ---- State snapshot to/from a file (--state-save / --state-load). Writes, per GPU layer in order, the first n_pos
    // positions of K and V (attention layers, in the cache format: iso3 400 B or FP32 4 KiB per position) or the conv and
    // DeltaNet state (DeltaNet layers). `io(ptr, bytes)` writes or reads one contiguous host block. Not for the token loop.
    void save_state(int n_pos, const std::function<void(const void *, size_t)> & io) const;
    void load_state(int n_pos, const std::function<void(void *, size_t)> & io);
    size_t state_bytes(int n_pos) const;
    void clear_kv();
    // GPU LM head (cfg.head). After forward() of T rows: RMSNorm(output_norm) of every row; the normed rows go to h_rows
    // (optional, [T][5120]) and the logits of the last n_out rows (n_out <= 8) to logits ([n_out][n_vocab], host).
    void head(int T, int n_out, float * logits, float * h_rows);
    bool has_head() const;
    // GPU LM head on already-normed host rows h_host [n][5120] (n <= 8): the CPU MTP block uses it for its draft logits,
    // so a CPU draft step does not read the 1 GiB LM head from DDR4.
    void head_host(const float * h_host, int n, float * logits);
    // MTP block on the GPU (cfg.mtp). concat_host [T][10240] = [rmsnorm(emb, enorm) | rmsnorm(h, hnorm)] (built on the host).
    // Fills the MTP KV at positions pos0 .. pos0+T-1; h_out (optional) receives shared_head_norm of the last row and logits
    // (optional) its logits through the GPU LM head.
    void mtp_forward(const float * concat_host, int T, int pos0, float * h_out, float * logits);
    bool has_mtp() const;
    size_t head_bytes() const;   // device bytes of the GPU LM-head weights

    size_t vram_bytes() const;     // device bytes owned by this object (arena plus the Q8 scratch)
    size_t weight_bytes() const;   // quantized weights and small vectors
    size_t kv_bytes() const;       // KV cache of the GPU attention layers
    int n_layers() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
