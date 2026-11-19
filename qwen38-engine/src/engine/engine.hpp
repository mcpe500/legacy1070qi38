// CPU inference engine for Qwen3.8-27B (Stage 2): 64 layers, quantized weights read straight from the mmap,
// FP32 activations, static arena for every buffer, and a layer residency policy for the 16 GB RAM limit.
//
// Memory design (skill memory-budget-24gb): the model file is mapped read-only. The first layers up to a byte
// budget stay in the page cache ("resident"); the later layers are read with readahead before use and dropped
// from the page cache right after use ("streamed"), so one forward pass never needs the whole 16.5 GB resident.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "arena.hpp"
#include "gguf_index.hpp"
#include "pool.hpp"
#include "qmat.hpp"
#include "tokenizer.hpp"

struct EngineConfig {
    std::string gguf;
    int ctx_max = 4096;          // KV cache positions (hard cap 8192 on the Qwen path; 131072 on the Bonsai ternary
                                 // path with iso3 KV and the tiled attention kernel above 8192)
    int t_max = 512;             // largest batch of tokens in one forward call
    int threads = 4;
    double resident_gib = 11.0;  // CPU layers kept resident, in GiB of file bytes (layers beyond it are streamed)
    // Default mode: hybrid. Layers 0 .. gpu_layers-1 run on the GPU (0 = CPU only), iso3 KV cache on the GPU attention
    // layers, Q8_1 + __dp4a for every quantization type that has a kernel (mask below = Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL,
    // IQ3_S, IQ4_XS; Q8_0 always uses dp4a). The default layer count comes from the VRAM sweep in docs/AFK_STATUS.md.
    int gpu_layers = 32;         // -1 = auto: the largest prefix whose plan fits vram_budget_mib
    // Layer placement mode: -1 = auto (the ctx-aware byte knapsack when the model or the context needs a downshift,
    // else the contiguous prefix), 0 = always the contiguous prefix 0 .. gpu_layers-1, 1 = always the knapsack.
    // gpu_layer_list (sorted, from --gpu-layer-list) overrides both. vram_budget_mib is the cap the knapsack fills.
    int gpu_placement = -1;
    double vram_budget_mib = 7373.0;   // device memory cap of the process (CUDA context included), --vram-budget-mib
    std::vector<int> gpu_layer_list;
    bool kv_iso3 = true;
    int gpu_head = -1;           // LM head on the GPU: -1 auto (when 64 layers + head fit), 0 CPU, 1 required
    int gpu_mtp = -1;            // MTP block (blk.64) on the GPU: -1 auto (needs the GPU head and room), 0 CPU, 1 required
    uint32_t dp4a_mask = (1u << 11) | (1u << 12) | (1u << 13) | (1u << 14) | (1u << 20) | (1u << 21) | (1u << 23);
    // Prefill batches of at least stream_min_t tokens (first call, position 0) run the CPU layers on the GPU too, one layer
    // at a time through a staging buffer. 0 disables it.
    int stream_min_t = 40;
    bool pin_stream = true;      // overlapped streamed prefill from pinned (registered) mmap pages
    bool mtp_gpu_prefill = true; // MTP KV of large prefill batches computed on the GPU
    bool cuda_graph = true;      // CUDA graph replay of the GPU layers for batches of 1..4 tokens
    int mtp_steps = 2;           // 0 .. 3 draft tokens per cycle from the MTP head (blk.64); 0 = plain greedy decoding
};

class GpuLayers;

struct LayerW {
    bool is_attn = false;
    const float * attn_norm = nullptr;  // [5120]
    const float * post_norm = nullptr;  // [5120]
    QMat ffn_gate, ffn_up, ffn_down;
    // Gated DeltaNet
    QMat qkv, gate_z, alpha, beta, ssm_out;
    const float * ssm_norm = nullptr;   // [128]
    const float * ssm_a = nullptr;      // [48]
    const float * ssm_dt = nullptr;     // [48]
    const float * conv_w = nullptr;     // [10240][4]
    // Gated Attention
    QMat wq, wk, wv, wo;
    const float * q_norm = nullptr;     // [256]
    const float * k_norm = nullptr;     // [256]
    // Residency
    uint64_t span_begin = 0, span_end = 0;  // file offsets (page aligned) covering every tensor of the layer
    bool resident = true;
};

class Engine {
public:
    explicit Engine(const EngineConfig & cfg);
    ~Engine();

    // Processes T tokens at positions pos0 .. pos0+T-1. State (KV cache, DeltaNet) of earlier calls is kept.
    // If logits != nullptr it receives the n_vocab logits of the last token.
    void forward(const int32_t * tokens, int T, int pos0, float * logits);

    // Like forward(), for speculative verification: logits of the last n_out rows go to logits[n_out][n_vocab] (n_out = 0 for none)
    // and, when h_rows != nullptr, the output-normed hidden state of every row to h_rows[T][5120] (the MTP head input).
    void forward_ex(const int32_t * tokens, int T, int pos0, float * logits, int n_out, float * h_rows);

    // MTP draft head (block 64). Batch of T tokens at positions pos0.. with the hidden rows h_in[T][5120] of the previous
    // positions (see src/engine/main.cpp for the pairing). Fills the MTP KV; the last row's hidden state (after the head norm)
    // goes to h_out and its logits to logits (either may be null).
    void mtp_forward(const int32_t * tokens, const float * h_in, int T, int pos0, float * h_out, float * logits);
    // Prefill of the MTP KV only (no output): on the GPU when prefill streaming is available and T >= stream_min_t, else mtp_forward.
    void mtp_prefill(const int32_t * tokens, const float * h_in, int T, int pos0);
    bool has_mtp() const { return mtp_ok_; }
    // First position the MTP attention looks at (the MTP KV below it was not filled when the prefill window is used).
    void set_mtp_kv_start(int p) { mtp_kv_start_ = p < 0 ? 0 : p; }
    const EngineConfig & config() const { return cfg_; }   // gpu_layers holds the resolved count (also after auto)

    // ---- Model state to/from a file (--state-save / --state-load): the first n_pos positions of every KV cache (GPU
    // layers in their VRAM format, CPU layers FP32, MTP layer) and every DeltaNet conv / recurrent state, in layer order.
    // `io(ptr, bytes)` writes or reads one contiguous block. Called with the engine idle, never in the token loop.
    void save_state(int n_pos, const std::function<void(const void *, size_t)> & io) const;
    void load_state(int n_pos, const std::function<void(void *, size_t)> & io);
    size_t state_bytes(int n_pos) const;
    // Zeroes every KV cache (GPU and CPU). Depth benchmark only: zero K/V = an empty but full-length attention span.
    void clear_kv();
    // Identity of the loaded model file and of the state layout (GPU layer count, KV format): a state file is only
    // accepted by an engine with the same fingerprint.
    uint64_t state_fingerprint() const;

    // DeltaNet state snapshot and rollback (GPU and CPU layers). With the snapshot on, a forward() of at most kRollT tokens keeps
    // the state from before the call; dn_rollback(keep) leaves the state as if only the first `keep` tokens had been processed.
    static constexpr int kRollT = 5;   // verification batch of up to 4 drafts + 1
    void dn_snapshot(bool on);
    void dn_rollback(int keep);

    // Zeroes the recurrent state and the position counter; the KV cache is overwritten as positions advance.
    void reset();

    // Called after each layer with the residual stream x[T][5120] (for parity tests). Not used in the token loop.
    std::function<void(int il, const float * x, int T)> tap;

    GgufIndex ix;
    Tokenizer tok;
    int64_t n_vocab = 0;
    int n_resident_layers = 0;
    double resident_gib_actual = 0, streamed_gib = 0;
    size_t arena_used() const { return arena_->used(); }
    // GPU side (0 when CPU only): device bytes owned by the GPU layer runner, and the device memory in use by this process.
    size_t gpu_arena_bytes() const;
    size_t gpu_weight_bytes() const;
    size_t gpu_kv_bytes() const;
    double vram_used_mib() const;      // total minus free device memory, includes the CUDA context
    // Wall time of the sections of the last forward() call, in seconds.
    double last_gpu_s = 0, last_cpu_s = 0, last_head_s = 0;

private:
    // ssm_out of a Gated DeltaNet layer reads its value heads in grouped order (prism.hadamard.gdn_v_grouped=1):
    // permute attn_o_ [v_head][128] with v' = 3 * (v % 16) + v / 16 into the rotated scratch buffer `dst`.
    static void gdn_v_permute(const float * src, float * dst, int64_t T);
    void dn_layer(int il, const LayerW & L, int T);
    void attn_layer(int il, const LayerW & L, int T, int pos0);
    void ffn(const LayerW & L, int T);
    void dn_conv_rec(int di, const LayerW & L, const float * qkv, const float * z, const float * gate, const float * beta, int T,
                     float * out);
    void forward_stream(int T, int pos0);
    void carve_buffers(Arena & a, int g0, bool roll);   // every host buffer; run on a measuring arena first
    size_t arena_bytes_need_ = 0;
    bool head_on_ = false;     // LM head runs on the GPU (GpuLayers::head)
    bool mtp_gpu_ = false;     // MTP block runs on the GPU (GpuLayers::mtp_forward)
public:
    bool gpu_head_on() const { return head_on_; }
    bool gpu_mtp_on() const { return mtp_gpu_; }
private:
    // y = W . hadamard(x) for a ternary (PTQ1_0 / Q2_0) weight, else y = W . x. `width` is the input dimension
    // the rotation applies to (5120, 6144 or 17408) and picks the sign vector and the rotated scratch buffer.
    void tmatmul(const QMat & W, const float * x, int64_t T, float * y, int width, uint8_t * act);
    void hint_layer(int il, bool willneed);   // readahead / drop for the streaming policy
    void check_memory(int il) const;

    EngineConfig cfg_;
    std::unique_ptr<Arena> arena_;
    std::unique_ptr<Pool> pool_;
    std::unique_ptr<GpuLayers> gpu_;
    std::vector<LayerW> layers_;
    QMat tok_embd_, lm_head_;
    const float * out_norm_ = nullptr;
    float eps_ = 1e-6f;

    // Scratch (all from the arena).
    float *x_ = nullptr, *xn_ = nullptr, *proj_ = nullptr;
    float *bufA_ = nullptr, *bufB_ = nullptr, *conv_silu_ = nullptr, *attn_o_ = nullptr;
    float *alpha_ = nullptr, *betar_ = nullptr, *gate_ = nullptr, *beta_ = nullptr;
    float *fg_ = nullptr, *fu_ = nullptr, *qbuf_ = nullptr, *kbuf_ = nullptr, *vbuf_ = nullptr;
    uint8_t * act_ = nullptr;
    float * sc_ = nullptr;    // attention scores [threads][kQG][ctx_max], one slab per pool slot (64K ctx does not fit the stack)
    float * hlast_ = nullptr;
    float * hn_ = nullptr;      // [8][5120] normed rows when forward_ex has no h_rows buffer
    // RoPE angle tables [ctx_max][kRope/2], generated once at load with the same accumulation chain as the old
    // per-call computation (theta = pos, then repeated theta_scale multiplies) so every value is bit-identical.
    float * rope_cos_ = nullptr;
    float * rope_sin_ = nullptr;
    // MTP head
    LayerW mtp_L_;
    QMat mtp_eh_;
    const float *mtp_enorm_ = nullptr, *mtp_hnorm_ = nullptr, *mtp_head_norm_ = nullptr;
    bool mtp_ok_ = false;
    int mtp_kv_start_ = 0;
    // DeltaNet rollback on the CPU layers (indexed like conv_state_; null for layers that run on the GPU).
    bool dn_save_ = false;
    std::vector<float *> snap_conv_, snap_ssm_, sv_qkv_, sv_z_, sv_gate_, sv_beta_;
    // Persistent state.
    std::vector<float *> conv_state_, ssm_state_;  // per DeltaNet layer
    std::vector<float *> kcache_, vcache_;         // per attention layer, [ctx_max][4][256]
    // Ternary Bonsai mode (auto-detected: any tensor of ggml type 42 or 143): weights live in the Hadamard-rotated
    // basis of the prism.hadamard.* metadata (block 1024, explicit signs, normalized Sylvester-Walsh), the token
    // embedding is stored pre-rotated (inverse_weight_names) and there is no MTP block.
    bool ternary_mode_ = false;
    std::vector<float> signs_;              // sign_values: [5120] ++ [6144] ++ [17408] (+1/-1 per value)
    const float * signs_5120_ = nullptr, * signs_6144_ = nullptr, * signs_17408_ = nullptr;
    float * rot5120_ = nullptr, * rot6144_ = nullptr, * rot17408_ = nullptr;  // rotated activations (arena scratch)
    bool gdn_v_grouped_ = false;
};
