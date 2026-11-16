// Gated DeltaNet layer followed by the SwiGLU FFN, FP32 reference path (Stage 2).
// Layout: activations are [token][feature], feature index fastest. The DeltaNet matrix state of
// one V head is M[j*128 + i] with j the value index and i the key index (the same layout as the
// ggml gated_delta_net op), so the reference tensor state_predelta/new_state can be compared directly.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gguf_index.hpp"

constexpr int64_t kHidden = 5120;
constexpr int64_t kQKHeads = 16;   // ssm_n_group
constexpr int64_t kVHeads = 48;    // ssm_dt_rank
constexpr int64_t kHeadDim = 128;  // ssm_d_state == head_v_dim
constexpr int64_t kInner = 6144;   // ssm_inner_size = kVHeads * kHeadDim
constexpr int64_t kConvCh = 10240; // 2 * kQKHeads * kHeadDim + kInner
constexpr int64_t kConvK = 4;      // ssm_conv_kernel
constexpr int64_t kFFN = 17408;

struct DNWeights {
    std::vector<float> attn_norm, post_norm, ssm_norm, ssm_a, ssm_dt, conv_w;   // small vectors
    std::vector<float> W_qkv, W_gate, W_alpha, W_beta, W_out;                   // projections
    std::vector<float> W_ffn_gate, W_ffn_up, W_ffn_down;
};

struct DNState {
    std::vector<float> conv;  // [kConvCh][kConvK-1], zero at start
    std::vector<float> ssm;   // [kVHeads][kHeadDim*kHeadDim], zero at start
    DNState();
};

// Optional per-stage capture, keyed like the Stage 1 reference names (e.g. "attn_norm-0").
struct Trace {
    std::map<std::string, std::vector<float>> t;
};

float g_rms_eps();  // from the GGUF key qwen35.attention.layer_norm_rms_epsilon

DNWeights load_dn_weights(const GgufIndex & ix, int il);

// Runs layer il on T tokens. x is [T][kHidden]; out receives the layer output [T][kHidden].
// st is updated in place (conv history and recurrent state).
void dn_layer_forward(const DNWeights & w, int il, const std::vector<float> & x, int64_t T,
                      DNState & st, std::vector<float> & out, Trace * tr);
