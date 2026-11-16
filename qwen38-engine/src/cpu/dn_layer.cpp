#include "dn_layer.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "cpu_ops.hpp"

namespace {

std::string key(const char * stem, int il) { return std::string(stem) + "-" + std::to_string(il); }

void capture(Trace * tr, const std::string & name, const std::vector<float> & v) {
    if (tr) tr->t[name] = v;
}

}  // namespace

DNState::DNState()
    : conv(static_cast<size_t>(kConvCh * (kConvK - 1)), 0.0f),
      ssm(static_cast<size_t>(kVHeads * kHeadDim * kHeadDim), 0.0f) {}

float g_rms_eps() { return 1e-6f; }  // checked against GGUF key in tests (see layer01_parity)

DNWeights load_dn_weights(const GgufIndex & ix, int il) {
    auto L = [&](const char * stem) { return "blk." + std::to_string(il) + "." + stem; };
    DNWeights w;
    w.attn_norm = dequant_tensor(ix, L("attn_norm.weight"));
    w.post_norm = dequant_tensor(ix, L("post_attention_norm.weight"));
    w.ssm_norm = dequant_tensor(ix, L("ssm_norm.weight"));
    w.ssm_a = dequant_tensor(ix, L("ssm_a"));
    w.ssm_dt = dequant_tensor(ix, L("ssm_dt.bias"));
    w.conv_w = dequant_tensor(ix, L("ssm_conv1d.weight"));
    w.W_qkv = dequant_tensor(ix, L("attn_qkv.weight"));
    w.W_gate = dequant_tensor(ix, L("attn_gate.weight"));
    w.W_alpha = dequant_tensor(ix, L("ssm_alpha.weight"));
    w.W_beta = dequant_tensor(ix, L("ssm_beta.weight"));
    w.W_out = dequant_tensor(ix, L("ssm_out.weight"));
    w.W_ffn_gate = dequant_tensor(ix, L("ffn_gate.weight"));
    w.W_ffn_up = dequant_tensor(ix, L("ffn_up.weight"));
    w.W_ffn_down = dequant_tensor(ix, L("ffn_down.weight"));
    return w;
}

void dn_layer_forward(const DNWeights & w, int il, const std::vector<float> & x, int64_t T,
                      DNState & st, std::vector<float> & out, Trace * tr) {
    const float eps = g_rms_eps();
    const int64_t Hk = kQKHeads, Hv = kVHeads, D = kHeadDim;
    out.assign(static_cast<size_t>(T * kHidden), 0.0f);

    // 1. Pre-attention norm (unweighted "norm" then weighted "attn_norm").
    std::vector<float> norm_pre(static_cast<size_t>(T * kHidden));
    std::vector<float> an(static_cast<size_t>(T * kHidden));
    for (int64_t t = 0; t < T; ++t) {
        rmsnorm(&x[t * kHidden], kHidden, nullptr, eps, &norm_pre[t * kHidden]);
        rmsnorm(&x[t * kHidden], kHidden, w.attn_norm.data(), eps, &an[t * kHidden]);
    }
    capture(tr, key("norm", il), norm_pre);
    capture(tr, key("attn_norm", il), an);

    // 2. Projections.
    std::vector<float> qkv(static_cast<size_t>(T * kConvCh));
    std::vector<float> z(static_cast<size_t>(T * kInner));
    std::vector<float> alpha(static_cast<size_t>(T * Hv)), beta_raw(static_cast<size_t>(T * Hv));
    for (int64_t t = 0; t < T; ++t) {
        const float * a = &an[t * kHidden];
        matvec(w.W_qkv.data(), kHidden, kConvCh, a, &qkv[t * kConvCh]);
        matvec(w.W_gate.data(), kHidden, kInner, a, &z[t * kInner]);
        matvec(w.W_alpha.data(), kHidden, Hv, a, &alpha[t * Hv]);
        matvec(w.W_beta.data(), kHidden, Hv, a, &beta_raw[t * Hv]);
    }
    capture(tr, key("linear_attn_qkv_mixed", il), qkv);
    capture(tr, key("z", il), z);
    capture(tr, key("alpha", il), alpha);

    // 3. Gate and beta. gate = softplus(alpha + dt_bias) * ssm_a (ssm_a holds -exp(A_log)).
    std::vector<float> beta(static_cast<size_t>(T * Hv)), gate(static_cast<size_t>(T * Hv));
    std::vector<float> a_sp(static_cast<size_t>(T * Hv));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < Hv; ++h) {
            const size_t k = static_cast<size_t>(t * Hv + h);
            beta[k] = sigmoidf_(beta_raw[k]);
            a_sp[k] = softplus_(alpha[k] + w.ssm_dt[h]);
            gate[k] = a_sp[k] * w.ssm_a[h];
        }
    }
    capture(tr, key("beta_sigmoid", il), beta);
    capture(tr, key("a_softplus", il), a_sp);
    capture(tr, key("gate", il), gate);

    // 4. Causal depthwise conv (kernel 4) over [state(3) | qkv(T)] per channel, then SiLU.
    //    conv_input[c][k] for k in [0, 3+T): state for k < 3, else qkv[k-3][c]. Layout [c][k].
    const int64_t NC = kConvK - 1 + T;
    std::vector<float> conv_in(static_cast<size_t>(kConvCh * NC));
    for (int64_t c = 0; c < kConvCh; ++c) {
        for (int64_t k = 0; k < kConvK - 1; ++k) conv_in[c * NC + k] = st.conv[c * (kConvK - 1) + k];
        for (int64_t t = 0; t < T; ++t) conv_in[c * NC + (kConvK - 1) + t] = qkv[t * kConvCh + c];
    }
    std::vector<float> conv_raw(static_cast<size_t>(T * kConvCh)), conv_silu(static_cast<size_t>(T * kConvCh));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < kConvCh; ++c) {
            float s = 0.0f;
            for (int64_t i = 0; i < kConvK; ++i) s += conv_in[c * NC + t + i] * w.conv_w[c * kConvK + i];
            conv_raw[t * kConvCh + c] = s;
            conv_silu[t * kConvCh + c] = silu_(s);
        }
    }
    // New conv state: the last kConvK-1 columns of conv_in.
    for (int64_t c = 0; c < kConvCh; ++c)
        for (int64_t k = 0; k < kConvK - 1; ++k) st.conv[c * (kConvK - 1) + k] = conv_in[c * NC + T + k];
    capture(tr, key("conv_input", il), conv_in);
    capture(tr, key("conv_output_raw", il), conv_raw);
    capture(tr, key("conv_output_silu", il), conv_silu);

    // 5. Split q (16 heads), k (16 heads), v (48 heads). L2-normalize q and k per head.
    //    q_head h uses k/q head (h % 16) (tiled broadcast, as in ggml gated_delta_net).
    std::vector<float> qn(static_cast<size_t>(T * Hk * D)), kn(static_cast<size_t>(T * Hk * D));
    std::vector<float> vv(static_cast<size_t>(T * Hv * D));
    for (int64_t t = 0; t < T; ++t) {
        const float * cs = &conv_silu[t * kConvCh];
        std::memcpy(&qn[t * Hk * D], cs, sizeof(float) * Hk * D);
        std::memcpy(&kn[t * Hk * D], cs + Hk * D, sizeof(float) * Hk * D);
        std::memcpy(&vv[t * Hv * D], cs + 2 * Hk * D, sizeof(float) * Hv * D);
        for (int64_t h = 0; h < Hk; ++h) {
            l2norm_inplace(&qn[(t * Hk + h) * D], D, eps);
            l2norm_inplace(&kn[(t * Hk + h) * D], D, eps);
        }
    }

    // 6. Gated delta rule recurrence (K=1), per V head, token by token.
    //    decay: M *= exp(g); delta_j = (v_j - sum_i M[j][i] k_i) * beta; M[j][i] += delta_j * k_i;
    //    out_j = (1/sqrt(D)) * sum_i M[j][i] q_i.
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> attn(static_cast<size_t>(T * Hv * D));
    for (int64_t h = 0; h < Hv; ++h) {
        float * M = &st.ssm[static_cast<size_t>(h * D * D)];
        const int64_t hk = h % Hk;
        for (int64_t t = 0; t < T; ++t) {
            const float * q = &qn[(t * Hk + hk) * D];
            const float * k = &kn[(t * Hk + hk) * D];
            const float * v = &vv[(t * Hv + h) * D];
            const float b = beta[t * Hv + h];
            const float ex = std::exp(gate[t * Hv + h]);
            for (int64_t j = 0; j < D * D; ++j) M[j] *= ex;
            for (int64_t j = 0; j < D; ++j) {
                float sum = 0.0f;
                for (int64_t i = 0; i < D; ++i) sum += M[j * D + i] * k[i];
                const float delta = (v[j] - sum) * b;
                for (int64_t i = 0; i < D; ++i) M[j * D + i] += k[i] * delta;
            }
            for (int64_t j = 0; j < D; ++j) {
                float sum = 0.0f;
                for (int64_t i = 0; i < D; ++i) sum += M[j * D + i] * q[i];
                attn[(t * Hv + h) * D + j] = sum * scale;
            }
        }
    }
    capture(tr, key("attn_output", il), attn);
    capture(tr, key("new_state", il), st.ssm);

    // 7. Gated norm per head: RMSNorm(out) (unweighted, captured as the Stage 1 "norm" tensor),
    //    then * ssm_norm weight, then * SiLU(z).
    std::vector<float> gn(static_cast<size_t>(T * kInner)), gn0(static_cast<size_t>(T * kInner)), fo(static_cast<size_t>(T * kInner));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < Hv; ++h) {
            const float * o = &attn[(t * Hv + h) * D];
            rmsnorm(o, D, nullptr, eps, &gn0[(t * Hv + h) * D]);
            for (int64_t d = 0; d < D; ++d) gn[(t * Hv + h) * D + d] = gn0[(t * Hv + h) * D + d] * w.ssm_norm[d];
        }
    }
    capture(tr, key("norm", il) + "__1", gn0);  // Stage 1 dump name for the unweighted gated norm
    for (int64_t i = 0; i < T * kInner; ++i) fo[i] = gn[i] * silu_(z[i]);
    capture(tr, key("final_output", il), fo);

    // 8. Output projection and residual.
    std::vector<float> lao(static_cast<size_t>(T * kHidden)), res(static_cast<size_t>(T * kHidden));
    for (int64_t t = 0; t < T; ++t) matvec(w.W_out.data(), kInner, kHidden, &fo[t * kInner], &lao[t * kHidden]);
    capture(tr, key("linear_attn_out", il), lao);
    for (int64_t i = 0; i < T * kHidden; ++i) res[i] = x[i] + lao[i];
    capture(tr, key("attn_residual", il), res);

    // 9. FFN: pre-norm (unweighted, named norm__2), post_attention_norm, SwiGLU, down, residual.
    std::vector<float> fn_pre(static_cast<size_t>(T * kHidden)), fn(static_cast<size_t>(T * kHidden));
    for (int64_t t = 0; t < T; ++t) {
        rmsnorm(&res[t * kHidden], kHidden, nullptr, eps, &fn_pre[t * kHidden]);
        rmsnorm(&res[t * kHidden], kHidden, w.post_norm.data(), eps, &fn[t * kHidden]);
    }
    capture(tr, key("norm", il) + "__2", fn_pre);
    capture(tr, key("attn_post_norm", il), fn);

    std::vector<float> fg(static_cast<size_t>(T * kFFN)), fu(static_cast<size_t>(T * kFFN));
    std::vector<float> fs(static_cast<size_t>(T * kFFN));
    for (int64_t t = 0; t < T; ++t) {
        matvec(w.W_ffn_gate.data(), kHidden, kFFN, &fn[t * kHidden], &fg[t * kFFN]);
        matvec(w.W_ffn_up.data(), kHidden, kFFN, &fn[t * kHidden], &fu[t * kFFN]);
    }
    for (int64_t i = 0; i < T * kFFN; ++i) fs[i] = silu_(fg[i]) * fu[i];
    capture(tr, key("ffn_gate", il), fg);
    capture(tr, key("ffn_up", il), fu);
    capture(tr, key("ffn_swiglu", il), fs);

    std::vector<float> fo2(static_cast<size_t>(T * kHidden));
    for (int64_t t = 0; t < T; ++t) matvec(w.W_ffn_down.data(), kFFN, kHidden, &fs[t * kFFN], &fo2[t * kHidden]);
    capture(tr, key("ffn_out", il), fo2);
    for (int64_t i = 0; i < T * kHidden; ++i) out[i] = res[i] + fo2[i];
    capture(tr, key("l_out", il), out);
}
