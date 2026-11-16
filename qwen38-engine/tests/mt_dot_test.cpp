// Self-test for the multi-token W4A8 kernel (Point A): mt_dot vs ggml vec_dot on random quantized matrices.
// The integer dot products must agree exactly in exact arithmetic; the accepted difference comes only from the
// float scale-epilogue order, so the bar is a tiny relative error against the reference row magnitude.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

#include "mt_dot.hpp"
#include "pool.hpp"

int main() {
    qmat_init();
    Pool pool(4);
    const int64_t in = 512, out = 8, T = 5;
    struct TC { int wtype; const char * name; };
    const TC cases[] = {{8, "Q8_0"}, {12, "Q4_K"}, {13, "Q5_K"}, {14, "Q6_K"}, {20, "IQ4_NL"}, {23, "IQ4_XS"}};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
    int fails = 0;
    for (const TC & c : cases) {
        if (!mt_dot_supported(c.wtype)) {
            std::printf("%-8s not supported (skipped)\n", c.name);
            continue;
        }
        const size_t wrow_bytes = ggml_row_size(static_cast<enum ggml_type>(c.wtype), in);
        std::vector<float> wsrc(static_cast<size_t>(in * out));
        for (float & v : wsrc) v = ud(rng);
        std::vector<uint8_t> wq(wrow_bytes * static_cast<size_t>(out));
        ggml_quantize_chunk(static_cast<enum ggml_type>(c.wtype), wsrc.data(), wq.data(), 0, static_cast<int>(out),
                            in, nullptr);
        QMat W;
        W.p = wq.data();
        W.type = c.wtype;
        W.in = in;
        W.out = out;
        W.row_bytes = wrow_bytes;
        const ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(static_cast<enum ggml_type>(c.wtype));
        const ggml_type_traits_cpu * ta = ggml_get_type_traits_cpu(tc->vec_dot_type);
        const size_t arow = ggml_row_size(tc->vec_dot_type, in);
        std::vector<uint8_t> act(qmat_act_bytes(c.wtype, in, T));
        std::vector<float> x(static_cast<size_t>(in * T));
        for (float & v : x) v = ud(rng);
        for (int64_t t = 0; t < T; ++t) ta->from_float(x.data() + t * in, act.data() + t * arow, in);
        std::vector<float> ref(static_cast<size_t>(out * T), 0.0f), mt(static_cast<size_t>(out * T), 0.0f);
        for (int64_t o = 0; o < out; ++o) {
            const uint8_t * row = W.p + static_cast<size_t>(o) * W.row_bytes;
            for (int64_t t = 0; t < T; ++t) {
                tc->vec_dot(static_cast<int>(in), ref.data() + t * out + o, 0, row, 0, act.data() + t * arow, 0, 1);
            }
        }
        mt_dot(pool, W, act.data(), static_cast<int64_t>(arow), T, mt.data(), 2);
        double worst = 0.0, nrm = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double diff = std::fabs(static_cast<double>(ref[i]) - mt[i]);
            if (diff > worst) worst = diff;
            nrm += static_cast<double>(ref[i]) * ref[i];
        }
        nrm = std::sqrt(nrm / static_cast<double>(ref.size()));
        const bool ok = worst <= 1e-4 * (nrm > 0 ? nrm : 1.0);
        std::printf("%-8s worst_abs=%.3e row_rms=%.3e %s\n", c.name, worst, nrm, ok ? "PASS" : "FAIL");
        fails += ok ? 0 : 1;
    }
    std::printf("MT-DOT TEST %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
