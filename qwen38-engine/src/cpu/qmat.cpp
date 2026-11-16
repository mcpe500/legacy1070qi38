#include "qmat.hpp"

#include <cstdlib>
#include <stdexcept>

#include "ggml-cpu.h"
#include "ggml.h"

void qmat_init() {
    static bool done = false;
    if (!done) {
        ggml_cpu_init();
        done = true;
    }
}

QMat make_qmat(const GgufIndex & ix, const std::string & name) {
    const GgufTensor & t = ix.get(name);
    if (t.ne[2] != 1 || t.ne[3] != 1) throw std::runtime_error("qmat: tensor is not 2-D: " + name);
    QMat m;
    m.p = ix.data(t);
    m.type = static_cast<int>(t.type);
    m.in = t.ne[0];
    m.out = t.ne[1];
    m.row_bytes = ggml_row_size(static_cast<enum ggml_type>(t.type), t.ne[0]);
    const ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(static_cast<enum ggml_type>(m.type));
    if (!tc || !tc->vec_dot) throw std::runtime_error("qmat: no vec_dot for type of " + name);
    const int64_t blck = ggml_blck_size(static_cast<enum ggml_type>(tc->vec_dot_type));
    if (m.in % blck != 0) throw std::runtime_error("qmat: in not a multiple of the block size for " + name);
    return m;
}

size_t qmat_act_bytes(int wtype, int64_t in, int64_t T) {
    const ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(static_cast<enum ggml_type>(wtype));
    return ggml_row_size(tc->vec_dot_type, in) * static_cast<size_t>(T);
}

void qmatmul(Pool & pool, const QMat & W, const float * x, int64_t T, float * y, uint8_t * act) {
    const ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(static_cast<enum ggml_type>(W.type));
    const ggml_type_traits_cpu * ta = ggml_get_type_traits_cpu(tc->vec_dot_type);
    const size_t arow = ggml_row_size(tc->vec_dot_type, W.in);
    for (int64_t t = 0; t < T; ++t) ta->from_float(x + t * W.in, act + t * arow, W.in);

    // 128 rows per chunk: measured 287 ms/token for the 32 CPU layers against 298 ms with 8 rows (fewer atomic hand-outs, longer
    // sequential runs for the hardware prefetcher). 512 was slower (310 ms, load imbalance); explicit _mm_prefetch was 30 % slower.
    // QWEN_QM_GRAIN overrides it (experiments only). Each row is computed the same way whatever the chunk, so results are identical.
    static const int64_t grain = [] { const char * e = std::getenv("QWEN_QM_GRAIN"); return e ? std::atoll(e) : 128; }();
    auto work = [&](int64_t b, int64_t e) {
        for (int64_t o = b; o < e; ++o) {
            const uint8_t * row = W.p + static_cast<size_t>(o) * W.row_bytes;
            for (int64_t t = 0; t < T; ++t) {
                tc->vec_dot(static_cast<int>(W.in), y + t * W.out + o, 0, row, 0, act + t * arow, 0, 1);
            }
        }
    };
    pool.parallel_for(W.out, grain, work);
}

void qmat_row_to_f32(const QMat & W, int64_t row, float * out) {
    const ggml_type_traits * tr = ggml_get_type_traits(static_cast<enum ggml_type>(W.type));
    tr->to_float(W.p + static_cast<size_t>(row) * W.row_bytes, out, W.in);
}
