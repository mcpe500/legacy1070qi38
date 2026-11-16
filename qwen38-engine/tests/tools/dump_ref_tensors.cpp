// Reference tensor dumper (test tool, NOT engine code).
// Based on llama.cpp examples/eval-callback. Writes the raw bytes of every graph tensor whose
// name ends in "-0" or "-1" (layers 0 and 1) plus model.input_embed, one file per tensor, and a TSV
// manifest. Build against the baseline llama.cpp build (see tests/tools/build_dump.sh).
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct dump_state {
    std::string out_dir;
    std::ofstream manifest;
    int n_dumped = 0;
    std::vector<uint8_t> buf;
    std::map<std::string, int> name_count;
};

// Which tensors to dump. Env DUMP_SUFFIXES: comma list of name suffixes (default "-0,-1").
// DUMP_LOUT_ALL=1 also dumps l_out-N for every layer. model.input_embed, result_norm and
// result_output (final logits of the last token) are always dumped.
static bool want(const char * name) {
    static std::vector<std::string> suf;
    static bool lout_all = false, init = false;
    if (!init) {
        init = true;
        const char * e = getenv("DUMP_SUFFIXES");
        std::string s = e ? e : "-0,-1";
        size_t b = 0;
        while (b <= s.size()) {
            size_t c = s.find(',', b);
            if (c == std::string::npos) c = s.size();
            if (c > b) suf.push_back(s.substr(b, c - b));
            b = c + 1;
        }
        const char * l = getenv("DUMP_LOUT_ALL");
        lout_all = l && l[0] == '1';
    }
    const std::string n(name);
    if (n == "model.input_embed" || n == "result_norm" || n == "result_output") return true;
    if (lout_all && n.rfind("l_out-", 0) == 0) return true;
    for (const auto & x : suf) {
        if (n.size() >= x.size() && n.compare(n.size() - x.size(), x.size(), x) == 0) return true;
    }
    return false;
}

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return want(t->name);
    }
    auto * st = (dump_state *) user_data;
    if (!want(t->name)) {
        return true;
    }
    const size_t n_bytes = ggml_nbytes(t);
    const uint8_t * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const uint8_t *) t->data;
    } else {
        st->buf.resize(n_bytes);
        ggml_backend_tensor_get(t, st->buf.data(), 0, n_bytes);
        data = st->buf.data();
    }
    // Graph names repeat (e.g. "norm-0" is used twice in a layer). Suffix repeats so no file is overwritten.
    const int seen = st->name_count[t->name]++;
    std::string stem = t->name;
    if (seen > 0) stem += "__" + std::to_string(seen);
    std::string fname = st->out_dir + "/" + stem + ".bin";
    std::ofstream f(fname, std::ios::binary);
    f.write((const char *) data, (std::streamsize) n_bytes);
    st->manifest << t->name << '\t' << ggml_type_name(t->type) << '\t'
                 << t->ne[0] << 'x' << t->ne[1] << 'x' << t->ne[2] << 'x' << t->ne[3] << '\t'
                 << n_bytes << '\t' << fname << '\n';
    st->n_dumped++;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    dump_state st;
    { const char * d = getenv("DUMP_DIR"); st.out_dir = d ? d : "tests/ref_tensors"; }
    st.manifest.open(st.out_dir + "/manifest.tsv");
    st.manifest << "name\ttype\tne\tbytes\tfile\n";

    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    common_batch batch = common_batch_get_one(ctx, tokens);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("decode failed\n");
        return 1;
    }
    printf("tokens=%zu dumped=%d dir=%s\n", tokens.size(), st.n_dumped, st.out_dir.c_str());
    llama_backend_free();
    return 0;
}
